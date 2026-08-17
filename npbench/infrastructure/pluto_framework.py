# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
"""Pluto polyhedral column for NPBench.

Pluto is a source-to-source polyhedral optimizer: ``polycc`` reads a C translation
unit carrying a ``#pragma scop`` region and writes a tiled/parallelized one. It
invokes no compiler, so compiling and calling the result is this module's job --
which is what makes this column a BUILD PATH rather than a flag preset.

The scop that gets transformed is the ORIGINAL PolyBench/C 4.2.1 kernel, tracked
beside each benchmark as ``<module>_pluto_reference.c``. NPBench's own kernels are
NumPy ports of those same PolyBench kernels, so the pair measures the same
computation from the two sources it actually has: the array-expression port and
the loop nest Pluto's model was designed for.

Correctness is NOT assumed from that shared ancestry. Every column NPBench runs is
validated against its NumPy reference (``infrastructure.test.Test.run``), and for
this column that check is load-bearing: the ports and the originals disagree on
loop-count conventions in the time-stepped stencils (see :data:`ARG_OVERRIDES`),
and a positional ctypes call cannot detect a permuted argument list on its own.

Pipeline per benchmark, cached on mtime under :func:`build_root`:

    <module>_pluto_reference.c            tracked PolyBench scop
      -> polycc --tile --parallel        POLYCC_ARGS (clan frontend)
      -> <module>_pluto.c                 transformed, same signature
      -> clang -O3 -march=native -fopenmp -shared
      -> lib<module>_pluto.so             exports <module>_fp64
      -> ctypes call in the C signature's own argument order

Requires ``polycc`` and ``clang`` on PATH; see ``slurm/npbench-env.sh``.
"""
import ctypes
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

import numpy as np

from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict, List, Sequence, Tuple

#: How ``polycc`` is invoked, and why each flag is there.
#:
#: * (no ``--pet``) -- polycc's DEFAULT frontend, ``clan``, is used deliberately. This column
#:                     originally passed ``--pet`` on the belief that clan rejects the tracked
#:                     scops' ``int64_t`` counters. Measured: it does not. clan extracts 22 of
#:                     the 23 scops (only ``adi``, which is excluded on other grounds, fails),
#:                     with the correct statement count on every one.
#:
#:                     ``--pet`` costs four kernels, through one Pluto defect and one of its
#:                     downstream effects. pet emits a KILL statement for every variable declared
#:                     inside the function; ``pet_to_pluto.cpp:mark_trivial_dead_code()`` then
#:                     deletes every statement writing a killed NAME -- "a HACK to get rid of old
#:                     IV init's and increments", in its own comment, and an unconditional name
#:                     match rather than liveness analysis. So ``gramschmidt`` loses ``nrm``'s two
#:                     writes (5 statements of 7) and reads it uninitialized; ``durbin`` loses 3
#:                     of 10; and ``ludcmp`` loses its accumulator's writes, leaving a degenerate
#:                     dependence graph on which Pluto aborts in ``pluto_auto_transform``. clan
#:                     emits no kills and extracts all three intact (7, 10 and 12 statements).
#:                     ``--pet`` additionally miscompiles ``nussinov``'s reversed ``i`` loop into
#:                     ``table[-(N-2)][...]`` -- row -58 at N=60, ~14 KB before the buffer, which
#:                     ASan reports as a SEGV at the generated line; clan emits no such index.
#:
#:                     Set ``NPBENCH_PLUTO_FRONTEND=pet`` to reproduce any of the above.
#: * ``--tile``     -- off by default in polycc. An untiled Pluto column measures
#:                     almost nothing Pluto is for.
#: * ``--parallel`` -- also off by default. Without it polycc marks no loop parallel
#:                     and emits no ``#pragma omp parallel for``.
#: * ``--codegen-context=1`` -- "parameters are at least as much as 1". Without it Pluto also
#:                     generates the degenerate cases where a size parameter is zero or negative,
#:                     and the guards separating them carry coefficients that OVERFLOW the loop
#:                     counter type. Measured: ``bicg`` emits ``if (9223372036854775808*N >= -M+1)``
#:                     (2^63, one past int64 max) and ``fdtd_2d`` a 29-digit literal; the first
#:                     compiles with a warning, is undefined at run time, and skips the loop it
#:                     guards, so the kernel returns its untouched input very fast. Both disappear
#:                     with the context set. It is not a tuning knob: every NPBench preset passes
#:                     positive sizes, and a PolyBench kernel is undefined for non-positive ones,
#:                     so this states a fact about the call rather than assuming one away.
POLYCC_ARGS: Tuple[str, ...] = ("--tile", "--parallel", "--codegen-context=1")

#: Frontend override, for reproducing the pet-specific defects above: ``NPBENCH_PLUTO_FRONTEND=pet``.
_FRONTEND = os.environ.get("NPBENCH_PLUTO_FRONTEND", "").strip().lower()
if _FRONTEND == "pet":
    POLYCC_ARGS = ("--pet", ) + POLYCC_ARGS

#: Whether to lift scop-local scratch into parameters (:func:`lift_scop_locals`). It works around
#: a pet-only defect, and under clan it would only cost parallelism -- a lifted scalar is a shared
#: cell, which Pluto must then serialize on -- so it defaults ON for pet and OFF otherwise.
_LIFT_LOCALS = os.environ.get("NPBENCH_PLUTO_LIFT", "1" if _FRONTEND == "pet" else "0").strip() \
    not in ("0", "no", "false")

#: Compile flags for polycc's output. ``-fopenmp`` (clang's own libomp, shipped in the
#: LLVM prefix) rather than ``-fopenmp=libgomp``: the pragma has to actually be honoured,
#: which is not automatic across OpenMP runtimes. No fast-math -- validation against the
#: NumPy reference has to mean something.
CLANG_FLAGS: Tuple[str, ...] = (
    "-O3",
    "-march=native",
    "-fopenmp",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-fno-signed-zeros",
    "-fstrict-aliasing",
    "-fPIC",
    "-fveclib=libmvec",
)

#: pet extracts the scop with a flag-less libclang whose default aarch64 target carries
#: no ``neon`` feature, so glibc's ``<bits/math-vector.h>`` -- pulled in by ``<math.h>``,
#: which every tracked scop includes -- fails on its ``__neon_vector_type__`` SIMD
#: typedefs and the translation unit is rejected before any scop is seen. It is an
#: aarch64-only breakage. One header is shadowed with glibc's own empty SIMD stubs, for
#: the polycc subprocess only: polycc invokes no compiler, so nothing that gets MEASURED
#: is built under this environment.
PET_MATH_VECTOR_SHIM = ("/* Neutralised for pet scop extraction only. These are the empty SIMD\n"
                        "   declarations glibc's own <bits/math-vector.h> starts from. */\n"
                        "#include <bits/libm-simd-decl-stubs.h>\n")

#: polycc processes a multi-scop translation unit one scop at a time, re-parsing its OWN
#: output for the next one -- and that output opens with the ``#include <omp.h>`` polycc
#: prepends, which pet's flag-less libclang does not find. Nothing polycc emits CALLS the
#: runtime (only ``#pragma omp parallel for``), so these declarations are all a re-parse
#: needs.
PET_OMP_SHIM = ("/* Parse-only <omp.h> for pet scop re-extraction. */\n"
                "typedef struct { int __pet_shim; } omp_lock_t;\n"
                "typedef struct { int __pet_shim; } omp_nest_lock_t;\n"
                "int omp_get_thread_num(void);\n"
                "int omp_get_num_threads(void);\n"
                "int omp_get_max_threads(void);\n"
                "int omp_in_parallel(void);\n"
                "void omp_set_num_threads(int);\n"
                "double omp_get_wtime(void);\n")

#: Per-kernel corrections from NPBench's argument list to the PolyBench scop's, keyed by
#: benchmark short name then by C PARAMETER name. The value is called with the dict of
#: NPBench arguments (``input_args`` zipped with the call's actuals, plus any symbol
#: already resolved from an array extent) and returns the value to pass.
#:
#: These exist because NPBench's ports and the PolyBench originals genuinely disagree,
#: not to paper over a mapping this module could derive:
#:
#: * ``jacobi_2d`` -- PolyBench sweeps ``for (t = 0; t < TSTEPS; t++)``; NPBench's port
#:   sweeps ``for t in range(1, TSTEPS)``, one fewer. Measured, not assumed: without
#:   this the column fails validation against its own NumPy reference.
#: * ``seidel_2d`` -- PolyBench sweeps ``t <= TSTEPS - 1`` (TSTEPS sweeps); NPBench's
#:   port sweeps ``range(0, TSTEPS - 1)``.
#:
#: A kernel absent from this table is mapped by name alone: every C parameter must be an
#: NPBench input argument or an array extent, or the benchmark is declined outright.
ARG_OVERRIDES: Dict[str, Dict[str, Callable[[Dict[str, Any]], Any]]] = {
    # PolyBench sweeps `for (t = 0; t < TSTEPS; t++)`     -> TSTEPS iterations.
    # NPBench's port sweeps `for t in range(1, TSTEPS)`   -> TSTEPS - 1 iterations.
    # Passing TSTEPS-1 makes the C loop run the port's iteration count. This is derived from
    # the two loop bounds, not fitted to make validation pass.
    "jacobi_2d": {
        "TSTEPS": lambda a: a["TSTEPS"] - 1
    },
    # PolyBench sweeps `for (t = 0; t <= TSTEPS - 1; t++)` -> TSTEPS iterations.
    # NPBench's port sweeps `for t in range(0, TSTEPS - 1)` -> TSTEPS - 1 iterations.
    "seidel_2d": {
        "TSTEPS": lambda a: a["TSTEPS"] - 1
    },
    # PolyBench sweeps `for (t = 1; t <= TSTEPS; t++)`    -> TSTEPS iterations.
    # NPBench's port sweeps `for t in range(1, TSTEPS)`   -> TSTEPS - 1 iterations.
    "heat_3d": {
        "TSTEPS": lambda a: a["TSTEPS"] - 1
    },
}


class Adapter(object):
    """How one benchmark's NPBench arguments map onto its PolyBench scop's parameters.

    Needed whenever the two do not line up by name alone. Three distinct gaps, each of which
    is a property of the PORT, not of this module:

    ``symbols``
        Scop parameters that no VLA extent determines. ``durbin`` takes only rank-1 pointers,
        so nothing in its signature states ``N``; it comes from the length of ``r``.
    ``constants``
        Scop parameters the NumPy port INLINED as literals. The value here must be the literal
        the port actually uses -- it is read off the port, never chosen to make a test pass.
        Some are ABI padding the scop explicitly discards (``(void)alpha;``), noted per entry.
    ``outputs``
        Buffers the scop writes through a pointer while the port ALLOCATES AND RETURNS them.
        Each is ``(name, dtype, shape_fn, init_fn)``; ``init_fn`` builds the buffer in the same
        initial state the port's own allocation gives it, so any cell the scop happens not to
        write still compares equal instead of holding whatever ``np.empty`` found.

        This is also where a kernel's SCRATCH arrays are allocated. PolyBench/C passes every
        scratch array as a caller-allocated parameter (``kernel_atax(..., tmp)``,
        ``kernel_correlation(..., mean, stddev)``), and that is not a stylistic choice here: pet
        does not model a scratch array declared as a function LOCAL, and Pluto then emits a scop
        with the statements writing it missing entirely -- measured on ``atax``, where pet reports
        2 statements for a 4-statement scop. Declared as a parameter, all 4 appear. Scratch
        buffers are listed in ``outputs`` but left out of ``returns``: the harness has to allocate
        them, the port never sees them.
    ``returns``
        What the wrapped call returns, in the PORT'S return order.

        This one is load-bearing rather than cosmetic. ``utilities.validate`` compares with
        ``zip(ref, val)``, which TRUNCATES to the shorter list -- so a kernel whose port returns
        its result and whose ``output_args`` is empty would, if this column returned ``None``,
        be validated by comparing zero pairs and pass vacuously. Any benchmark that needs
        ``returns`` and does not declare it is declined outright; see :meth:`PlutoFramework.implementations`.
    """

    def __init__(self, symbols=None, constants=None, outputs=None, returns=None):
        self.symbols: Dict[str, Callable[[Dict[str, Any]], Any]] = symbols or {}
        self.constants: Dict[str, Any] = constants or {}
        self.outputs: Sequence[Tuple[str, Any, Callable, Callable]] = outputs or ()
        self.returns: Sequence[str] = returns or ()


#: Benchmarks whose NumPy port and PolyBench original compute DIFFERENT THINGS, with the
#: difference. These are refused before any transform is attempted, because no argument mapping can
#: reconcile them -- the only ways to make them agree are to edit NPBench's kernel (changing what
#: the suite measures) or to edit the tracked PolyBench scop (making it no longer the original).
#:
#: Recorded here rather than only in prose so that a future fix to Pluto cannot quietly turn one of
#: these into a column that runs and reports plausible, wrong numbers.
SEMANTIC_DIVERGENCE: Dict[str, str] = {
    # Empty as of the canonicalization of `adi` and `deriche`. Both entries described
    # upstream NPBench PORTING discrepancies against PolyBench/C 4.2.1 -- adi took `b`
    # from mul2 instead of mul1, deriche dropped the factor 2.0 from k's denominator --
    # and both have been corrected in the ports themselves (NumPy and DaCe alike), so
    # there is no longer a divergence for this column to refuse. The mechanism stays:
    # it is what keeps a future port drift from quietly becoming a Pluto "result".
}


#: Benchmarks where polycc's output is CORRECT SEQUENTIALLY but its OpenMP parallelization is
#: unsound -- Pluto marked a loop parallel that carries a dependence.
#:
#: This is the most dangerous failure mode in this module, and the only one that can pass
#: validation. A dropped statement returns NaN, an overflowing bound skips a loop, a negated
#: subscript segfaults; all of those are reliably visible. A race is not: it corrupts a
#: data-dependent fraction of the output, so at a small preset the kernel sometimes lands on the
#: right answer and NPBench prints SUCCESS. Observed exactly that -- two consecutive
#: `run_benchmark.py -b nussinov -f pluto -p S` invocations, unchanged binary, one SUCCESS and one
#: "Relative error: 0.0048".
#:
#: Refused up front rather than left to validation, because validation cannot be trusted to catch
#: it and a single lucky run would put a wrong number in the results database under Pluto's name.
#:
#: Every other tracked kernel was screened for this and is deterministic: each was run once
#: sequentially and three times on 32 threads, and the outputs compared exactly.
UNSOUND_PARALLELIZATION: Dict[str, str] = {
    # Rechecked against PolyBench/C 4.2.1 from scratch: `table[i][j]` reads `table[k+1][j]` for
    # k+1 in (i, j], i.e. rows BELOW i, which the decreasing `i` loop writes in later iterations.
    # The i loop therefore carries a real dependence and cannot be parallel -- the structure
    # confirms the measurement rather than the other way round.
    #
    # Configuration sweep, each validated over many runs at several sizes and thread counts:
    # --lastwriter, --rar, --nofuse, --innerpar and --second-level-tile all still race
    # (--second-level-tile looked clean at 9 runs and failed 72 of 288 under stress, which is why
    # none of these is trusted on a short test). `--multipar` survived 504 runs at N=120..900 on
    # 4..72 threads with a full-table FNV hash and never differed -- promising, but it rests on the
    # same dependence analysis that is demonstrably wrong here, so it is recorded and not adopted.
    "nussinov": ("polycc marks the tiled `i` loop `#pragma omp parallel for`, but nussinov's "
                 "`table[i][j] = max(table[i][j], table[i][k] + table[k+1][j])` reads row k+1 while "
                 "another thread is still writing it. Measured against the untransformed reference "
                 "compiled directly, N=200: 0/20 runs differ on 1 thread, 20/20 differ on 8 threads "
                 "and 20/20 on 32. The sequential result is exact, so the transformation itself is "
                 "sound and only the parallel decoration is wrong."),
}


def _zeros(dtype):
    return lambda shape, a: np.zeros(shape, dtype=dtype)


#: Per-benchmark argument adapters. Everything here was read off the NumPy port and the tracked
#: scop side by side; nothing is fitted to a validation result.
PLUTO_ADAPTERS: Dict[str, Adapter] = {
    # y = A^T (A x). The port returns the result; the scop writes it through `out` (and zeroes
    # it itself). `tmp` is a scop-local VLA, not a parameter.
    "atax":
    Adapter(outputs=[("out", np.float64, lambda a: (a["N"], ), _zeros(np.float64)),
                     # scratch: PolyBench's own kernel_atax takes tmp as a parameter
                     ("tmp", np.float64, lambda a: (a["M"], ), _zeros(np.float64))],
            returns=["out"]),
    # The port returns (r @ A, A @ p); the scop writes them as out0 (length M) and out1 (length N).
    "bicg":
    Adapter(outputs=[("out0", np.float64, lambda a: (a["M"], ), _zeros(np.float64)),
                     ("out1", np.float64, lambda a: (a["N"], ), _zeros(np.float64))],
            returns=["out0", "out1"]),
    # Every parameter is rank-1, so no extent names N; the port takes it from r's length.
    # The port allocates y with np.empty_like(r) and the scop writes all N entries.
    "durbin":
    Adapter(symbols={"N": lambda a: int(np.shape(a["r"])[0])},
            outputs=[("y", np.float64, lambda a: (a["N"], ), _zeros(np.float64))],
            returns=["y"]),
    # The port allocates Q as zeros_like(A) and R as zeros((N, N)) and returns both. R's lower
    # triangle is never written by either side, so the zero initializer is what makes them agree.
    "gramschmidt":
    Adapter(outputs=[("Q", np.float64, lambda a: (a["M"], a["N"]), _zeros(np.float64)),
                     ("R", np.float64, lambda a: (a["N"], a["N"]), _zeros(np.float64))],
            returns=["Q", "R"]),
    # imgIn is declared [W][H], so W and H come from its two extents; imgOut has the same shape.
    "deriche":
    Adapter(outputs=[("imgOut", np.float64, lambda a: (a["W"], a["H"]), _zeros(np.float64)),
                     # scratch: parameters of PolyBench's kernel_deriche
                     ("y1", np.float64, lambda a: (a["W"], a["H"]), _zeros(np.float64)),
                     ("y2", np.float64, lambda a: (a["W"], a["H"]), _zeros(np.float64))],
            returns=["imgOut"]),
    # The port inlines `stddev[stddev <= 0.1] = 1.0`; the scop takes the same rule as two
    # parameters. corr is allocated as np.eye(M) by the port -- matched exactly here, so the
    # diagonal agrees whichever side writes it.
    "correlation":
    Adapter(constants={
        "stddev_eps": 0.1,
        "stddev_replacement": 1.0
    },
            outputs=[("corr", np.float64, lambda a: (a["M"], a["M"]),
                      lambda shape, a: np.eye(shape[0], dtype=np.float64)),
                     # scratch: parameters of PolyBench's kernel_correlation
                     ("mean", np.float64, lambda a: (a["M"], ), _zeros(np.float64)),
                     ("stddev", np.float64, lambda a: (a["M"], ), _zeros(np.float64))],
            returns=["corr"]),
    # The port's `match(b1, b2)` returns 1 when b1 + b2 == 3; the scop takes that pair as
    # parameters. An int32 kernel: seq and table are both int32 in the port and in the scop.
    "nussinov":
    Adapter(constants={
        "complement_sum": 3,
        "pair_bonus": 1
    },
            outputs=[("table", np.int32, lambda a: (a["N"], a["N"]), _zeros(np.int32))],
            returns=["table"]),
    # The port allocates x and y as zeros_like(b) and returns both; A it modifies in place,
    # which is already NPBench's `output_args`. The scop writes x and y through pointers.
    "ludcmp":
    Adapter(outputs=[("x", np.float64, lambda a: (a["N"], ), _zeros(np.float64)),
                     ("y", np.float64, lambda a: (a["N"], ), _zeros(np.float64))],
            returns=["x", "y"]),
    # adi is DECLINED. Re-investigated from scratch, including the hypothesis that the NumPy or
    # DaCe side was still non-canonical. It is not: both match untransformed canonical PolyBench/C
    # across nine (N, TSTEPS) combinations -- worst relative error 1.3e-13 for NumPy and 5.6e-14
    # for DaCe, i.e. floating-point reassociation and nothing else.
    #
    # TWO separate defects sit between adi and a correct transform.
    #
    # 1. CLAN TRANSPOSES THE SCOP PARAMETERS. Clan's own scop for the statement `v[0][i] = 1.0`
    #    carries `-t+N >= 0` and `-i+TSTEPS-2 >= 0`, when the loops are `t=1..TSTEPS` and
    #    `i=1..N-1`. The two parameters are swapped in the domain constraints, so polycc emits
    #    `if (TSTEPS >= 3)` guarding spatial loops bounded by `TSTEPS-2`: at TSTEPS=1 the whole
    #    scop is skipped and u is returned untouched, and at TSTEPS > N it writes past the arrays
    #    (measured: SIGSEGV at N=8, TSTEPS=20). The trigger is that adi's parameters first appear
    #    inside non-affine casts (`1.0/(DATA_TYPE)_PB_N`) ahead of any loop; hoisting that
    #    loop-invariant scalar block above `#pragma scop` -- bit-identical to canonical at every
    #    size tested -- gives Clan the correct parameter order and correct bounds.
    #
    # 2. PLUTO ORDERS THE BACK-SUBSTITUTION BEFORE ITS PRODUCER. With (1) fixed, polycc still
    #    emits, inside one tile, the reversed `j` loop that READS p and q ahead of the forward
    #    loop that WRITES them, so v is computed from zeros. Identical wrong output under
    #    --tile/--parallel/--nofuse/--maxfuse/--lastwriter/--rar/--innerpar/--nointratileopt/
    #    --nodiamond-tile and plain polycc, on both frontends, and identical on 1, 8 and 32
    #    threads -- a wrong schedule, not a race. There is no configuration that fixes it, so
    #    the hoist in (1) is not applied: it would trade a loud failure for a quieter one.
    #
    # Validation is what stands between this and a silent wrong number, and it holds: every
    # candidate above was rejected by comparison against the untransformed canonical kernel.
    #
    # b1/b2 are ABI padding: the scop computes B1 = 2.0 and B2 = 1.0 itself, inside the scop,
    # and discards these two with `(void)`. Passed as those literals so the ABI reads as what
    # the kernel computes, though no value can reach the result. `u` is updated in place and is
    # already NPBench's `output_args`; v/p/q are scop-local scratch.
    "adi":
    Adapter(constants={
        "b1": 2.0,
        "b2": 1.0
    }),
    # alpha is ABI padding: the scop hard-codes 0.125 and discards the parameter with
    # `(void)alpha;`. Passed as 0.0 because no value can affect the result.
    "heat_3d":
    Adapter(constants={"alpha": 0.0}),
    # Same shape: the scop hard-codes PolyBench's 0.5/0.5/0.7 Courant factors -- the same
    # literals the NumPy port uses -- and discards these three parameters with `(void)`.
    # Passed as those literals rather than 0.0 so the ABI reads as what the kernel computes,
    # though no value can reach the result.
    "fdtd_2d":
    Adapter(constants={
        "ex_courant": 0.5,
        "ey_courant": 0.5,
        "hz_courant": 0.7
    }),
}

#: Prototype of the tracked scop's exported symbol. ``_fp64`` names the SYMBOL, not the
#: element type of every parameter: PolyBench fixes a ``DATA_TYPE`` per kernel and a few
#: are integer kernels (``floyd_warshall`` is ``int``), so the type is read per parameter
#: from its own declaration rather than assumed.
_PROTO = re.compile(r"void\s+(\w+)_fp64\s*\((.*?)\)\s*\{", re.S)

#: C element type -> (numpy dtype, ctypes type). The closed set the tracked scops use.
#: Anything else is declined rather than guessed at: a positional ctypes call cannot
#: detect a wrong element width, it just returns different numbers.
_ELEM_TYPES: Dict[str, Tuple[Any, Any]] = {
    "double": (np.float64, ctypes.c_double),
    "float": (np.float32, ctypes.c_float),
    "int32_t": (np.int32, ctypes.c_int32),
    "int64_t": (np.int64, ctypes.c_int64),
    "int": (np.int32, ctypes.c_int32),
}


def _element_type(decl: str) -> Tuple[Any, Any]:
    """``(numpy dtype, ctypes type)`` for a parameter declaration, or raise.

    Matched longest-first so ``int64_t`` is not read as ``int``.
    """
    for name in sorted(_ELEM_TYPES, key=len, reverse=True):
        if re.search(r"\b{}\b".format(re.escape(name)), decl):
            return _ELEM_TYPES[name]
    raise PlutoUnavailable("unsupported element type in scop parameter `{d}`".format(d=decl.strip()))

_REFERENCE_SUFFIX = "_pluto_reference.c"


class PlutoUnavailable(RuntimeError):
    """Raised when a benchmark has no tracked scop, or its scop cannot be mapped.

    ``Test.run`` treats this like any other implementation-loading failure: the column
    records nothing for the benchmark. That is the intended outcome -- a Pluto column
    that silently timed something other than polycc's output would be worse than a gap.
    """


def build_root() -> pathlib.Path:
    """Where transformed sources and shared objects are cached.

    ``NPBENCH_PLUTO_BUILD_DIR`` wins; otherwise ``$SCRATCH`` (the parallel FS handles the
    compiler's many small writes better than a repo on a home filesystem), otherwise a
    directory in the repo. Give each rank of a multi-rank job its OWN root: two ranks
    building the same kernel would write one ``lib<module>_pluto.so`` at the same time.
    """
    env = os.environ.get("NPBENCH_PLUTO_BUILD_DIR")
    if env:
        root = pathlib.Path(env)
    elif os.environ.get("SCRATCH"):
        root = pathlib.Path(os.environ["SCRATCH"]) / "npbench-pluto"
    else:
        root = pathlib.Path(__file__).parent.parent.parent.absolute() / ".pluto_build"
    root.mkdir(parents=True, exist_ok=True)
    return root


def reference_source(bench: Benchmark) -> pathlib.Path:
    """The tracked PolyBench scop for ``bench``, beside its NumPy port."""
    parent = pathlib.Path(__file__).parent.parent.parent.absolute()
    return (parent / "npbench" / "benchmarks" / bench.info["relative_path"] /
            (bench.info["module_name"] + _REFERENCE_SUFFIX))


def parse_prototype(text: str) -> Tuple[str, List[Tuple[str, str, List[str]]]]:
    """``(base, params)`` for the scop's exported symbol.

    Each param is ``(name, kind, extents)`` with kind in ``symbol`` (an ``int64_t``
    loop/extent parameter), ``array`` (a VLA parameter, whose ``extents`` name the
    symbols giving its shape), ``pointer`` (a rank-1 ``double *``) or ``scalar``.

    The ORDER is polycc's calling convention and is read from the source rather than
    re-derived: C requires a VLA parameter's extents to be declared before it, so the
    signature runs symbols, then arrays, then scalars -- not the order NPBench passes.
    polycc is source-to-source and leaves the signature alone, so the reference's
    prototype is the transformed unit's prototype.
    """
    m = _PROTO.search(text)
    if not m:
        raise PlutoUnavailable("no `void <base>_fp64(...)` prototype in the tracked scop")
    base, argstr = m.group(1), m.group(2)

    # Split on top-level commas only: a VLA extent may itself contain one.
    args, depth, cur = [], 0, ""
    for ch in argstr:
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        cur += ch
    if cur.strip():
        args.append(cur.strip())

    params: List[Tuple[str, str, List[str], Any]] = []
    for a in args:
        head = a.split("[")[0].strip()
        name = head.split()[-1].lstrip("*")
        extents = [e.strip() for e in re.findall(r"\[\s*(?:restrict\s+)?([^\]]*)\]", a)]
        extents = [e for e in extents if e and e != "restrict"]
        if extents:
            params.append((name, "array", extents, _element_type(a)))
        elif "*" in a:
            params.append((name, "pointer", [], _element_type(a)))
        elif "int64_t" in a or re.search(r"\bint\b", a):
            params.append((name, "symbol", [], _ELEM_TYPES["int64_t"]))
        else:
            params.append((name, "scalar", [], _element_type(a)))
    return base, params


#: Wall-clock ceiling on one ``polycc`` run, and on the clang compile of its output. Generous
#: relative to a healthy transform (~1.5s for gemm), and there purely to bound the pathological
#: case: a Pluto that has corrupted its heap does not always exit.
POLYCC_TIMEOUT = 300
CLANG_TIMEOUT = 300

#: Largest value a C `long long` can hold. Pluto's scheduler can produce rational coefficients
#: whose denominators blow up, and it prints them into the emitted loop bounds verbatim.
_INT64_MAX = (1 << 63) - 1


#: A scratch declaration eligible for lifting: one uninitialized data-typed local, on its own
#: line, between the prototype and `#pragma scop`. Integer declarations are deliberately NOT
#: matched -- `int i, j, k;` are the loop counters, and eliminating the statements that write
#: THOSE is the behaviour the pass below exists for and gets right.
_LOCAL_DECL = re.compile(r"^([ \t]*)(DATA_TYPE|double|float)[ \t]+([^;=]+);[ \t]*\n", re.M)


def _resolve_data_type(text: str) -> str:
    """What ``DATA_TYPE`` expands to in ``text``; PolyBench fixes it per kernel."""
    m = re.search(r"^\s*#\s*define\s+DATA_TYPE\s+(\w+)\s*$", text, re.M)
    return m.group(1) if m else "double"


def _writes_name(region: str, name: str) -> bool:
    """Whether ``region`` assigns to ``name`` anywhere, at any nesting.

    Deliberately NOT :func:`_assigned_names`, which anchors to the start of a line so that a
    ``for (i = 0; ...)`` header is not read as an assignment. That anchoring is right for
    comparing write sets, and wrong for deciding what to lift: ``deriche`` chains its
    coefficients (``a1 = a5 = k;``, ``c1 = c2 = 1;``), and an anchored scan sees only the
    leftmost target -- which would lift ``a1`` and leave ``a5`` behind in the same statement.
    """
    return re.search(r"\b{n}\b\s*(?:\[[^\]]*\]\s*)*(?:[-+*/]?=(?!=)|\+\+|--)".format(n=re.escape(name)),
                     region) is not None


def lift_scop_locals(text: str) -> Tuple[str, Dict[str, List[str]]]:
    """Rewrite scop-local scratch temporaries into caller-allocated scratch PARAMETERS.

    Returns the rewritten translation unit and ``{name: extents}`` for each variable lifted
    (``[]`` for a scalar, which becomes a one-element pointer). Returns the input unchanged
    when nothing qualifies, which is the case for 17 of the 23 tracked scops.

    WHY THIS EXISTS -- the defect it works around, named exactly:

        pluto/tool/pet_to_pluto.cpp, ``mark_trivial_dead_code()``

    pet emits a *kill* statement for every variable declared inside the function. Pluto reads
    those kills and then, in its own words,

        // Mark any other writes to the same variable name dead.
        // This is a HACK to get rid of old IV init's and increments.

    deletes every statement writing a killed NAME. That is not liveness analysis: it is an
    unconditional name match, written to drop induction-variable bookkeeping, and it takes out
    genuine computation with it. A twelve-line kernel whose local accumulator is stored to an
    output array immediately afterwards -- unambiguously live -- still loses both of its writes.

    Consequences measured here, all of which this pass removes:

      * ``gramschmidt`` loses ``nrm = 0.0`` and ``nrm += ...``; the emitted code takes
        ``sqrt()`` of an uninitialized scalar (5 statements out of 7).
      * ``durbin`` loses 3 of 10 statements the same way.
      * ``ludcmp`` does not merely lose statements -- with its accumulator's writes deleted the
        remaining dependence graph is degenerate and Pluto aborts inside ``pluto_auto_transform``
        on ``hyp_search_mode == LAZY || num_sols_left == ...``. Lifting ``w`` makes it transform.

    A function PARAMETER has no declaration inside the function, so pet emits no kill for it and
    nothing is deleted. That is the whole mechanism; it is the same shape PolyBench/C already
    uses for its own scratch (``kernel_atax(..., tmp)``).

    WHY IT IS SEMANTICS-PRESERVING. A lifted scalar becomes a one-element array cell, so every
    read and write keeps its address, its order and its dependences -- Pluto sees exactly the
    dependences the C scalar already had, and may schedule less aggressively but never more.
    Three conditions keep the rewrite honest, and a declaration failing any of them is left
    alone rather than guessed at:

      1. no initializer -- ``DATA_TYPE eps = stddev_eps;`` carries a value INTO the scop that a
         freshly allocated buffer would not have;
      2. not assigned between the declaration and ``#pragma scop``, for the same reason;
      3. assigned at least once INSIDE the scop -- a variable the scop only reads loses nothing
         to the hack, because the hack only deletes writes.

    Under (1) and (2) the C original reads an indeterminate value on any path that reads before
    writing, so zero-filling the scratch buffer is a valid realization of what the original
    already did. Every lifted kernel is still validated against the NumPy reference; this pass
    can only produce a build that fails to compile or fails to validate, never a wrong number
    that passes.
    """
    m = _PROTO.search(text)
    if not m:
        return text, {}
    elem = _resolve_data_type(text)
    head, body = text[:m.end()], text[m.end():]

    scop = _scop_region(text)

    split = body.find("#pragma scop")
    if split < 0:
        return text, {}
    pre, rest = body[:split], body[split:]

    lifted: Dict[str, List[str]] = {}
    new_pre, cursor = [], 0
    for decl in _LOCAL_DECL.finditer(pre):
        indent, _type, declarators = decl.group(1), decl.group(2), decl.group(3)
        keep, take = [], []
        for d in (d.strip() for d in declarators.split(",")):
            name = d.split("[")[0].strip()
            extents = [e.strip() for e in re.findall(r"\[([^\]]*)\]", d)]
            if _writes_name(scop, name) and not _writes_name(pre, name):
                take.append((name, extents))
            else:
                keep.append(d)
        if not take:
            continue
        new_pre.append(pre[cursor:decl.start()])
        if keep:
            new_pre.append("{i}{t} {d};\n".format(i=indent, t=decl.group(2), d=", ".join(keep)))
        cursor = decl.end()
        for name, extents in take:
            lifted[name] = extents
    if not lifted:
        return text, {}
    new_pre.append(pre[cursor:])
    body = "".join(new_pre) + rest

    # Scalars become `name[0]`. Done on the BODY only, before the signature is rebuilt, so the
    # rewrite cannot reach the parameter declarations it is about to create.
    for name, extents in lifted.items():
        if extents:
            continue
        body = re.sub(r"\b{n}\b".format(n=re.escape(name)), "{n}[0]".format(n=name), body)

    params = []
    for name, extents in lifted.items():
        if extents:
            dims = "[restrict {e}]".format(e=extents[0]) + "".join("[{e}]".format(e=e) for e in extents[1:])
            params.append("{t} {n}{d}".format(t=elem, n=name, d=dims))
        else:
            params.append("{t} *restrict {n}".format(t=elem, n=name))
    head = head.rstrip()
    assert head.endswith("{")
    head = head[:-1].rstrip().rstrip(")")
    head = "{h}, {p}) {{".format(h=head, p=", ".join(params))
    return head + body, lifted


#: An array subscript opening with a unary minus applied to a parenthesized expression or to
#: another minus. See :func:`_negated_subscripts` for why that is the defect's signature and why
#: the legitimate case (``b[-t4]``, a bare reversed iterator) is deliberately not matched.
_NEG_SUBSCRIPT = re.compile(r"\[\s*-\s*[-(]")


def _negated_subscripts(text: str) -> List[str]:
    """Array subscripts polycc emitted with one negation too many.

    A THIRD Pluto defect, specific to the ``--pet`` frontend, and the one that crashes rather
    than lying. Pluto captures each statement's source text from an isl AST in which a reverse
    loop has ALREADY been normalized -- ``for (i = N-1; i >= 0; i--)`` becomes ``c0`` with
    ``i = -c0``, so the captured text reads ``table[-c0][...]``. Pluto then runs its own
    scheduling and CLooG codegen, which normalizes the same loop again, and substitutes its
    iterator into that text without folding the sign. The two reversals compose instead of
    cancelling:

        nussinov   table[-(N-2)][(N-1)]      should be table[N-2][N-1]
        deriche    y2[t2][- -t4]             should be y2[t2][-t4]
        ludcmp     x[-(N-1)]                 should be x[N-1]

    ``-(N-2)`` is negative for every N > 2, so the kernel reads and writes whole rows BELOW its
    array. Measured with AddressSanitizer: nussinov faults on a read at an address below the
    allocation; deriche writes 376 bytes before ``y2`` and corrupts the heap, which under NPBench
    kills the interpreter mid-benchmark.

    What is matched is a subscript whose leading unary minus applies to a parenthesized
    expression or to a second minus. A correct reversed iterator prints as ``b[-t4]`` -- minus
    directly on a bare iterator name -- and is NOT matched, which is what keeps the 17 working
    kernels working. Detected on the generated source and turned into a decline; never repaired
    there, because folding the sign by hand would mean timing code polycc did not produce.
    """
    return sorted({m.group(0) for m in _NEG_SUBSCRIPT.finditer(text)})


def _int64_overflow_literals(text: str) -> List[str]:
    """Integer literals in ``text`` too large for a 64-bit signed type.

    This is a known Pluto defect, and it is DANGEROUS rather than merely broken. Sometimes the
    literal lands somewhere clang rejects outright (``fdtd_2d``: "integer literal is too large to
    be represented in any integer type"), which is loud. Sometimes it lands inside a loop guard --
    ``if (9223372036854775808*N >= -M+1)`` -- which compiles with a warning, is undefined at
    runtime, and skips the guarded loop. The kernel then returns its untouched input buffers and
    the column reports a very fast, completely wrong measurement.

    Detected on polycc's output BEFORE anything is compiled, so both cases become the same clean
    decline. This inspects the generated source to reject it; it never edits it.
    """
    bad = []
    for lit in re.findall(r"(?<![\w.])(\d{19,})(?![\w.])", text):
        if int(lit) > _INT64_MAX:
            bad.append(lit)
    return bad


#: An assignment statement's target, at the start of a line: ``out[i] =``, ``nrm +=``, ``A[i][j] -=``.
#: Anchored to the line start so a ``for (i = 0; ...)`` header is not read as an assignment to ``i``,
#: and excluding ``==`` so a comparison is not read as one either.
_ASSIGN = re.compile(r"^[ \t]*([A-Za-z_]\w*)\s*(?:\[[^;]*?\])?\s*(?:[-+*/]?=)(?!=)", re.M)


def _assigned_names(text: str) -> set:
    """Names that ``text`` assigns to at least once."""
    return set(_ASSIGN.findall(text))


def _scop_region(text: str) -> str:
    """The ``#pragma scop`` ... ``#pragma endscop`` body, or the whole text if unmarked."""
    m = re.search(r"#pragma\s+scop(.*?)#pragma\s+endscop", text, re.S)
    return m.group(1) if m else text


def _dropped_writes(reference_text: str, generated_text: str) -> List[str]:
    """Names the scop writes that polycc's output never writes.

    A SECOND known Pluto defect, and the quieter of the two. pet models a temporary declared
    inside the function -- ``atax``'s ``tmp[M]``, ``gramschmidt``'s scalar ``nrm``,
    ``deriche``'s ``y1``/``y2`` -- as scop-local, and Pluto then eliminates the statements that
    write it because nothing outside the scop reads it. But the scop's OWN later statements do,
    so the emitted code reads an uninitialized buffer and returns denormal noise or NaN.

    Nothing warns: the result compiles, runs fast, and is wrong. Comparing the two write sets is
    what turns it into a decline. Detected on the generated source; never repaired in it, because
    a repair would mean writing the missing statements back by hand and timing something polycc
    did not produce.
    """
    return sorted(_assigned_names(_scop_region(reference_text)) - _assigned_names(generated_text))


def _pet_env(scratch: pathlib.Path) -> Dict[str, str]:
    """Environment for the polycc subprocess: the two parse-only shims on C_INCLUDE_PATH."""
    shim = scratch / "pet-include"
    (shim / "bits").mkdir(parents=True, exist_ok=True)
    (shim / "bits" / "math-vector.h").write_text(PET_MATH_VECTOR_SHIM)
    (shim / "omp.h").write_text(PET_OMP_SHIM)
    env = dict(os.environ)
    existing = env.get("C_INCLUDE_PATH", "")
    env["C_INCLUDE_PATH"] = f"{shim}{os.pathsep}{existing}" if existing else str(shim)
    return env


def run_polycc(reference: pathlib.Path, out: pathlib.Path) -> None:
    """Transform ``reference`` into ``out``, or raise. Never falls back to the input.

    Publishing is atomic (transform to a temporary, then ``os.replace``) so an
    interrupted run cannot leave a truncated source that a later build would happily
    compile and time.
    """
    exe = shutil.which("polycc")
    if exe is None:
        raise PlutoUnavailable("polycc is not on PATH (source slurm/npbench-env.sh)")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Staged INSIDE the destination directory, not in the default temp dir: on Alps the build root
    # is on $SCRATCH while /tmp is a tmpfs, and `os.replace` across filesystems raises
    # EXDEV ("Invalid cross-device link"). Same directory means the publish is a same-filesystem
    # rename, which is what makes it atomic in the first place.
    with tempfile.TemporaryDirectory(prefix="npbench-polycc-", dir=str(out.parent)) as tmp:
        scratch = pathlib.Path(tmp)
        tmp_out = scratch / out.name
        try:
            proc = subprocess.run([exe, *POLYCC_ARGS, str(reference), "-o", str(tmp_out)],
                                  cwd=scratch,
                                  env=_pet_env(scratch),
                                  capture_output=True,
                                  text=True,
                                  timeout=POLYCC_TIMEOUT)
        except subprocess.TimeoutExpired:
            # Pluto can corrupt its own heap and then spin rather than exit (`nussinov`:
            # "double free or corruption (out)"). Unbounded, that consumes the rest of a
            # campaign's allocation on one kernel it was never going to transform.
            raise PlutoUnavailable("polycc did not finish within {t}s on {r}".format(t=POLYCC_TIMEOUT,
                                                                                     r=reference.name))
        if proc.returncode != 0 or not tmp_out.is_file():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
            raise PlutoUnavailable("polycc failed on {r}:\n{t}".format(r=reference.name, t="\n".join(tail)))
        text = tmp_out.read_text()
        if "#pragma omp parallel for" not in text:
            # polycc ran but marked nothing parallel. NOT a decline: Pluto is a locality
            # optimizer as well as a parallelizer, the output is still tiled, and on `durbin`
            # and `ludcmp` -- the only two kernels here that reach this -- there genuinely is no
            # parallelism to find. Both are inherently sequential recurrences, and clan and pet
            # agree on it, so a blank row would misreport an honest Pluto answer as a failure.
            #
            # It is announced rather than swallowed, because a one-thread result sitting in a
            # table beside 72-thread ones has to be labelled as such. The launcher captures this
            # line per pair, `collect_status.py` picks it up, and the report marks the row.
            print("PlutoSequential: {r}: polycc marked no loop parallel; the transformed code is "
                  "tiled but single-threaded, so this row is not comparable to the parallel "
                  "ones".format(r=reference.name))
        dropped = _dropped_writes(reference.read_text(), text)
        if dropped:
            raise PlutoUnavailable(
                "polycc dropped the statements writing {n} in {r}; the emitted code reads them "
                "uninitialized and returns wrong numbers with no diagnostic".format(n=", ".join(
                    "`{}`".format(d) for d in dropped), r=reference.name))
        overflow = _int64_overflow_literals(text)
        if overflow:
            raise PlutoUnavailable(
                "polycc emitted loop bounds with integer literals too large for int64 in {r} "
                "(e.g. {lit}); the guard they sit in is undefined at runtime, which silently "
                "skips the loop rather than computing a wrong number loudly".format(r=reference.name,
                                                                                    lit=overflow[0]))
        negated = _negated_subscripts(text)
        if negated:
            raise PlutoUnavailable(
                "polycc emitted array subscripts with one negation too many in {r} (e.g. `{s}`); "
                "the index is negative for every valid size, so the kernel reads and writes below "
                "its own arrays -- a segfault or silent heap corruption, not a wrong "
                "number".format(r=reference.name, s=negated[0]))
        os.replace(str(tmp_out), str(out))


def compile_shared(source: pathlib.Path, so: pathlib.Path, openmp: bool = True) -> None:
    """Compile polycc's output into ``so``, or raise.

    ``openmp=False`` drops ``-fopenmp`` so the ``#pragma omp parallel for`` polycc emitted is
    ignored by the compiler and the tiled nest runs on one thread. Nothing in the generated C
    changes -- it is compiled exactly as polycc wrote it -- which is what separates this from
    editing the transform. It is how a kernel whose parallel decoration is UNSOUND can still be
    measured as what Pluto actually computed; see :data:`UNSOUND_PARALLELIZATION`.
    """
    exe = shutil.which("clang")
    if exe is None:
        raise PlutoUnavailable("clang is not on PATH (source slurm/npbench-env.sh)")
    tmp_so = so.with_suffix(so.suffix + ".tmp")
    flags = CLANG_FLAGS if openmp else tuple(f for f in CLANG_FLAGS if f != "-fopenmp")
    cmd = [exe, *flags, "-shared", "-o", str(tmp_so), str(source), "-lm"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CLANG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise PlutoUnavailable("clang did not finish within {t}s on {s}".format(t=CLANG_TIMEOUT, s=source.name))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        raise PlutoUnavailable("clang failed on {s}:\n{t}".format(s=source.name, t="\n".join(tail)))
    os.replace(str(tmp_so), str(so))


def _stale(target: pathlib.Path, *sources: pathlib.Path) -> bool:
    if not target.is_file():
        return True
    t = target.stat().st_mtime
    return any(s.is_file() and s.stat().st_mtime > t for s in sources)


class PlutoFramework(Framework):
    """Times polycc's output on NPBench's PolyBench-derived kernels."""

    def version(self) -> str:
        """polycc's version. NOT ``pkg_resources``: Pluto is not a Python distribution."""
        exe = shutil.which("polycc")
        if exe is None:
            return "unavailable"
        try:
            proc = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
            first = (proc.stdout or proc.stderr or "").strip().splitlines()
            if not first:
                return "unknown"
            # "PLUTO version 0.12.0-33-gdc46216 - An automatic parallelizer and ..." -> the
            # version token alone; the tagline is the same for every run and only bloats the
            # `version` column it is stored in.
            m = re.search(r"PLUTO\s+version\s+(\S+)", first[0])
            return "PLUTO {v}".format(v=m.group(1)) if m else first[0].strip()
        except Exception:
            return "unknown"

    def impl_files(self, bench: Benchmark) -> Sequence[Tuple[str, str]]:
        """The tracked scop, so ``LineCount`` measures the source this column compiles."""
        return [(str(reference_source(bench)), "pluto")]

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """Transform, compile and wrap ``bench``'s scop; a single ``pluto`` implementation."""
        divergence = SEMANTIC_DIVERGENCE.get(bench.bname)
        if divergence:
            raise PlutoUnavailable("{b}: port and PolyBench original differ semantically. {d}".format(
                b=bench.bname, d=divergence))

        # An unsound parallel decoration is no longer a decline. polycc's TRANSFORMATION of these
        # kernels is correct -- it is only the `omp parallel for` it hangs on a loop that carries a
        # dependence that is not -- so the transform is kept exactly as generated and compiled
        # without -fopenmp. What gets timed is genuinely Pluto's tiled code; it just runs on one
        # thread, and says so, exactly like `durbin` and `ludcmp`. Timing the parallel build
        # instead would be timing a race.
        racy = UNSOUND_PARALLELIZATION.get(bench.bname)

        reference = reference_source(bench)
        if not reference.is_file():
            raise PlutoUnavailable("no tracked PolyBench scop for {b} (expected {p})".format(
                b=bench.bname, p=reference.name))

        # Scratch temporaries out of the function body and into the signature, so Pluto's
        # name-matched dead-code hack has no kill statement to act on. A no-op for the 17 scops
        # that declare no data-typed local; see `lift_scop_locals` for the defect and the proof
        # that the rewrite preserves semantics.
        source_text, lifted = (lift_scop_locals(reference.read_text()) if _LIFT_LOCALS else
                               (reference.read_text(), {}))
        base, params = parse_prototype(source_text)
        root = build_root() / bench.bname
        root.mkdir(parents=True, exist_ok=True)
        transformed = root / "{b}_pluto.c".format(b=base)
        so = root / "lib{b}_pluto.so".format(b=base)

        # The lifted unit is written out rather than piped: it is what polycc actually consumed,
        # so it has to be on disk beside the transform for any of this to be reviewable.
        source = reference
        if lifted:
            source = root / "{b}_lifted.c".format(b=base)
            if not source.is_file() or source.read_text() != source_text:
                source.write_text(source_text)

        if _stale(transformed, source):
            run_polycc(source, transformed)
        if _stale(so, transformed):
            compile_shared(transformed, so, openmp=not racy)
        if racy:
            print("PlutoSequential: {b}: polycc's transformation is correct but its `omp parallel "
                  "for` is not, so the unmodified transform is compiled without -fopenmp and runs "
                  "single-threaded. {d}".format(b=bench.bname, d=racy))

        lib = ctypes.CDLL(str(so))
        try:
            fn = getattr(lib, "{b}_fp64".format(b=base))
        except AttributeError:
            raise PlutoUnavailable("{s} exports no {b}_fp64".format(s=so.name, b=base))

        argtypes = []
        for _name, kind, _extents, (_np_t, c_t) in params:
            if kind == "symbol":
                argtypes.append(ctypes.c_int64)
            elif kind in ("array", "pointer"):
                argtypes.append(ctypes.POINTER(c_t))
            else:
                argtypes.append(c_t)
        fn.argtypes = argtypes
        fn.restype = None

        input_args = list(bench.info["input_args"])
        overrides = ARG_OVERRIDES.get(bench.bname, {})
        adapter = PLUTO_ADAPTERS.get(bench.bname, Adapter())

        # A benchmark whose port RETURNS its result and whose `output_args` is empty has nothing
        # for `Test` to compare in place. Since `utilities.validate` zips the reference and the
        # result -- truncating to the shorter -- returning None there would be validated against
        # zero pairs and pass without comparing anything. Refuse instead.
        if not bench.info.get("output_args") and not adapter.returns:
            raise PlutoUnavailable(
                "{b}: the port returns its result and `output_args` is empty, but no adapter "
                "declares `returns`; validation would compare nothing and pass vacuously".format(b=bench.bname))

        def impl(*actuals):
            if len(actuals) != len(input_args):
                raise PlutoUnavailable("{b}: expected {n} arguments, got {m}".format(b=bench.bname,
                                                                                    n=len(input_args),
                                                                                    m=len(actuals)))
            named: Dict[str, Any] = dict(zip(input_args, actuals))

            # Symbols first, resolved from the shapes of the arrays that declare them.
            # A VLA extent IS the array's dimension, so this cannot disagree with the
            # buffer actually passed -- which a value read from the preset could.
            for name, kind, extents, _elem in params:
                if kind != "array":
                    continue
                arr = named.get(name)
                if arr is None:
                    # Not necessarily an error: the scop may write this array as an output the
                    # port allocates. Only a name the adapter does not supply either is fatal,
                    # and that is caught below when the call is assembled.
                    continue
                for axis, sym in enumerate(extents):
                    if axis < np.ndim(arr):
                        named.setdefault(sym, np.shape(arr)[axis])

            # Symbols no extent determines (durbin's N), then the literals the port inlined.
            for name, fnc in adapter.symbols.items():
                named[name] = fnc(named)
            for name, const in adapter.constants.items():
                named.setdefault(name, const)

            # Output buffers the scop writes and the port allocates. Built in the same initial
            # state the port's own allocation gives them, so a cell neither side writes still
            # compares equal.
            for name, dtype, shape_fn, init_fn in adapter.outputs:
                try:
                    shape = tuple(int(d) for d in shape_fn(named))
                except KeyError as e:
                    raise PlutoUnavailable("{b}: cannot size output `{n}`, missing symbol {k}".format(
                        b=bench.bname, n=name, k=e))
                named[name] = init_fn(shape, named)

            # Scratch buffers for the lifted temporaries. Zero-filled: under the lifting
            # conditions the C original read an indeterminate value on any read-before-write
            # path, so zero is a valid realization of it and a deterministic one.
            for name, kind, extents, (np_t, _c_t) in params:
                if name not in lifted or name in named:
                    continue
                try:
                    shape = tuple(int(named[e]) for e in extents) if extents else (1, )
                except KeyError as e:
                    raise PlutoUnavailable("{b}: cannot size lifted scratch `{n}`, missing symbol {k}".format(
                        b=bench.bname, n=name, k=e))
                named[name] = np.zeros(shape, dtype=np_t)

            for name, fnc in overrides.items():
                named[name] = fnc(named)

            call = []
            keep = []  # references to any temporary contiguous copies, alive for the call
            for name, kind, _extents, (np_t, c_t) in params:
                if name not in named:
                    raise PlutoUnavailable("{b}: cannot resolve scop parameter `{n}`".format(b=bench.bname, n=name))
                value = named[name]
                if kind == "symbol":
                    call.append(ctypes.c_int64(int(value)))
                elif kind in ("array", "pointer"):
                    arr = np.ascontiguousarray(value, dtype=np_t)
                    if arr is not value:
                        # The scop writes through this pointer, so a converted copy would
                        # take the result with it and validation would compare stale data.
                        # This is also the check that catches an element-width mismatch
                        # between the port and the scop -- a positional ctypes call cannot.
                        raise PlutoUnavailable(
                            "{b}: argument `{n}` is not a C-contiguous {d} array (scop declares {d}); "
                            "the scop writes in place and a converted copy would not be observed".format(
                                b=bench.bname, n=name, d=np.dtype(np_t).name))
                    keep.append(arr)
                    call.append(arr.ctypes.data_as(ctypes.POINTER(c_t)))
                else:
                    call.append(c_t(value))

            fn(*call)
            del keep

            # Returned in the PORT'S order, so `Test` lines the two result lists up element for
            # element. A bare `None` here would be correct only for a kernel that mutates every
            # output in place.
            if not adapter.returns:
                return None
            if len(adapter.returns) == 1:
                return named[adapter.returns[0]]
            return tuple(named[n] for n in adapter.returns)

        return [(impl, "pluto")]

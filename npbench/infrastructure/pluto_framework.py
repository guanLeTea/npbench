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
      -> polycc --pet --tile --parallel   POLYCC_ARGS
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
#: * ``--pet``      -- the tracked scops use ``int64_t`` counters, which the default
#:                     ``clan`` extractor rejects.
#: * ``--tile``     -- off by default in polycc. An untiled Pluto column measures
#:                     almost nothing Pluto is for.
#: * ``--parallel`` -- also off by default. Without it polycc marks no loop parallel
#:                     and emits no ``#pragma omp parallel for``.
POLYCC_ARGS: Tuple[str, ...] = ("--pet", "--tile", "--parallel")

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
    "adi": ("NPBench's port computes `b = 1.0 + mul2` where PolyBench/C computes `b = 1.0 + mul1` "
            "(adi.py vs adi_pluto_reference.c). With the shared initialization mul1 = 2*mul2, so "
            "the two solve different tridiagonal systems -- 81 vs 161 at the S preset, not a "
            "rounding difference."),
}


def _zeros(dtype):
    return lambda shape, a: np.zeros(shape, dtype=dtype)


#: Per-benchmark argument adapters. Everything here was read off the NumPy port and the tracked
#: scop side by side; nothing is fitted to a validation result.
PLUTO_ADAPTERS: Dict[str, Adapter] = {
    # y = A^T (A x). The port returns the result; the scop writes it through `out` (and zeroes
    # it itself). `tmp` is a scop-local VLA, not a parameter.
    "atax":
    Adapter(outputs=[("out", np.float64, lambda a: (a["N"], ), _zeros(np.float64))], returns=["out"]),
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
    Adapter(outputs=[("imgOut", np.float64, lambda a: (a["W"], a["H"]), _zeros(np.float64))], returns=["imgOut"]),
    # The port inlines `stddev[stddev <= 0.1] = 1.0`; the scop takes the same rule as two
    # parameters. corr is allocated as np.eye(M) by the port -- matched exactly here, so the
    # diagonal agrees whichever side writes it.
    "correlation":
    Adapter(constants={
        "stddev_eps": 0.1,
        "stddev_replacement": 1.0
    },
            outputs=[("corr", np.float64, lambda a: (a["M"], a["M"]), lambda shape, a: np.eye(shape[0],
                                                                                              dtype=np.float64))],
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
    # alpha is ABI padding: the scop hard-codes 0.125 and discards the parameter with
    # `(void)alpha;`. Passed as 0.0 because no value can affect the result.
    "heat_3d":
    Adapter(constants={"alpha": 0.0}),
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
            # polycc ran but marked nothing parallel. Timing that would report a
            # sequential build under Pluto's name.
            raise PlutoUnavailable("polycc marked no loop parallel in {r}".format(r=reference.name))
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
        os.replace(str(tmp_out), str(out))


def compile_shared(source: pathlib.Path, so: pathlib.Path) -> None:
    """Compile polycc's output into ``so``, or raise."""
    exe = shutil.which("clang")
    if exe is None:
        raise PlutoUnavailable("clang is not on PATH (source slurm/npbench-env.sh)")
    tmp_so = so.with_suffix(so.suffix + ".tmp")
    cmd = [exe, *CLANG_FLAGS, "-shared", "-o", str(tmp_so), str(source), "-lm"]
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

        reference = reference_source(bench)
        if not reference.is_file():
            raise PlutoUnavailable("no tracked PolyBench scop for {b} (expected {p})".format(
                b=bench.bname, p=reference.name))

        base, params = parse_prototype(reference.read_text())
        root = build_root() / bench.bname
        root.mkdir(parents=True, exist_ok=True)
        transformed = root / "{b}_pluto.c".format(b=base)
        so = root / "lib{b}_pluto.so".format(b=base)

        if _stale(transformed, reference):
            run_polycc(reference, transformed)
        if _stale(so, transformed):
            compile_shared(transformed, so)

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

# NPBench on CSCS Daint/Alps

## Campaign base

The M campaign runs on branch **`bench/daint-pluto`**, which is
**spcl/npbench PR #47** (`fix/well-conditioned-gramschmidt-and-zeros-init`, tip `3944369`)
with our three commits rebased on top. PR #47's base is upstream `main` at `f2d7f27`.

**Not plain `main`.** PR #47 makes the benchmark inputs deterministic and well-conditioned,
which is a precondition for comparable numbers:

| Change | Kernels |
|---|---|
| `np.empty` -> `np.zeros` in the input generator | `cholesky`, `cholesky2`, `lu`, `ludcmp`, `symm` |
| `np.empty_like` -> `np.zeros_like` inside the kernel | `deriche` (y1, y2), `durbin` (y) |
| deterministic well-conditioned input (replaces a nondeterministic `while matrix_rank(A) < N` resample) | `gramschmidt` |
| input drawn from the seeded generator | `mlp` |
| new `dace_canonicalize_cpu` / `dace_canonicalize_gpu` columns | — |
| `openmp_array_reductions = False` pinned for `dace_cpu`/`dace_gpu` | — |

Two commits in PR #47 (`c77d0f4`, `3944369`) also rename kernel functions in `pythran`,
`numba_np`, `legate` and `dask` variants. None of those columns is in this campaign, and no
`_numpy.py` or `_dace.py` file is touched by them.

**Known wart in PR #47:** commit `3944369` also commits a 45 KB `npbench.db` (155 numpy/numba
rows from another machine) into the repo root. NPBench *appends* to whatever `npbench.db` it
finds in the CWD, so an interactive run started from the repo root would mix those rows into
its own. The launcher is unaffected -- every rank runs from its own directory -- but the file
should be deleted before it causes confusion.


Everything in this directory is site-specific: the uenv name, the scratch paths and the account are
Daint's. Nothing here is upstream NPBench.

## Environment

The base interpreter, GCC, cmake and OpenBLAS all come from the `prgenv-gnu/26.3:v1` uenv. The
repo-local venv is built on top of it, so it only works with that uenv mounted.

```bash
uenv start --view=default prgenv-gnu/26.3:v1
source ~/npbench/slurm/npbench-env.sh
```

In a batch job, request the uenv from Slurm instead of starting it first — that way the mount
reaches the batch step *and* every nested `srun` step:

```bash
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default
```

`npbench-env.sh` activates the venv, puts clang and `polycc` on PATH, pins the compiler DaCe hands
to CMake, and refuses to continue if any of that is missing. It deliberately does **not** set thread
counts — that depends on how many ranks are splitting the node, so it belongs to the launcher.

### What is installed

`~/npbench/.venv`, created from the uenv's Python 3.14.3:

| Package | Version | Why |
|---|---|---|
| numpy | 2.5.2 | the baseline column |
| scipy, pandas, matplotlib, pygount | | `requirements.txt` |
| setuptools | **< 81** | NPBench imports `pkg_resources`, removed in setuptools 81 |
| dace | 2.0.0a5, **editable** from `~/dace` | the DaCe column |

`setuptools<81` is a pin, not an accident. `npbench/infrastructure/framework.py` and
`dace_framework.py` call `pkg_resources.get_distribution(...)`; pinning keeps the repo unmodified.

DaCe is an **editable** install of `~/dace`, so whichever branch that tree is on is the DaCe that
runs. `npbench-env.sh` prints the branch in its banner. Check it before a measurement and stamp the
answer on the run — a `dace_cpu` row does not record which tree produced it.

### BLAS

NumPy's wheel carries its own `scipy-openblas` 0.3.34, built `DYNAMIC_ARCH` and correctly selecting
the `neoversev2` kernels on Grace. It is left alone; nothing is rebuilt against the uenv's OpenBLAS
0.3.30, and no second BLAS is installed. Note `MAX_THREADS=64` in that build: a rank given 72 CPUs
runs BLAS on at most 64 of them.

The uenv's OpenBLAS is what DaCe-generated and Pluto-generated C link against.

## Running

```bash
python run_benchmark.py -b <benchmark> -f <framework> -p <preset> -r <repeat>
python run_framework.py  -f <framework> -p <preset>          # every benchmark
```

Frameworks: `numpy`, `dace_cpu`, `dace_canonicalize_cpu`, `pluto`. `dace_cpu` builds and times
three SDFG variants per kernel (`fusion`, `parallel`, `auto_opt`) and records all three;
`dace_canonicalize_cpu` (from PR #47) times the fork's canonicalize pipeline as one variant and
turns ON OpenMP array-section reductions, which `dace_cpu` deliberately leaves off.

Every non-NumPy column is validated against the NumPy reference on its first execution; the verdict
is stored per row in the `validated` column of `npbench.db`.

## Pluto coverage

The Pluto column transforms the **original PolyBench/C 4.2.1 kernel**, tracked beside each NumPy
port as `npbench/benchmarks/polybench/<kernel>/<kernel>_pluto_reference.c` (23 files). NPBench's own
kernels are NumPy ports of those same PolyBench kernels, so the pair measures the same computation
from the two sources it actually has.

Correctness is not assumed from that shared ancestry — every row is validated against the NumPy
reference, and that check is load-bearing here: `jacobi_2d` and `seidel_2d`'s ports sweep one fewer
timestep than the PolyBench originals (corrected in `ARG_OVERRIDES`), and `floyd_warshall` is an
`int32` kernel where the rest are `float64`.

**Working — 13 of 23**, every one validated against its NumPy reference:
`cholesky`, `floyd_warshall`, `gemm`, `gemver`, `heat_3d`, `jacobi_2d`, `lu`, `mvt`, `seidel_2d`,
`syr2k`, `syrk`, `trisolv`, `trmm`.

The remaining 10 are **all blocked upstream in Pluto**, not by missing adapters. The adapters were
written (`PLUTO_ADAPTERS` covers output buffers, inlined constants and non-extent symbols) and they
are correct — the kernels still fail because `polycc` miscompiles or crashes on their scops.

### Two Pluto defects, and why they are declines rather than workarounds

**1. Dropped statements (6 kernels).** pet models a temporary declared inside the function —
`atax`'s `tmp[M]`, `gramschmidt`'s scalar `nrm`, `deriche`'s `y1`/`y2` — as scop-local, and Pluto
then eliminates the statements that write it because nothing *outside* the scop reads it. The
scop's own later statements do. The emitted code reads an uninitialized buffer and returns
denormal noise or NaN.

This is the dangerous one: it compiles clean, runs fast, and is wrong. `_dropped_writes()` compares
the scop's write set against the generated file's and declines on any difference.

**2. Integer-literal overflow (2 kernels).** Pluto's scheduler produces coefficients that overflow
`int64` and prints them into the loop bounds. In `fdtd_2d` clang rejects them outright. In `bicg`
the literal lands inside a guard — `if (9223372036854775808*N >= -M+1)` — which compiles with only
a warning, is undefined at runtime, and **skips the guarded loop**, so the kernel returns its
untouched input. `_int64_overflow_literals()` catches both before anything is compiled.

| Kernel | Cause |
|---|---|
| `atax`, `gramschmidt`, `durbin`, `correlation`, `deriche`, `adi` | dropped statements (defect 1) |
| `bicg`, `fdtd_2d` | int64 literal overflow (defect 2) |
| `ludcmp` | `polycc` aborts on an internal assertion in `pluto_auto_transform` |
| `nussinov` | `polycc` corrupts its own heap (`double free or corruption`) and then hangs |

Tried and rejected: `--lastwriter` / `--nolastwriter` / dropping `--tile` change nothing for defect
1. Hoisting `atax`'s `tmp` into the signature (which is what PolyBench upstream actually does) fixes
defect 1 for it — and then it hits defect 2 instead.

**`adi` is additionally a semantic divergence** and would stay blocked even if Pluto were fixed:
NPBench's port computes `b = 1.0 + mul2` where PolyBench computes `b = 1.0 + mul1`. With the shared
initialization those are 81 vs 161 — a different linear system, not a rounding difference.

### Rules this column holds to

- A kernel that cannot be mapped is **declined outright**, never mapped approximately: a positional
  `ctypes` call cannot detect a permuted or mistyped argument list, so a wrong mapping runs and
  returns plausible numbers.
- polycc's output is **inspected and rejected, never edited**. Repairing a dropped statement by hand
  would mean timing something Pluto did not produce.
- `polycc` and `clang` run under a 300s timeout, so a Pluto that hangs cannot consume a campaign's
  allocation.
- `utilities.validate` compares with `zip(ref, val)`, which **truncates to the shorter list**. A
  kernel whose port returns its result and whose `output_args` is empty would, if this column
  returned `None`, be "validated" by comparing zero pairs. Any such benchmark that does not declare
  `returns` in its adapter is declined.

## Reproducibility

DaCe is an editable install, so its version string (`2.0.0a5`) is identical across branches and
commits and cannot identify what ran. Two things fix that:

- **Per row.** `DaceFramework.version()` appends `+<branch>@<commit>` (and `-dirty` when the tree has
  uncommitted changes), so the `version` column of `npbench.db` reads
  `2.0.0a5+extended@eb7b1352a`. `NPBENCH_DACE_BUILD` overrides the probe for a non-git tree.
- **Per campaign.** Each job writes `manifest.json` into its results namespace: both git commits,
  compiler versions, polycc version, node and preset.

Result namespaces are keyed on the Slurm job id and the launcher **refuses to start** if one already
exists — NPBench *appends* to whatever `npbench.db` it finds, so a reused namespace silently mixes
two runs into one database.

## Per-rank state (read before writing a launcher)

NPBench writes its results to **`npbench.db` relative to the current working directory**
(`infrastructure/test.py`, `infrastructure/line_count.py`). Several ranks sharing a working
directory means several processes writing one SQLite file. Give each rank its own directory and
merge afterwards with `merge_db.py`. Nothing else depends on the CWD: `run_benchmark.py` puts its
own directory on `sys.path`, and `bench_info` / `framework_info` / the kernel sources are all
resolved from `__file__`.

Two more per-rank directories are required:

- `DACE_default_build_folder` — DaCe derives an SDFG's build folder from its **name**, and NPBench
  names every variant identically on every kernel (`fusion`, `parallel`, `auto_opt`). One shared
  folder has every rank overwriting one `libauto_opt.so`.
- `NPBENCH_PLUTO_BUILD_DIR` — same hazard with `lib<kernel>_pluto.so`.

## CPU affinity

`--cpus-per-task` must be spelled on the `#SBATCH` line **and** passed explicitly to `srun`. Slurm
25.05 does not propagate it into the step (that implicit inheritance went away in 22.05). Without it
on the `srun` line, `cpus-per-task` defaults to 1 and `--cpu-bind=cores` pins each rank to a single
core while the rank still starts a full thread pool. The result looks like a measurement and is not.

Each rank sizes its thread pool from `SLURM_CPUS_PER_TASK`, never from `nproc` — `nproc` reports the
whole node, so every rank would oversubscribe it by the rank count.

A Daint node is 4 Grace sockets, 72 cores each, 288 total. 4 ranks × 72 gives one socket per rank;
`slurm/verify_small.sbatch` confirmed the ranks land on `0-71`, `72-143`, `144-215`, `216-287`.

## Files

| File | What it does |
|---|---|
| `npbench-env.sh` | sourced by every job; venv + toolchain + fatal preflight |
| `verify_small.sbatch` | setup check: numpy/pluto/dace_cpu, preset S, 7 kernels, 1 node |
| `merge_db.py` | merge per-rank `npbench.db` shards; `--summarize` prints validation + means |

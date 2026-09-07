# Original-NPBench / PolyBench-derived benchmarking phase — factual technical summary

Source material for thesis writing. Not prose. Every number below is taken from the tracked
repository, the campaign manifests, or the campaign result database; provenance is named per item.

Primary artifact directory: `~/npbench/results/paper-final50-4499210/`
Primary provenance file: that directory's `manifest.json`.

---

## 1. CSCS / hardware / software environment

### 1.1 Site and node

| Item | Value |
|---|---|
| Site | CSCS, Alps / Daint |
| Partition | `normal` |
| Allocation | 1 node, `--exclusive` |
| CPU | NVIDIA Grace (aarch64), 288 CPUs per node, 4 sockets x 72 cores |
| Node used, final campaign | `nid005315` (recorded in `manifest.json`) |
| Nodes used, earlier campaigns | `nid006513` (M x1, L x20), `nid005312` (Paper x20) |

### 1.2 Slurm / rank layout (identical in every campaign)

```
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72        # 4 x 72 = 288 = whole node
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default
srun --ntasks=4 --ntasks-per-node=4 --cpus-per-task=72 --cpu-bind=cores
```

`--cpus-per-task` is passed to `srun` explicitly as well as to `sbatch`: Slurm 25.05 does not
propagate the batch value into the step.

Measured affinity, read from the final job log `slurm-4499210.out` (not assumed):

| rank | CPUs | threads | kernels assigned |
|---|---|---|---|
| 0 | 0–71 | 72 | adi, correlation, floyd_warshall, heat_3d, mvt, syrk |
| 1 | 72–143 | 72 | atax, deriche, gemm, jacobi_2d, nussinov, trisolv |
| 2 | 144–215 | 72 | bicg, durbin, gemver, lu, seidel_2d, trmm |
| 3 | 216–287 | 72 | cholesky, fdtd_2d, gramschmidt, ludcmp, syr2k |

Rank-to-kernel assignment is round-robin over the alphabetically sorted kernel list
(`i % RANKS == rank`), so it is deterministic and reproducible.

### 1.3 Thread / BLAS settings (set by the launcher, per rank)

```
OMP_NUM_THREADS      = 72
OPENBLAS_NUM_THREADS = 72
MKL_NUM_THREADS      = 72
NUMEXPR_NUM_THREADS  = 72
OMP_PROC_BIND        = close
OMP_PLACES           = cores
```

`npbench-env.sh` deliberately does **not** set thread counts; they belong to the launcher because
the correct value depends on how many ranks split the node.

### 1.4 Software stack (final campaign, from `manifest.json` + `slurm/README.md`)

| Component | Version / identifier |
|---|---|
| uenv | `prgenv-gnu/26.3:v1`, `default` view (`UENV_VIEW=/user-environment:prgenv-gnu:default`) |
| Python | 3.14.3 (uenv interpreter; venv built on top of it) |
| venv | `~/npbench/.venv` |
| NumPy | 2.5.2 |
| DaCe | 2.0.0a5, **editable** install of `~/dace` |
| SciPy / pandas / matplotlib / pygount | from `requirements.txt` |
| setuptools | pinned `< 81` (NPBench imports `pkg_resources`, removed in setuptools 81) |
| GCC / G++ | 14.3.0 (uenv) |
| Clang / LLVM | 17.0.6, prefix `/capstor/scratch/cscs/levwidmer/opt/llvm-17.0.6` |
| polycc | `PLUTO version 0.12.0-33-gdc46216` |
| cmake | uenv |

DaCe's CMake compiler is pinned explicitly (`DACE_compiler_cpu_executable`,
`DACE_compiler_linker_executable` = the uenv `g++`). Without the pin, CMake falls through the uenv
to SUSE `/usr/bin/c++` (GCC 7.5), which has no `-std=c++23`, and every SDFG build fails.

### 1.5 BLAS configuration (two distinct BLAS in the experiment)

- **NumPy column**: the wheel's bundled `scipy-openblas` **0.3.34**, built `DYNAMIC_ARCH`,
  correctly selecting the `neoversev2` kernels on Grace. Nothing was rebuilt against the uenv
  OpenBLAS. That build carries `MAX_THREADS=64`, so a rank given 72 CPUs runs BLAS on at most 64
  of them.
- **DaCe-generated and Pluto-generated C**: link the **uenv OpenBLAS 0.3.30**.

This asymmetry is deliberate (nothing rebuilt, nothing swapped) and is recorded as a limitation.

### 1.6 Compiler flags for the Pluto column

`polycc` invokes no compiler; the framework compiles its output itself:

```
clang -O3 -march=native -fopenmp -fno-math-errno -fno-trapping-math -fno-signed-zeros
      -fstrict-aliasing -fPIC -fveclib=libmvec -shared
```

`-fopenmp` uses clang's own libomp from the LLVM prefix (not `-fopenmp=libgomp`); the pragma has
to actually be honoured. **No fast-math** — validation against the NumPy reference has to mean
something.

### 1.7 Measured revisions

| Tree | Branch | Commit |
|---|---|---|
| NPBench | `bench/daint-pluto` | `7f49c1ae764a2c986a063cc3c60b203b58d4bd94` ("Recover nussinov as a labelled single-threaded Pluto result") |
| DaCe (`~/dace`) | `extended` | `eb7b1352a09298451eb8c65b937d0c102dc2cfae` |
| Pluto | `bondhugula/pluto` | `dc46216`, `git describe` = `0.12.0-33-gdc46216` |

Pluto submodule pins at the measured build:

| Submodule | Pin |
|---|---|
| clan | `fb9bd2c` (`0.8.1-7-gfb9bd2c`) — **the frontend used for all measured results** |
| pet | `2320f24` — present but not used (see §5) |
| isl | `0114734` (master) |
| cloog-isl | `d9108b9` (`cloog-0.18.3-88-gd9108b9`) |
| candl | `08b7186` (0.6.3) |
| openscop | `b79af02` (`0.9.1-126-gb79af02`) |
| piplib | `261eec0` |
| polylib | `597776a` |

Both trees were **clean** at the final campaign: `npbench_dirty_files: 0`, `dace_dirty_files: 0`
(recorded in `manifest.json`, computed with `git status --porcelain --untracked-files=no`).

### 1.8 Provenance and isolation discipline

- **Job-isolated result directories.** Namespace is `results/${CAMPAIGN}-${JOB}` and the launcher
  **refuses to start** if the namespace already contains artifacts. Rationale: NPBench *appends*
  to whatever `npbench.db` it finds in the CWD, so a reused namespace would silently merge two
  campaigns.
- **No shared DB append.** Each rank runs in its own directory (`rank-N/`) and writes its own
  `npbench.db`; the four are merged after the run by `slurm/merge_db.py`.
- **Per-rank build trees.** `DACE_default_build_folder` and `NPBENCH_PLUTO_BUILD_DIR` are
  per-rank: DaCe names every SDFG variant identically across kernels, and the Pluto column names
  every library `lib<kernel>_pluto.so`, so a shared folder would have ranks overwriting each other.
- **DaCe identity stamped on every result row.** The `version` column reads
  `2.0.0a5+extended@eb7b1352a`; DaCe is an editable install, so a bare `dace_cpu` row would not
  otherwise record which tree produced it.
- **`manifest.json`** captures, per campaign: campaign name, job id, stated purpose, ISO date,
  node, preset, repeat, timeout, kernel count, framework list, ranks, cpus-per-task, NPBench
  commit/branch/dirty-count, DaCe tree/branch/commit/dirty-count, Python, NumPy, DaCe, GCC, Clang
  and polycc versions, and the uenv view.
- **Kernel list derived from the tree**, not typed: the launcher globs `*_pluto_reference.c` so
  the corpus cannot drift from what the Pluto column can be asked about.

Version stamps as actually written into the final DB:

```
dace_cpu_autoopt   2.0.0a5+extended@eb7b1352a   details=auto_opt   1150 rows
numpy              2.5.2                        details=default    1150 rows
pluto              PLUTO 0.12.0-33-gdc46216     details=pluto      1100 rows
```

---

## 2. Benchmark corpus and source audit

### 2.1 Why original NPBench rather than HPCAgent-Bench

The final synthetic evaluation was moved onto **original NPBench** in order to:

1. Remove source-generation / translation confounders — no LLM- or tool-generated implementation
   sits between the specification and the measured code.
2. Measure the **original NPBench ports** as published, so the baseline is a known, citable
   artifact rather than a derived one.
3. Allow the PolyBench-derived subset to be compared **directly against canonical PolyBench/C
   4.2.1**, which is what makes the Pluto column meaningful at all.
4. Keep NumPy, Pluto and DaCe under **one controlled harness** — same driver, same validation,
   same timing path, same preset, same allocation.
5. Make compiler state, polyhedral frontend, CPU affinity and validation **explicitly controlled
   and recorded** rather than implicit.

### 2.2 Stable-input base: spcl/npbench PR #47

The measurement branch `bench/daint-pluto` is **spcl/npbench PR #47**
(`fix/well-conditioned-gramschmidt-and-zeros-init`, tip `3944369`, base upstream `main` @
`f2d7f27`) with the campaign commits rebased on top. It is **not** plain `main`.

What PR #47 changes:

| Change | Kernels |
|---|---|
| `np.empty` → `np.zeros` in the input generator | cholesky, cholesky2, lu, ludcmp, symm |
| `np.empty_like` → `np.zeros_like` inside the kernel | deriche (y1, y2), durbin (y) |
| deterministic well-conditioned input, replacing a nondeterministic `while matrix_rank(A) < N` resample | gramschmidt |
| input drawn from the seeded generator | mlp |
| new `dace_canonicalize_cpu` / `dace_canonicalize_gpu` columns | — |
| `openmp_array_reductions = False` pinned for `dace_cpu` / `dace_gpu` | — |

Why deterministic inputs matter here: uninitialised output buffers make a validation verdict
depend on heap contents; a resampling loop makes both the input *and* the number of RNG draws
vary run to run, so neither validation nor timing is reproducible across repetitions.

What did **not** affect the measured frameworks:
- Two PR #47 commits (`c77d0f4`, `3944369`) rename kernel functions in the `pythran`, `numba_np`,
  `legate` and `dask` variants. None of those columns is in this campaign, and no `_numpy.py` or
  `_dace.py` file is touched by them.
- `dace_canonicalize_cpu/gpu` are added by PR #47 but are **not** the measured DaCe column
  (see §4).

Known wart, recorded and mitigated: PR #47 commits a 45 KB `npbench.db` (155 numpy/numba rows from
another machine) into the repo root. NPBench appends to whatever `npbench.db` is in the CWD, so an
interactive run started from the repo root would mix those rows in. The launcher is structurally
immune (every rank runs from its own directory); the file was removed on the campaign branch
(commit `e9932e2`).

### 2.3 Final corpus — 23 PolyBench-derived NPBench kernels

Exactly those kernels carrying a tracked `<kernel>_pluto_reference.c`:

| Category | Kernels |
|---|---|
| BLAS / linear algebra kernels | gemm, gemver, syrk, syr2k, trmm, atax, bicg, mvt |
| Solvers / factorisations | cholesky, lu, ludcmp, trisolv, durbin, gramschmidt |
| Stencils | fdtd_2d, heat_3d, jacobi_2d, seidel_2d, adi |
| Datamining | correlation |
| Medley | deriche, floyd_warshall, nussinov |

`floyd_warshall` is an **int32** kernel; every other kernel is float64.

### 2.4 Three-way source audit

Every kernel was compared across three sources:

1. **Canonical PolyBench/C 4.2.1** (fetched from the 4.2.1 sources, not read off a transcription).
2. **NPBench's Python/NumPy port**.
3. **The tracked Pluto reference C** (`<kernel>_pluto_reference.c`), which is the canonical
   PolyBench kernel verbatim.

Outcome: **most references were equivalent** to canonical PolyBench. Two classes of genuine
difference were found.

**(a) Loop-count conventions — differences in the ports, handled by explicit argument mapping,
not by editing either side** (`ARG_OVERRIDES` in `pluto_framework.py`):

- `jacobi_2d`: PolyBench sweeps `for (t = 0; t < TSTEPS; t++)`; the NPBench port sweeps
  `range(1, TSTEPS)` — one fewer. The C call is given `TSTEPS-1`.
- `seidel_2d`: PolyBench sweeps `t <= TSTEPS-1` (TSTEPS sweeps); the port sweeps
  `range(0, TSTEPS-1)`.

Both overrides are derived from the two loop bounds, not fitted to make validation pass.

**(b) Two genuine NPBench semantic porting defects**, corrected in commit `da8e07c`:

| Kernel | Canonical PolyBench/C 4.2.1 | NPBench as shipped | Effect |
|---|---|---|---|
| **adi** | `stencils/adi/adi.c`: `b = SCALAR_VAL(1.0) + mul1;` | `b = 1.0 + mul2` | breaks the symmetry of PolyBench's two coefficient triples — `(a,b,c)` from `mul1`, `(d,e,f)` from `mul2` — while leaving `e = 1.0 + mul2`. Present since NPBench's first commit. |
| **deriche** | `medley/deriche/deriche.c:83`, denominator `1.0 + 2.0*alpha*exp(-alpha) - exp(2.0*alpha)` | the `2.0` factor missing | `k` scales `a1..a8`, so every output pixel was scaled. Measured relative error **2.07**. |

Corrections were applied **consistently to every implementation this experiment measures** —
the NumPy port and the DaCe source alike — so no framework is bent to make another one validate.
Neither tracked `*_pluto_reference.c` changed: both were already byte-identical to the canonical
scop, which is exactly why the divergence surfaced. Both corrected ports were then validated
against the **untransformed canonical C compiled directly**.

Consequence for framing: the final experiment evaluates **canonical PolyBench semantics as
expressed through NPBench**, not vanilla NPBench, for these two kernels. The pre-correction state
is recoverable in git history at commit `0a58d76` and earlier.

The mechanism that refused divergent kernels (`SEMANTIC_DIVERGENCE`) is now **empty but retained**:
it is what prevents future port drift from quietly becoming a plausible-looking Pluto "result".

---

## 3. Framework implementation

Three columns, one harness, one driver invocation shape:

```
python -u run_benchmark.py -b <kernel> -f <framework> -p <preset> -r <repeat> -t <timeout> -v True
```

`-v True` enables validation. `python -u` is load-bearing: a Pluto-transformed kernel can SIGSEGV
the interpreter (measured on nussinov, rc=139), and block-buffered stdout is lost when a process
dies by signal — without `-u`, the log of the pair that most needs explaining arrives empty. The
process exit code is captured beside each log in a `.rc` file, so a signal death (128+N) is
distinguishable from a clean run that printed nothing.

### 3.1 NumPy column

Unmodified NPBench NumPy ports (with the §2.4(b) canonicalizations), NumPy 2.5.2 on bundled
`scipy-openblas` 0.3.34. This is the validation reference for the other two columns and the
speedup denominator.

### 3.2 Pluto column (`pluto`) — a build path, not a flag preset

`polycc` is a source-to-source polyhedral optimizer that invokes no compiler, so compiling and
calling the result is the framework's job. Pipeline per benchmark, cached on mtime:

```
<module>_pluto_reference.c                 tracked PolyBench/C 4.2.1 scop
  -> polycc --tile --parallel --codegen-context=1     (Clan frontend, polycc default)
  -> <module>_pluto.c                       transformed, same signature
  -> clang -O3 -march=native -fopenmp ... -shared
  -> lib<module>_pluto.so                   exports <module>_fp64
  -> ctypes call in the C signature's own argument order
```

**Important framing point** (also recorded in project memory): the Pluto column times the
**original PolyBench/C kernel**, *not* a translated NumPy port. NPBench's kernels are NumPy ports
of those same PolyBench kernels, so the pair measures the same computation from the two sources
that actually exist: the array-expression port and the loop nest Pluto's model was designed for.

Correctness is **not** assumed from shared ancestry. Every row is validated against the NumPy
reference by `infrastructure.test.Test.run`, and that check is load-bearing here: a positional
ctypes call cannot detect a permuted argument list on its own.

Argument mapping: each C parameter must be an NPBench input argument, an array extent, or an
explicit per-kernel adapter entry (`PLUTO_ADAPTERS`: output buffers, scratch arrays, inlined
constants, non-extent symbols). A kernel whose parameters cannot all be resolved is **declined
outright** rather than guessed at.

### 3.3 DaCe column (`dace_cpu_autoopt`) — see §4.

---

## 4. DaCe setup

### 4.1 Why a dedicated framework was created

NPBench's stock `dace_cpu` builds and records **three** SDFG variants per kernel: `fusion`,
`parallel`, `auto_opt`. Any report against it must pick one, and the initial plotting picked the
**fastest validated variant** per kernel. That is a defensible answer to "how fast is DaCe", but it
is the **wrong baseline for this comparison**: Pluto runs one fixed pipeline, so scoring it against
the best of three DaCe transformations compares one optimizer against a per-kernel search. It did
not match the intended comparison.

### 4.2 What `dace_cpu_autoopt` is

`npbench/infrastructure/dace_autoopt_framework.py` — a **narrowing, not a reimplementation**. The
class body is one attribute:

```python
class DaceAutoOptFramework(DaceFramework):
    VARIANTS: Tuple[str, ...] = ("auto_opt", )
```

Everything else — parsing, simplification, compilation, the results row, the version stamp — is
`DaceFramework`'s own code path.

Structural guarantees:

- `auto_opt` derives from the **simplified (`strict`)** SDFG, never from the fused or parallelized
  one (`parallel` is the only variant with such a dependency, and it derives from `fusion`).
  Narrowing therefore skips the other two without changing what `auto_optimize` is handed.
- `set_fast_implementations` was already skipped for `auto_opt` in the shared path, so nothing that
  reaches `auto_optimize` changes.
- `fusion` is still *constructed* when `parallel` is wanted (parallel starts from it) but is only
  *recorded* when itself in `VARIANTS`.
- **No best-of-three selection is structurally possible** in this campaign.
- The DaCe version/branch/commit stamp is inherited unchanged:
  `2.0.0a5+extended@eb7b1352a`.

### 4.3 Equivalence check against the prior `auto_opt` path

Verified on a compute node (**job 4489828**, preset M, 10 kernels spanning BLAS, stencils and
solvers, deliberately including both kernels where `auto_opt` was *not* the fastest variant):

- validation verdicts agree **10/10**;
- `dace_cpu_autoopt` records `auto_opt` rows and no others; `dace_cpu` still records all three;
- timings agree within **9% on 9 of 10**; `syrk` differs by 42% at a **0.13 ms** median, inside
  each run's own ~60% spread at that scale.

---

## 5. Pluto debugging and recovery

### 5.1 Coverage progression

| Stage | Coverage | Driver |
|---|---|---|
| Initial direct-NPBench Pluto column | **13 / 23** | as-integrated |
| After integration fixes (codegen context, scratch arrays as parameters, exit-code recovery) | **17 / 23** | commits `2405f50`, `502fa74`, `59c5e22` |
| After dropping the pet frontend | **20 / 23** | commit `395f9d8` |
| After canonical semantic fixes (deriche) | **21 / 23** | commit `da8e07c` |
| After nussinov recovery | **22 / 23** | commit `7f49c1a` |
| **Final** | **22 / 23** — adi the only decline | measured state |

The 13 working at the initial stage: cholesky, floyd_warshall, gemm, gemver, heat_3d, jacobi_2d,
lu, mvt, seidel_2d, syr2k, syrk, trisolv, trmm. The remaining 10 were blocked **upstream in Pluto**,
not by missing adapters — the adapters were written and correct, and the kernels still failed
because `polycc` miscompiled or crashed on their scops.

### 5.2 Finding 1 — pet vs Clan frontend

The column originally passed `--pet`, on the belief that polycc's default `clan` extractor rejects
the tracked scops' `int64_t` counters. **Measured: it does not.** Clan extracts 22 of the 23 scops
with the correct statement count on every one (only adi fails, on separate grounds).

`--pet` cost four kernels through one Pluto defect and its downstream effects:

- pet emits a **KILL statement for every function-local variable**. Then
  `pet_to_pluto.cpp:mark_trivial_dead_code()` deletes **every statement writing a killed NAME** —
  described in Pluto's own source comment as *"a HACK to get rid of old IV init's and increments"*,
  and implemented as an unconditional name match rather than liveness analysis.
  - `gramschmidt` lost `nrm`'s two writes (5 statements of 7) and read it uninitialised.
  - `durbin` lost 3 statements of 10.
  - `ludcmp` lost its accumulator's writes, leaving a degenerate dependence graph on which Pluto
    **aborted in `pluto_auto_transform`**.
  - Clan emits no kills and extracts all three intact (7, 10 and 12 statements).
- pet separately **miscompiled `nussinov`'s reversed `i` loop** into `table[-(N-2)][...]` — row −58
  at N=60, ≈14 KB before the buffer — which ASan reports as a SEGV at the generated line. Clan
  emits no such index.

`NPBENCH_PLUTO_FRONTEND=pet` still reproduces all of the above; the defect-declining guards
(`_negated_subscripts`, `lift_scop_locals`) are retained and tested but inactive under Clan.

`lift_scop_locals` (lifting scop-local scratch into explicit parameters) was written, verified
semantics-neutral (durbin bit-identical to plain C at `-O0`/`-O1`), and is **kept but OFF** under
Clan, where a lifted scalar is a shared cell that Pluto must serialize on — it would only cost
parallelism.

### 5.3 Finding 2 — `--codegen-context=1`

Meaning: "parameters are at least 1". Without it, Pluto also generates the degenerate cases where a
size parameter is zero or negative, and the guards separating those cases carry coefficients that
**overflow the loop counter type**:

- `bicg` emitted `if (9223372036854775808*N >= -M+1)` — 2^63, one past int64 max;
- `fdtd_2d` emitted a 29-digit literal.

The first compiles with a warning, is undefined at run time, and **skips the loop it guards**, so
the kernel returns its untouched input very fast — a silent wrong-and-fast result. Both disappear
with the context set. This is not a tuning knob: every NPBench preset passes positive sizes, and a
PolyBench kernel is undefined for non-positive ones, so the flag states a fact about the call
rather than assuming one away.

### 5.4 Finding 3 — canonical scratch arrays and parameters

The C signature mattered: several PolyBench kernels take scratch/output buffers as **parameters**
where the NumPy port returns them. Handled in `PLUTO_ADAPTERS`, read off the NumPy port and the
tracked scop side by side, never fitted to a validation result. Examples:

- `atax`: scop writes through `out` and takes `tmp` as a parameter (`kernel_atax` does too); the
  port returns the result and treats `tmp` as a local VLA.
- `bicg`: port returns `(r @ A, A @ p)`; scop writes them as `out0` (length M) and `out1` (length N).
- `correlation`, `deriche`, `durbin`, `gramschmidt`, `ludcmp`, `nussinov`: output buffers
  zero-initialised so regions neither side writes (e.g. `R`'s lower triangle in gramschmidt) agree
  by construction. The full adapter set is exactly: atax, bicg, durbin, gramschmidt, deriche,
  correlation, nussinov, ludcmp, adi, heat_3d, fdtd_2d — 11 of 23; the other 12 map by name alone.
- `heat_3d`, `fdtd_2d`, `adi`: some parameters are **ABI padding** — the scop hard-codes the
  constant and discards the parameter with `(void)`. They are passed as the literals the kernel
  computes so the ABI reads correctly, though no value can reach the result.

### 5.5 Finding 4 — nussinov (recovered as a labelled single-threaded result)

- **Dependence structure, rechecked against PolyBench/C 4.2.1 from scratch**:
  `table[i][j] = max(table[i][j], table[i][k] + table[k+1][j])` reads `table[k+1][j]` for
  `k+1 ∈ (i, j]` — rows **below** `i`, which the *decreasing* `i` loop writes in later iterations.
  The `i` loop therefore carries a real dependence and **cannot be parallel**. The structure
  confirms the measurement rather than the other way round.
- **polycc's transformation is correct; only its OpenMP decoration is wrong.** Measured against the
  untransformed reference compiled directly, N=200: **0/20 runs differ on 1 thread; 20/20 differ on
  8 threads; 20/20 on 32.**
- This is the most dangerous failure mode in the column and **the only one that can pass
  validation**: a race corrupts a data-dependent fraction of the output, so at a small preset the
  kernel sometimes lands on the right answer and NPBench prints SUCCESS. Observed exactly that —
  two consecutive `run_benchmark.py -b nussinov -f pluto -p S` invocations, unchanged binary, one
  SUCCESS and one "Relative error: 0.0048".
- **Resolution**: the transform is kept **exactly as generated** and compiled **without
  `-fopenmp`**, which leaves the `#pragma omp parallel for` inert and runs Pluto's own tiled code on
  one thread. **Nothing in the generated C is edited.** Reported through the same
  `PlutoSequential` channel as durbin/ludcmp, with its own distinct stated reason.
- **Post-fix validation**: 27/27 through NPBench over presets S, M, L at 1, 8 and 72 threads, and
  0/48 mismatches against the untransformed reference at N=60…800.
- **Configurations swept and rejected**: `--lastwriter`, `--rar`, `--nofuse`, `--innerpar` and
  `--second-level-tile` all still race. `--second-level-tile` is the cautionary one — clean over
  9 runs, wrong in **72 of 288** under stress. `--multipar` survived 504 runs at N=120…900 on 4…72
  threads with a full-table FNV hash and never differed; it is **recorded but not adopted**,
  because it rests on the same dependence analysis that is demonstrably wrong here, and 504 clean
  runs is evidence, not soundness.
- Every other tracked kernel was screened for the same failure mode — each run once sequentially
  and three times on 32 threads, outputs compared exactly — and all are deterministic.

### 5.6 Finding 5 — durbin and ludcmp (single-threaded)

Both produce **correct Pluto transformations**; Pluto simply finds **no legal, useful parallel
loop**. They are tiled but sequential. Reported as validated single-threaded Pluto results,
explicitly labelled as such in the summary and in the report PDF, so they are never read as
parallel numbers.

### 5.7 Finding 6 — adi (the only remaining decline)

adi's semantics are canonical on all sides. Re-investigated **from scratch**, explicitly including
the hypothesis that the NumPy or DaCe side was still non-canonical:

- **It is not.** Both match untransformed canonical PolyBench/C across **nine (N, TSTEPS)
  combinations** — worst relative error **1.3e-13** for NumPy and **5.6e-14** for DaCe, i.e.
  floating-point reassociation and nothing else.

Two separate defects sit between adi and a correct transform:

**(1) Clan transposes adi's scop parameters.** Clan's own scop for `v[0][i] = 1.0` carries
`-t+N >= 0` and `-i+TSTEPS-2 >= 0`, when the loops are `t = 1..TSTEPS` and `i = 1..N-1` — the two
parameters are swapped in the domain constraints. polycc then emits `if (TSTEPS >= 3)` guarding
spatial loops bounded by `TSTEPS-2`: at TSTEPS=1 the whole scop is skipped and `u` is returned
untouched, and at TSTEPS > N it writes past the arrays (**measured: SIGSEGV at N=8, TSTEPS=20**).
The trigger is that adi's parameters first appear inside **non-affine casts**
(`1.0/(DATA_TYPE)_PB_N`) ahead of any loop. Two independent bypasses were built and verified:
hoisting the loop-invariant scalar block above `#pragma scop` (bit-identical to canonical at every
size tested), and expanding the kernel's own macros before Clan sees them (Clan does not run the C
preprocessor, so `(DATA_TYPE)` never parses as a cast). The macro expansion was proved
**semantics-neutral, bit-identical to canonical at `-O0/-O1/-O2/-O3` over 3600 values**, and made
Clan extract all **27 statements**.

**(2) Pluto orders the back-substitution before its producer.** With (1) bypassed, polycc still
emits, inside one tile, the reversed `j` loop that **reads `p` and `q` ahead of the forward loop
that writes them**, so `v` is computed from zeros.

- **1514 of 1600 values differ** from the untransformed canonical at N=40, T=5.
- The error is **identical** under `--tile`, `--parallel`, `--nofuse`, `--maxfuse`, `--lastwriter`,
  `--rar`, `--innerpar`, `--nointratileopt`, `--nodiamond-tile`, **and under plain polycc with
  neither tiling nor parallelization**.
- Identical on **both frontends**, and identical on **1, 8 and 32 threads** — therefore a **wrong
  schedule, not a race**.
- Already wrong at **TSTEPS=1**.
- Identical again after lifting `v`/`p`/`q` and the scalars to parameters, which is itself
  bit-identical untransformed.

Additional bounding evidence: the toolchain axis was closed by comparing the scheduler source
across Pluto revisions — Pluto 0.12.0's scheduler is byte-identical to the measured `dc46216`
(the only differences are `+1` line in `lib/constraints_isl.c` and `tool/osl_pluto.c`), so no
adjacent release can change this result. Pluto 0.11.x was not tested and is flagged as untested.

**Final classification: a Pluto toolchain limitation** (scheduling), downstream of a Clan frontend
defect, and not a semantics problem in NPBench's port, in the NumPy implementation, in DaCe, or in
the harness. The failure is *not* attributable to insufficient debugging: the semantic axis, the
frontend axis, the option axis, the tiling/parallelization axis, the thread-count axis, the
parameter-lifting axis and the Pluto-revision axis were each eliminated with measurements. The
macro-expansion preprocessing step was **not integrated**, because it would change the source fed
to Clan for all 23 kernels and buy nothing — adi is wrong on the far side of it.

Supervisor decision: **not to pursue adi further.** It is reported as declined, with its reason
stated at the correct stage in the report.

### 5.8 What was never done in the Pluto column

- No generated C was patched or hand-edited.
- No loops were manually reordered.
- No numerical tolerance was weakened.
- No silent fallback to untransformed/original C.
- No output values hard-coded.
- No per-kernel hack applied where a general fix was available; the per-kernel entries that exist
  (`ARG_OVERRIDES`, `PLUTO_ADAPTERS`) are argument-mapping facts read off the two sources, and the
  refusal tables (`SEMANTIC_DIVERGENCE`, `UNSOUND_PARALLELIZATION`) *decline* kernels rather than
  fixing them.

---

## 6. Correctness and coverage

### 6.1 Final coverage (from `results/paper-final50-4499210/summary.txt`)

```
preset paper -- 23 kernels x 3 frameworks

framework            validated   invalid  declined   error  missing
numpy                       23         0         0       0        0
pluto                       22         0         1       0        0
dace_cpu_autoopt            23         0         0       0        0
```

- **NumPy: 23/23**
- **DaCe auto_opt: 23/23**
- **Pluto: 22/23** — `adi` declined, reason recorded verbatim:
  `polycc failed on adi_pluto_reference.c: [Clan] Error: syntax error at line 33, column 39.`
- **Validated but single-threaded Pluto results (labelled as such): `durbin`, `ludcmp`,
  `nussinov`.**

### 6.2 Validation discipline

- Every non-NumPy row is validated against the NumPy reference on first execution; the verdict is
  stored per row in the `validated` column of `npbench.db`.
- **Invalid results never contribute timings.** In the final campaign all 3400 recorded rows carry
  `validated = 1` (verified by direct query), and `stats.json` records `validated_rows_only: true`,
  `excluded_rows: 0`.
- A decline is **recorded, never forced**: the framework raises `PlutoUnavailable` with a named
  cause, the launcher captures it per (kernel, framework) into `status.json`, and the plot marks
  the row rather than leaving an ambiguous gap. Forcing a timing there would mean timing something
  polycc did not produce.
- No untransformed fallback, no tolerance weakening, no manual generated-C patching (see §5.8).

---

## 7. Final measurement protocol

### 7.1 Campaign progression

| Campaign | Job | Date | Node | Preset | REPEAT | NPBench commit | Purpose |
|---|---|---|---|---|---|---|---|
| M verification | 4490818 | 2026-08-15 | nid006513 | M | 1 | `5e65cf7` | correctness verification only; explicitly *not* a performance measurement |
| L final | 4492759 | 2026-08-16 | nid006513 | L | 20 | `3b49520` | first performance campaign |
| Paper final | 4492789 | 2026-08-16 | nid005312 | paper | 20 | `497c926` | preset comparison |
| **Paper final ×50** | **4499210** | **2026-08-17** | **nid005315** | **paper** | **50** | **`7f49c1a`** | **the reported dataset** |

DaCe was `extended @ eb7b1352a` and polycc was `0.12.0-33-gdc46216` in **all four**.

### 7.2 Final campaign configuration

| Item | Value |
|---|---|
| Slurm job | **4499210** |
| Preset | **`paper`** |
| REPEAT | **50** |
| Timeout | 600 s per (kernel, framework) |
| Node / ranks | 1 node, 4 ranks, 72 CPUs per rank, 288 total, `--cpu-bind=cores`, `--exclusive` |
| Affinity | as §1.2, disjoint per rank |
| Frameworks | `numpy pluto dace_cpu_autoopt` |
| Result directory | `~/npbench/results/paper-final50-4499210/` |
| Working tree | clean (0 dirty files in both NPBench and DaCe) |

Example preset sizes (from `bench_info/`): gemm `NI=2000, NJ=2300, NK=2600`; heat_3d
`TSTEPS=500, N=120`; nussinov `N=500`; trmm `M=1000, N=1200`; mvt `N=16000`.

### 7.3 Recorded rows

| Framework | Rows | Validated |
|---|---|---|
| numpy | 1150 | 1150 |
| dace_cpu_autoopt | 1150 | 1150 |
| pluto | 1100 | 1100 |
| **Total** | **3400** | **3400** |

`1150 = 23 × 50`, `1100 = 22 × 50` — **exactly 50 samples per valid pair**, 68 valid pairs.
No failed or invalid timing entered the statistics.

### 7.4 What is and is not timed

The recorded `time` is **kernel runtime only**. Compilation, polyhedral transformation, SDFG
construction and JIT/build cost are **excluded** from it — the Pluto pipeline is built and cached
before timing (cached on mtime under the per-rank build root), and DaCe's build is separate from
its measured execution. This is a like-for-like *runtime* comparison, not a
time-to-first-result comparison.

---

## 8. Statistical methodology

### 8.1 Initial approach

- median runtime per (kernel, framework) pair,
- IQR as the dispersion measure,
- **95% percentile IID bootstrap** CI of the median.

### 8.2 Independent audit of the CIs

The intervals were recomputed from raw samples and cross-checked against **BCa bootstrap** and the
**exact order-statistic (sign-test) interval**. The intervals were numerically correct, but the
**IID assumption was not justified**. Measured over all 68 pairs (n = 50 each):

| Diagnostic | Count |
|---|---|
| pairs flagged autocorrelated (lag-1) | 32 / 68 |
| pairs flagged drifting (Spearman ρ vs sample index) | 17 / 68 |
| **union — pairs with meaningful temporal dependence** | **37 / 68** |
| pairs flagged possibly multimodal | 27 / 68 |
| pairs with ≥1 sample beyond 3×IQR | 38 / 68 |
| pairs flagged wide-CI | 3 / 68 |
| pairs flagged wide-IQR | 1 / 68 |

Observed dependence forms: positive autocorrelation (slow warm-up / thermal drift), **negative**
lag-1 autocorrelation (alternating two-state behaviour, e.g. heat_3d DaCe at r₁ = −0.58), monotone
drift, and genuine bimodality.

Effect on the intervals: the moving-block CI is wider than the IID CI on most pairs — median width
ratio **1.10×**, with a long tail: ludcmp/NumPy **2.99×**, lu/DaCe **2.64×**,
floyd_warshall/NumPy **2.37×**, gemver/NumPy **2.30×**, lu/NumPy **2.10×**. So the IID interval
understates uncertainty by up to a factor of three on the worst pairs.

### 8.3 Final reported method

| Parameter | Value |
|---|---|
| Point estimate | **median** of the 50 samples |
| Dispersion | **IQR** (and IQR/median) |
| Interval | **95% moving-block bootstrap CI of the median** |
| Block length | **6** |
| Resamples | **10 000** |
| Seed | **20260817** (fixed, reproducible) |
| Outlier removal | **none** |
| Rows used | validated only; `excluded_rows: 0` |

Implemented in `slurm/stats.py`; per-pair output in `stats.json` and `stats.csv`, which carry both
the IID (`ci95_median_*`) and the block (`ci95_block_*`) intervals side by side, plus
`lag1_autocorr`, `drift_spearman`, `iqr_over_median`, `max_over_min`, `largest_gap_frac`,
`n_outliers_3iqr` and a `flags` list per pair.

### 8.4 Interpretation caveats to state explicitly

- The CI describes **uncertainty in the median**, not a range containing 95% of samples. A tight CI
  around a visibly bimodal distribution is not a claim that the runtime is stable.
- The **block bootstrap was chosen to preserve short-range serial dependence**, which the IID
  bootstrap destroys by resampling individual observations.
- **Block length 6 is a methodological judgement**, based on ACF decay across the 68 pairs — not a
  value derived from an optimality criterion. Results should not be presented as insensitive to it
  without a sensitivity check.
- All 50 samples of a pair come from **one Slurm job on one node**. The interval therefore captures
  within-run variability only, **not** across-allocation variability (node-to-node differences,
  differing co-tenancy, different boot state).

### 8.5 Notable distributions (from `stats.json`)

| Pair | Median | Shape |
|---|---|---|
| heat_3d / DaCe | 60.87 ms | bimodal, **negative** lag-1 (r₁ = −0.58) → alternating states; flagged `wide-CI`; IQR/median 8.5% |
| gramschmidt / DaCe | 201.0 ms | bimodal (lobes ≈200.5 and ≈201.8 ms), strong positive autocorrelation r₁ = +0.75 with drift ρ = +0.79 |
| mvt / Pluto | 19.74 ms | two-state: dense cluster ≈19.7 ms plus a distinct upper lobe ≈23–24.5 ms; the single `wide-IQR` pair (11.5%) |
| lu / DaCe | 2.115 s | bimodal (≈2.09 and ≈2.12 s), r₁ = +0.88, drift ρ = −0.78 |
| seidel_2d (all three) | 5.878 s / 56.87 ms / 816.8 ms | exceptionally tight — Pluto IQR/median 0.11%, DaCe 0.09% |
| NumPy, several kernels | — | moderate positive lag-1: trmm +0.59, lu +0.57, seidel_2d +0.57, cholesky +0.41, heat_3d +0.66 |

---

## 9. Final performance results

All values from `results/paper-final50-4499210/stats.json`. Speedup = median NumPy runtime /
median implementation runtime, computed per kernel from the medians of 50 validated samples.

### 9.1 Aggregate

| Aggregate | Value | Denominator |
|---|---|---|
| **Pluto geometric mean speedup** | **16.0×** | its **22** valid kernels |
| **DaCe geometric mean speedup** | **7.25×** | all **23** kernels |
| **DaCe over the same 22 kernels** | **8.54×** | the 22 Pluto also answers |

**Denominator warning — state this explicitly in the thesis.** Pluto's 16.0× is over 22 kernels;
DaCe's 7.25× is over 23. Those two numbers are **not** a head-to-head comparison. The like-for-like
statement is **Pluto 16.0× vs DaCe 8.5× over the same 22 kernels**. The difference between DaCe's
7.25× and 8.54× is entirely `adi`, where DaCe scores 0.20× — i.e. DaCe's 23-kernel figure is
dragged down by exactly the kernel Pluto declines, so quoting 16.0× against 7.25× would flatter
Pluto twice over.

### 9.2 Full per-kernel table (medians, 50 samples each)

| Kernel | NumPy | Pluto | DaCe auto_opt | Pluto speedup | DaCe speedup |
|---|---|---|---|---|---|
| adi | 360.5 ms | *declined* | 1789 ms | — | 0.20× |
| atax | 20.34 ms | 23.36 ms | 23.09 ms | 0.87× | 0.88× |
| bicg | 19.96 ms | 23.65 ms | 22.03 ms | 0.84× | 0.91× |
| cholesky | 2567 ms | 60.13 ms | 326.9 ms | 42.7× | 7.85× |
| correlation | 29.43 ms | 9.48 ms | 39.90 ms | 3.10× | 0.74× |
| deriche | 1123 ms | 441.4 ms | 489.7 ms | 2.54× | 2.29× |
| durbin | 295.6 ms | 121.9 ms *(seq.)* | 431.1 ms | 2.42× | 0.69× |
| fdtd_2d | 2253 ms | 40.98 ms | 29.54 ms | 55.0× | 76.3× |
| floyd_warshall | 14009 ms | 2139 ms | 5708 ms | 6.55× | 2.45× |
| gemm | 26.74 ms | 111.5 ms | 13.49 ms | **0.24×** | 1.98× |
| gemver | 183.3 ms | 6.23 ms | 9.64 ms | 29.4× | 19.0× |
| gramschmidt | 47.27 ms | 3.67 ms | 201.0 ms | 12.9× | **0.24×** |
| heat_3d | 10238 ms | 536.5 ms | 60.87 ms | 19.1× | **168×** |
| jacobi_2d | 50811 ms | 270.0 ms | 166.9 ms | 188× | 304× |
| lu | 7806 ms | 97.65 ms | 2115 ms | 79.9× | 3.69× |
| ludcmp | 8048 ms | 2243 ms *(seq.)* | 2113 ms | 3.59× | 3.81× |
| mvt | 13.41 ms | 19.74 ms | 16.50 ms | 0.68× | 0.81× |
| nussinov | 7126 ms | 10.82 ms *(seq.)* | 11.84 ms | **658×** | 602× |
| seidel_2d | 5878 ms | 56.87 ms | 816.8 ms | 103× | 7.20× |
| syr2k | 6823 ms | 27.55 ms | 21.87 ms | 248× | 312× |
| syrk | 2868 ms | 17.28 ms | 9.23 ms | 166× | 311× |
| trisolv | 103.6 ms | 6.76 ms | 99.79 ms | 15.3× | 1.04× |
| trmm | 1743.6 ms | 3.29 ms | 17.46 ms | 531× | 99.9× |

*(seq.)* = validated single-threaded Pluto result (Pluto found no legal parallel loop for durbin
and ludcmp; nussinov's parallel decoration was unsound and was compiled inert — see §5.5–5.6).

### 9.3 Medians with 95% moving-block CIs, selected kernels

| Kernel | NumPy | Pluto | DaCe auto_opt |
|---|---|---|---|
| gemm | 26.74 ms | 111.5 ms | 13.49 ms |
| heat_3d | 10.238 s | 536.5 ms | 60.87 ms [wide, bimodal] |
| trmm | 1744 ms [1739, 1749] | 3.285 ms [3.276, 3.297] | 17.46 ms [17.42, 17.49] |
| cholesky | 2567 ms [2565, 2570] | 60.13 ms [60.03, 60.19] | 326.9 ms [326.4, 327.8] |
| gramschmidt | 47.27 ms [47.23, 47.33] | 3.666 ms [3.644, 3.677] | 201.0 ms [200.6, 201.8] |
| lu | 7806 ms [7759, 7831] | 97.65 ms [97.62, 97.66] | 2115 ms [2092, 2120] |
| seidel_2d | 5878 ms [5875, 5881] | 56.87 ms [56.85, 56.88] | 816.8 ms [816.7, 817.1] |
| mvt | 13.41 ms [13.27, 13.58] | 19.74 ms [19.67, 19.81] | 16.50 ms [16.33, 16.59] |

Full per-pair intervals for all 68 pairs are in `stats.csv` / `stats.json`.

---

## 10. Limitations and interpretation

### 10.1 Interpretation — kept separate from measurement

These are readings of the numbers, not additional measurements. None of them is encoded into the
plotted data; where they appear in the report they appear as short neutral captions.

- **Pluto can dominate on several affine, loop-heavy kernels**: nussinov 658×, trmm 531×,
  syr2k 248×, jacobi_2d 188×, syrk 166×, seidel_2d 103×, lu 79.9×.
- **DaCe dominates on some stencil / BLAS-shaped kernels**: heat_3d 168× vs Pluto's 19.1×,
  jacobi_2d 304× vs 188×, syrk 311× vs 166×, syr2k 312× vs 248×, fdtd_2d 76.3× vs 55.0×.
- **GEMM is a clean counterexample**: Pluto **0.24×** — its generic tiled loop nest loses badly to
  the optimized matrix-multiplication path NumPy reaches through BLAS, while DaCe manages 1.98×.
  A polyhedral scheduler is not competitive with a hand-tuned GEMM microkernel.
- **Low arithmetic-intensity, BLAS-2-shaped kernels show little or no gain** for either
  implementation: atax (0.87× / 0.88×), bicg (0.84× / 0.91×), mvt (0.68× / 0.81×) — all at or
  below the NumPy baseline. These are memory-bandwidth bound; loop transformation has little to
  work with.
- **Single-threaded Pluto results can still beat multi-threaded DaCe.** nussinov: Pluto 658×
  *on one thread* vs DaCe 602×. durbin: 2.42× (sequential) vs DaCe 0.69×. Thread count alone does
  not determine performance — locality and the transformed schedule can dominate it.
- **Coverage and profitability are distinct axes.** Pluto answers 22 of 23 kernels; it is *faster*
  than DaCe on some and much slower on others. A high geometric mean does not imply broad
  applicability, and broad coverage does not imply profitability.
- **No universal-superiority claim is supported by this data.** Each framework wins a
  characterisable class of kernels.

### 10.2 Limitations

1. **One node, one architecture.** All results are from a single NVIDIA Grace (aarch64) node on
   CSCS Alps/Daint. Nothing here generalises to x86, to GPUs, or to multi-node execution.
2. **One preset.** Only the `paper` preset is reported. Problem-size sensitivity is not
   characterised, and several kernels are small enough at this preset that fixed overheads matter
   (e.g. mvt medians of 13–20 ms).
3. **Four ranks share node-level resources.** The ranks have disjoint cores, but they share memory
   bandwidth, the memory controllers, LLC-adjacent resources and power/thermal headroom. A
   memory-bound kernel is therefore measured under co-tenancy from three other kernels, not in
   isolation. This is a plausible contributor to the observed autocorrelation, drift and bimodality.
4. **50 sequential samples from one campaign.** Samples within a pair are contiguous in time and
   demonstrably not IID (37/68 pairs).
5. **The moving-block CI does not capture across-job variability.** It quantifies uncertainty
   within one allocation only. No repeat allocation was measured, so node-to-node and
   run-to-run-allocation variance is unquantified.
6. **Block length 6 is a judgement call** (§8.4), not an optimised parameter.
7. **adi is not supported by Pluto** in this configuration, so Pluto's aggregate is over 22 of 23
   kernels and every head-to-head aggregate must state its denominator.
8. **Three Pluto rows are single-threaded** (durbin, ludcmp, nussinov) while the corresponding
   NumPy and DaCe rows use the full 72-CPU rank. They are labelled, but they are not
   thread-count-matched comparisons.
9. **The NumPy baseline may invoke tuned numerical libraries.** NumPy dispatches to
   `scipy-openblas` 0.3.34 with Neoverse-V2 kernels for the BLAS-shaped kernels, so the baseline is
   not "naive Python" — it is a strong, vendor-tuned baseline on some kernels and an unoptimised
   array-expression baseline on others. This asymmetry is the main reason GEMM and the BLAS-2
   kernels look so different from the stencils.
10. **Two BLAS libraries are in play** (NumPy's bundled 0.3.34 vs the uenv's 0.3.30 linked by
    DaCe- and Pluto-generated C), and NumPy's build carries `MAX_THREADS=64` against a 72-CPU rank.
11. **The experiment is not exactly vanilla NPBench** for `adi` and `deriche`: both ports were
    corrected against canonical PolyBench/C 4.2.1 (§2.4b). The corrections are applied uniformly to
    NumPy and DaCe, are documented, and the pre-correction state is in git history — but any
    comparison to published NPBench numbers for those two kernels must account for it.
12. **Pluto times the PolyBench/C original, not a translated NumPy port** — deliberate and
    documented, but it means the Pluto column and the NumPy column start from different source
    artifacts of the same algorithm. Validation against the NumPy reference is what bridges them.
13. **Runtime only.** Compilation and transformation time are excluded (§7.4). Pluto's and DaCe's
    build costs are real and are not represented in any reported number.

---

## Artifacts

All under `~/npbench/results/paper-final50-4499210/`:

| File | Content |
|---|---|
| `manifest.json` | full provenance (§1.4, §1.7) |
| `npbench.db` | merged result DB, 3400 validated rows |
| `status.json`, `summary.txt` | per-pair classification and decline reasons |
| `kernels.txt` | the 23-kernel corpus, derived from the tree |
| `stats.json`, `stats.csv` | per-pair statistics, both CI methods, all diagnostics |
| `logs/<kernel>.<framework>.log` + `.rc` | full per-pair log and process exit code |
| `rank-0/` … `rank-3/` | per-rank working directories and databases |
| `slurm-4499210.out` | job log, includes the measured per-rank affinity |
| `paper-final50-4499210-thesis.pdf` | 3-page report (preserved) |
| **`paper-final50-4499210-thesis-extended.pdf`** | **5-page report — the main artifact** |
| `paper-final50-4499210-thesis-extended-p{1..5}.png` | page previews |
| `ci-audit-sequences.png` | sample-vs-index sequences from the CI audit |

Extended PDF page layout:

| Page | Content |
|---|---|
| 1 | heatmap overview — speedups per kernel × implementation, with the toolchain caption (`polycc --tile --parallel` (Clan frontend), `clang -O3 -march=native`) |
| 2 | diagnostics — coverage table, every decline with its reason at the correct stage, single-threaded labels |
| 3 | violin plots: **gemm, heat_3d** |
| 4 | violin plots: **trmm, cholesky, gramschmidt** |
| 5 | violin plots: **lu, seidel_2d, mvt** |

Violin conventions, identical on pages 3–5: all 50 raw samples shown as points, KDE outline,
median marked, **95% moving-block bootstrap CI** drawn, lag-1 autocorrelation annotated where
significant, **per-implementation y-ranges** (the three implementations of one kernel span up to
three orders of magnitude, so a shared axis flattens every distribution to a line),
`n = 50` stated, actual runtime units, and **no outlier removal**.

Code artifacts on branch `bench/daint-pluto`:

| File | Role |
|---|---|
| `npbench/infrastructure/pluto_framework.py` | the Pluto column, plus all recorded defect evidence |
| `npbench/infrastructure/dace_autoopt_framework.py` | the `dace_cpu_autoopt` narrowing |
| `framework_info/dace_cpu_autoopt.json` | framework registration |
| `slurm/run_campaign.sbatch` | the campaign launcher (all four campaigns) |
| `slurm/npbench-env.sh` | environment, with fatal checks on every required tool |
| `slurm/merge_db.py` | per-rank DB merge |
| `slurm/collect_status.py` | per-pair classification, decline-reason capture |
| `slurm/stats.py` | bootstrap and diagnostics |
| `slurm/dispersion.py` | median / IQR / IQR-over-median / max-over-min |
| `slurm/plot_thesis.py` | the report PDF |
| `slurm/README.md` | campaign base, environment, Pluto coverage narrative |

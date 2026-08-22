# Why Pluto beats DaCe auto_opt on the kernels where it wins — investigation

Performance numbers: job **4499210** (Paper x50). Generated code: the campaign's own build tree
`/capstor/scratch/cscs/levwidmer/m-verify-4499210/pluto-rank-*/`, plus DaCe auto_opt SDFGs
regenerated through `DaceFramework`'s exact path. Diagnostic runs: jobs **4512040**, **4512060**
(3 reps, thread-count sweeps only). No measured code changed; nothing committed.

## 1. Overall finding

**The hypothesis is largely NOT confirmed.** The gap is not mainly Pluto finding schedules with
better locality. Measured serially, Pluto's schedule is *worse* than DaCe's on two of the five
priority kernels (seidel_2d 135 ms vs 57 ms; cholesky 522 ms vs ~327 ms).

The recurring cause is **where each tool places the parallelism**:

- **Pluto** hoists exactly one `omp parallel for` to the outermost legal (tile) level. Tiling and
  skewing matter mainly as the *enablers* of that outer parallel loop, not as locality wins.
- **DaCe auto_opt** parallelizes the innermost array/slice operation and leaves it nested inside
  sequential loops — so it pays an OpenMP fork/join per inner operation — **or does not
  parallelize at all** (lu, cholesky, trisolv emit zero parallel regions).

Measured fork/join cost at 72 threads on this node: **11.05 us** per parallel region (398 elems,
stride 1) and **11.13 us** (240 elems, stride 200). At 1 thread: 0.50 / 0.37 us.

Attribution from the kernels themselves (DaCe 72-thread minus DaCe 1-thread, divided by the
region count) agrees with that microbenchmark to within ~15%:

| kernel | parallel regions entered | overhead measured in-kernel | microbenchmark |
|---|---|---|---|
| seidel_2d | 2 x 99 x 398 = 78,804 | (800-57) ms / 78,804 = **9.4 us** | 11.05 us |
| gramschmidt | 200 + 19,900 = 20,100 | (209-11) ms / 20,100 = **9.9 us** | 11.13 us |

DaCe's parallelization is **net harmful** on both: seidel_2d 57 ms -> 800 ms (14x slower),
gramschmidt 11 ms -> 209 ms (19x slower) going from 1 to 72 threads.

Confirmed structurally: `auto_optimize`'s own docstring lists Simplify, LoopToMap, greedy fusion,
**tiled write-conflict resolution only**, MapCollapse, and fast library selection. It has no
general loop tiling, no skewing, no interchange, no unroll-and-jam. Static scan of 11 generated
DaCe kernels: `tiled=0 skewed=0 unrolled=0` on **all 11**. Pluto: tiled on 8/9, unroll-and-jam on
7/9, skewed on 2/9.

## 2. Summary table

| kernel | Pluto adv. (DaCe/Pluto) | key Pluto transformation | DaCe auto_opt behaviour | likely cause | confidence |
|---|---|---|---|---|---|
| gramschmidt | 54.8x | interchange j inward (stride-1 inner), 32-tiles, unroll-and-jam x8, 1 parallel region per k | ddot + `omp parallel for` over M, both column-strided (stride N), nested in (k,j) | ~20,100 fork/joins (19x self-inflicted) x ~2.8x worse serial schedule | high |
| lu | 21.7x | 3D 32-tiling, wavefront `omp parallel for` on tile index, unroll-and-jam x8, stride-1 inner | **zero** parallel regions; O(N^2) `cblas_ddot` with one stride-N operand | no parallelism at all + strided BLAS-1 | high |
| trisolv | 14.8x | 2D wavefront tiling of the substitution, unroll x8 giving x-register reuse (BLAS-1 -> BLAS-2-like) | **zero** parallel regions; N sequential `cblas_ddot` | no parallelism at all; 19x of Pluto's win is threads | high |
| seidel_2d | 14.4x | **time skewing** (i=t5-t4, j=t6-t4-t5) + 3D 32-tiling + wavefront `omp parallel for` | 2 tiny `omp parallel for` (len N-2) nested in (t,i) + materialized temp row | 78,804 fork/joins; DaCe's *serial* code is 2.4x faster than Pluto's serial code | high |
| cholesky | 5.4x | 3D 32-tiling, wavefront parallel tile loop, unroll-and-jam x8 | **zero** parallel regions; O(N^2) `cblas_ddot`, both operands stride-1 | no parallelism; DaCe serial is *better* than Pluto serial (327 vs 522 ms) | high |
| trmm | 5.3x | 32-tiling, 2 top-level parallel regions, unroll x8, near-linear scaling (36x on 72 threads) | `cblas_ddot` per (i,j) with stride-M and stride-N operands, inside sequential i | strided BLAS-1 + 1,000 fork/joins (~63% of its runtime) | high |
| floyd_warshall | 2.7x | 2D tiling of (i,j), parallel tile loop, unroll x8, **in place** | full N x N temporary + `memcpy` of 31 MB **per k** (2,800 times) | ~260 GB of avoidable data movement; fork/join negligible here | high |
| durbin | 3.5x | none — Pluto emits **zero** parallel regions (sequential, skewed) | 2 `omp parallel for` per k iteration = ~32,000 fork/joins | Pluto wins *with no parallelism at all*; purely DaCe's fork/join | high |
| correlation | 4.2x | 9 top-level parallel regions, tiled, unroll | 7 top-level regions (granularity fine) + full N x M `expr_pow_2` temporary, then sequential `cblas_dgemv` loop | materialized temporaries / extra passes, not fork/join | medium |

## 3. Per-kernel notes

**seidel_2d.** Pluto applies textbook dependence-aware **time skewing**: the statement is indexed
`A[-t4+t5][-t4-t5+t6]`, i.e. `i = t-shifted`, `j = 2t+i+j`, then tiles 32x32x32 and puts
`omp parallel for` on the middle tile index inside the sequential wavefront index `t1` — **19
parallel regions total**. DaCe cannot do this and does not try: it keeps the port's `t` and `i`
loops sequential (correctly — Gauss-Seidel carries a real row dependence), emits two
`omp parallel for` of length N-2 *inside* them, and materializes an N-2 temporary between them.
That is 78,804 fork/joins for 125 Mflop of work.
*But the honest reading is the opposite of the hypothesis*: DaCe at 1 thread is **57 ms**, Pluto at
72 threads is **58 ms**. Pluto's skewing + tiling + 72 threads buys essentially nothing over DaCe's
own serial code; the entire 14.4x gap is DaCe's harmful parallelization. Pluto's skewed code is in
fact 2.4x *slower* serially (135 ms) — skewing costs vectorization and locality, and only pays for
itself because it exposes an outer parallel loop (2.3x on 72 threads).
Caveat: the two run different source formulations (see §4).

**lu.** DaCe emits **no `#pragma omp` at all** — verified both statically and by measurement
(1 thread 2121 ms, 8 threads 2135 ms, 72 threads 2139 ms: perfectly flat). Each (i,j) does
`cblas_ddot` with `A_1 = &A[j]`, **stride N** — a column walk, one cache line per element. Pluto
tiles 32^3, parallelizes the tile loop under a wavefront, and unroll-and-jams by 8 so eight rows
share the loaded `A[t5][t6]`, with the inner loop stride-1. Pluto: 1294 ms -> 99 ms (13x scaling).
Both effects present; missing parallelism is the larger one.

**trmm.** The cleanest "BLAS-1-ization" case, and it is from the measured campaign build, not a
regeneration: DaCe calls `cblas_ddot(M-i-1, A_0, M, loop_body_B_0, N)` — **both operands strided**
— once per (i,j), inside a sequential i loop with an `omp parallel for` over j (1,000 fork/joins,
~63% of its 17.5 ms). Pluto tiles and scales 36x (109 ms -> 3 ms).

**cholesky.** Same shape as lu — zero parallel regions, O(N^2) `cblas_ddot` — but here both ddot
operands are stride-1 (row x row), so DaCe's serial code is genuinely good: **327 ms vs Pluto's
522 ms serially**. Pluto wins only by parallelizing (11x, 522 -> 47 ms). This kernel is the
strongest single counterexample to "Pluto finds better schedules": its schedule is *worse*, it just
runs on 72 cores.

**gramschmidt (why DaCe loses to NumPy).** The port is column-oriented: `np.dot(Q[:,k], A[:,j])`
and `A[:,j] -= Q[:,k]*R[k,j]`. DaCe preserves that literally — `cblas_ddot(M, Q_0, N, A_3, N)`
(both **stride N**) plus an `omp parallel for` over M, also stride N, **inside the (k,j) loops** =
19,900 fork/joins over 240 elements each. NumPy runs the identical column-strided algorithm but
with *no* fork/join, so NumPy (47 ms) beats DaCe (201 ms). At 1 thread DaCe drops to **11 ms** and
beats NumPy 4x — so "DaCe slower than NumPy" is entirely a parallelization artifact, not a
vectorization or BLAS artifact. Pluto instead **interchanges** so the innermost loop runs over `j`
(stride-1 in row-major `A[i][j]`, marked `#pragma ivdep`), unroll-and-jams by 8 over `i`, and emits
one parallel region per `k` (~400 total). Pluto: 3.67 ms.

## 4. Is aggressive tiling/wavefront scheduling the recurring explanation?

**Partly, and not in the way the hypothesis states.** Separating the two effects by thread count:

| kernel | Pluto 1 thr | Pluto 72 thr | Pluto's parallel gain | DaCe 1 thr | DaCe 72 thr |
|---|---|---|---|---|---|
| seidel_2d | 135 ms | 58 ms | 2.3x | **57 ms** | 800 ms |
| gramschmidt | 4 ms | 3 ms | 1.3x | 11 ms | 209 ms |
| trisolv | 134 ms | 7 ms | **19x** | ~100 ms (no omp) | ~100 ms |
| cholesky | 522 ms | 47 ms | **11x** | ~327 ms (no omp) | ~327 ms |
| lu | 1294 ms | 99 ms | **13x** | 2121 ms | 2139 ms |
| trmm | 109 ms | 3 ms | **36x** | — | 17.5 ms |

Reading:
1. **Tiling/skewing as a locality win: not supported.** On seidel_2d and cholesky, Pluto's serial
   code is *slower* than DaCe's. Only lu (1.6x) and gramschmidt (2.8x) show Pluto with a genuinely
   better serial schedule, and there the mechanism is **interchange to a stride-1 inner loop plus
   unroll-and-jam**, not tile-level cache blocking.
2. **Tiling/wavefronting as a parallelism enabler: strongly supported.** Pluto's 11x-36x thread
   scaling on trisolv/cholesky/lu/trmm exists only because tiling created a coarse outer parallel
   loop over a dependence pattern (triangular solve, factorization) that DaCe's LoopToMap cannot
   turn into a map at all.
3. **The single largest recurring term is DaCe's parallel granularity**, which is a code-generation
   policy question, not a scheduling-power question. Two of the nine kernels (floyd_warshall,
   correlation) are explained instead by **materialized temporaries and extra data movement** —
   also a consequence of faithfully lowering NumPy array-expression semantics.

**Confounder that must be stated.** Pluto consumes the canonical PolyBench/C scalar loop nest;
NumPy and DaCe consume NPBench's array-expression port. These are different source formulations.
For seidel_2d the port is even a partial hand-rewrite (a vectorized 7-neighbour add plus a scalar
recurrence). For floyd_warshall the port's `np.add.outer` + whole-array assignment is what *forces*
DaCe's N x N temporary and per-k memcpy. So part of every gap is attributable to the port, not to
DaCe's optimizer. This experiment cannot separate the two without re-porting the kernels.

## 5. Assessment: should DaCe selectively invoke a polyhedral scheduler?

**The evidence does not support the idea in the form stated, and points at a cheaper fix first.**
The measured problem on seidel_2d, gramschmidt, durbin and trmm is not that DaCe's *schedule* is
poor — on seidel_2d and cholesky DaCe's serial schedule beats Pluto's. It is that DaCe emits
`omp parallel for` on maps nested inside sequential loops without a cost model, so it pays ~11 us
per region for microseconds of work. A granularity rule — do not open a parallel region whose
estimated work is below a fork/join threshold, or hoist the parallel level outward — would recover
seidel_2d 14x, gramschmidt 19x and durbin ~3x **with no polyhedral machinery at all**. DaCe's own
`extended` branch already implements exactly this rule for *library nodes*
(`libnode_is_sequential`: "never a contended `omp atomic` or a re-forked team per outer
iteration"); it simply does not cover ordinary maps. That is the highest-value finding here and it
is a much more defensible thesis contribution than a Pluto bridge.

Where a polyhedral scheduler *would* genuinely add capability is the second group: **lu, cholesky,
trisolv** (and trmm's tiling), where DaCe emits **zero** parallelism because `LoopToMap` cannot
prove a factorization's or triangular solve's loop is a map. Pluto's 11x-36x there comes from
wavefront tiling over dependences DaCe has no representation for. These regions are recognizable —
they are perfectly affine, dense, statically-bounded loop nests over `dace.data.Array` with affine
subscripts — and DaCe already has the pieces to detect them, since its `Map` ranges and `Memlet`
subsets are symbolic affine expressions. Export would mean emitting an OpenSCoP/isl domain +
access-relation description per candidate region (statement domains from loop bounds, read/write
relations from memlet subsets, a schedule from nesting order), and import would mean rebuilding a
nest of Maps and sequential loops from the returned isl schedule tree — the hard part is not the
export but the import, because a skewed/tiled schedule produces loop bounds (`max(ceild(...))`,
`min(floord(...))`) that DaCe's Map range representation must be able to express and that its code
generator must lower without losing the parallel annotation.

Deciding when to use it needs more than a heuristic. A static rule ("region is affine and
LoopToMap failed") covers lu/cholesky/trisolv but would *misfire* on seidel_2d and cholesky, where
Pluto's schedule is serially worse — so the polyhedral result must be treated as a *candidate*, not
a replacement. That implies measurement: build both, keep the faster, i.e. autotuning over a small
variant set, which DaCe's existing multi-variant infrastructure (`fusion`/`parallel`/`auto_opt`)
already resembles. Cost: polyhedral scheduling time is not free and NPBench's timings exclude
compilation, so a thesis claim here must be about *runtime* only unless build time is measured
separately. Honest scoping: the promising, defensible version of this idea is "**a polyhedral
scheduler as an additional candidate transformation for affine regions DaCe currently leaves
sequential, selected by measurement**", not "DaCe should call Pluto where auto_opt is weaker".

## 6. Thesis-safe claims

Safe to claim (each backed by generated code plus a measurement):

1. DaCe's `auto_optimize` performs no general loop tiling, skewing, interchange or unroll-and-jam;
   its only tiling is for write-conflict resolution. *(Its own docstring; `tiled=0 skewed=0
   unrolled=0` across 11 generated kernels; Pluto tiled on 8/9.)*
2. On lu, cholesky and trisolv, DaCe auto_opt emits no parallel region at all and the kernels run
   single-threaded. *(Static scan; lu measured flat at 2121/2135/2139 ms on 1/8/72 threads.)*
3. On seidel_2d and gramschmidt, DaCe's parallelization is net harmful: the same generated code is
   14x and 19x faster on one thread than on 72. *(Job 4512040.)*
4. The dominant cost in those two cases is OpenMP region entry, at ~11 us per region on 72 threads;
   region counts times that cost account for 85-89% of the measured runtime difference.
   *(Microbenchmark plus in-kernel arithmetic, two independent kernels agreeing.)*
5. Pluto's advantage on lu, cholesky, trisolv and trmm comes principally from wavefront-tiled
   parallelism (11x-36x thread scaling) over dependence patterns DaCe does not parallelize.
6. On floyd_warshall the gap is extra data movement: DaCe materializes an N x N temporary and
   memcpy's it once per k (2,800 x 31 MB), which Pluto's in-place tiled nest does not.
7. Pluto beats DaCe on durbin (3.5x) while emitting **zero** parallel regions — a direct
   demonstration that thread count does not determine performance.

Claims to avoid:

- **"Pluto finds better loop schedules / has better locality than DaCe."** Contradicted on
  seidel_2d (Pluto 135 ms vs DaCe 57 ms serially) and cholesky (522 vs ~327 ms).
- **"Tiling/skewing explains the gap."** It explains the availability of parallelism, not the
  serial quality; stating it as a locality win is unsupported.
- **"DaCe is slower than NumPy on gramschmidt because of DaCe's code quality."** At 1 thread DaCe
  is 4x *faster* than NumPy; the inversion is purely fork/join.
- **Any claim that isolates DaCe's optimizer from the port.** Pluto reads canonical PolyBench C,
  DaCe reads NPBench's array-expression port (and for seidel_2d a partially hand-vectorized one);
  floyd_warshall's temporary is forced by `np.add.outer` + whole-array assignment. The experiment
  cannot separate optimizer quality from source formulation.
- **Generalizing beyond this node/preset.** Fork/join cost scales with thread count; at 8 threads
  the same seidel_2d penalty is 3x rather than 14x, so the ranking is thread-count dependent.
- **"DaCe should call Pluto."** The measured first-order fix is a parallel-granularity cost model,
  which DaCe already applies to library nodes but not to maps.

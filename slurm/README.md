# Benchmark campaigns on CSCS Daint/Alps

Everything here is site-specific: the uenv name, the scratch paths and the Slurm account are
Daint's. Nothing in this directory is upstream NPBench.

A campaign runs a set of NPBench kernels across three frameworks on one node, validates every
result against the NumPy reference, and turns the raw timings into the statistics and figures
reported in the thesis.

```
run_campaign.sbatch          one node, 4 ranks x 72 CPUs, one kernel list
  -> npbench-env.sh          uenv + venv + clang/polycc on PATH
  -> run_benchmark.py        per (kernel, framework), validated against NumPy
  -> rank-N/npbench.db       one database per rank
  -> merge_db.py             -> npbench.db
  -> collect_status.py       -> status.json, summary.txt   (why a cell is empty)
  -> plot_m_verify.py        -> m-verify-<job>.pdf         (coverage overview)
  -> stats.py                -> stats.json, stats.csv      (REPEAT > 1 only)
  -> dispersion.py           -> dispersion.json            (REPEAT > 1 only)
  -> plot_thesis.py          -> <campaign>-<job>-thesis.pdf + -p1/-p2 .png
```

## Environment

`npbench-env.sh` activates the repo venv, puts `clang` and `polycc` on PATH, pins the compiler
DaCe hands to CMake, and refuses to continue if any of that is missing. It deliberately does not
set thread counts -- that depends on how many ranks split the node, so it belongs to the launcher.

Request the uenv from Slurm, so the mount reaches the batch step *and* every nested `srun`:

```bash
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default
```

Overridable: `NPBENCH_ENV_REPO`, `NPBENCH_ENV_LLVM_PREFIX`, `NPBENCH_ENV_PLUTO_BIN`.

NumPy uses the `scipy-openblas` build bundled in its wheel; DaCe- and Pluto-generated C link the
uenv's OpenBLAS. That bundled build carries `MAX_THREADS=64`, so a rank given 72 CPUs runs NumPy's
BLAS on at most 64 of them.

## Running a campaign

`run_campaign.sbatch` is the launcher for every campaign below. It is configured entirely through
the environment; the script itself is not edited per run.

| variable | meaning | default |
|---|---|---|
| `CAMPAIGN` | result directory name, `results/<CAMPAIGN>-<job>` | derived from `REPEAT` |
| `PRESET` | NPBench data-size preset | `M` |
| `REPEAT` | timed executions per pair; `> 1` enables statistics and the thesis report | `1` |
| `TIMEOUT` | per-pair execution timeout, seconds | `600` |
| `FRAMEWORKS` | frameworks to run | `numpy pluto dace_cpu_autoopt` |
| `KERNELS_FILE` | kernel list; default is every kernel carrying a tracked scop | generated |
| `CATEGORIES` | kernel grouping for the thesis figure | PolyBench's own |
| `THESIS_TITLE` | title on the figures | PolyBench wording |
| `DACE_TREE` | DaCe checkout to stamp on every row | `$HOME/dace` |

The namespace is refused if it already holds artifacts: NPBench *appends* to whatever
`npbench.db` it finds, so a reused directory would silently mix two campaigns.

`REPEAT=1` is a verification run. A median over one sample is that sample, so no statistics or
thesis report are produced.

## The two final thesis campaigns

**23 PolyBench-derived kernels** (job 4499210). No `KERNELS_FILE` is given: the launcher derives
the list from the tree, taking every kernel under `npbench/benchmarks/polybench` that carries a
tracked `<kernel>_pluto_reference.c`. That yielded 23 on the branch this campaign ran from; a
branch which has since added scops for further PolyBench-family kernels yields more, so pass an
explicit `KERNELS_FILE` to reproduce exactly these 23.

```bash
CAMPAIGN=paper-final50 PRESET=paper REPEAT=50 \
sbatch -A <project> --output=results/paper-final50-%j/slurm-%j.out slurm/run_campaign.sbatch
```

**31 additional NPBench kernels** (job 4523913) -- the complement of the 23 above.

```bash
CAMPAIGN=nonpoly-final50 PRESET=paper REPEAT=50 TIMEOUT=900 \
KERNELS_FILE=slurm/kernels_nonpolybench.txt \
CATEGORIES=slurm/categories_nonpolybench.json \
NPBENCH_PLUTO_POLYCC_TIMEOUT=900 NPBENCH_PLUTO_CLANG_TIMEOUT=900 \
sbatch -A <project> --output=results/nonpoly-final50-%j/slurm-%j.out slurm/run_campaign.sbatch
```

The two Pluto build limits cap polycc and clang separately, so a kernel that exceeds one is
recorded as a build timeout at the stage that timed out rather than blocking the corpus.

Job 4499210 predates the statistics and report steps being wired into the launcher; its
`stats.json` and thesis PDF were produced afterwards by running `stats.py` and `plot_thesis.py`
against its result directory. Its extended report adds the per-kernel distribution pages:

```bash
python slurm/plot_thesis.py --db <dir>/npbench.db --status <dir>/status.json \
    --kernels <dir>/kernels.txt --preset paper --repeat 50 \
    --dace-framework dace_cpu_autoopt --stats <dir>/stats.json --extra-violins \
    --output <dir>/<campaign>-<job>-thesis-extended.pdf \
    --png-prefix <dir>/<campaign>-<job>-thesis-extended
```

## Correctness gating

Every non-NumPy result is validated against the NumPy reference on its first execution and the
verdict is stored per row. A speedup is drawn only from a validated row; declined, crashed,
timed-out and unvalidated pairs stay empty in the figures and are explained on page 2 of the
report. `stats.py` reads validated rows only. Nothing falls back to untransformed code.

## Kernel selection

| file | contents |
|---|---|
| `kernels_nonpolybench.txt` | the 31 NPBench kernels outside the PolyBench-derived 23, keyed by `bench_info` stem (`conv2d_bias`, not `conv2d`) |
| `categories_nonpolybench.json` | groups those 31 by program structure for the figure's row order |

The 23-kernel list is not stored: the launcher derives it from the tracked scops, so it cannot
drift from what the Pluto column can be asked about.

## Statistics

`stats.py` reports, per validated pair: median, IQR, and a 95% moving-block bootstrap CI of the
median (block 6, 10000 resamples, seed 20260817), with no outlier removal. The block bootstrap is
used because the samples are sequential and many pairs show lag-1 autocorrelation or drift, which
an IID bootstrap would understate. Both intervals are written so the choice stays visible.
`dispersion.py` flags pairs whose spread makes a median unreliable.

## Derived datasets

`build_combined_dataset.py` assembles a dataset whose frameworks come from different campaigns.
It produced `results/nonpoly-final50-dace-fixed`, which is **derived, not a single Slurm campaign**:

| rows | source campaign |
|---|---|
| NumPy, Pluto | `results/nonpoly-final50-4523913` (job 4523913) |
| DaCe `auto_opt`, all 31 kernels | `results/nonpoly-dace50-4525319` (job 4525319) |

The DaCe column was re-measured in full at a newer DaCe revision and swapped in whole rather than
patched per kernel, so it stays internally consistent at one revision; NumPy and Pluto are
unchanged. Both source job ids and both DaCe revisions are written into the output, because such a
dataset is not reproducible from any single Slurm job. Both source directories are kept so the
combination can be rebuilt:

```bash
python slurm/build_combined_dataset.py \
    results/nonpoly-final50-4523913 results/nonpoly-dace50-4525319 <out>
```

then regenerate statistics and figures from `<out>` with `stats.py`, `dispersion.py` and
`plot_thesis.py`, exactly as for a real campaign.

## Supporting tooling

Not part of the final results; kept because each answers a question the thesis relies on.

| file | purpose |
|---|---|
| `verify_small.sbatch` | setup check: all three columns build, run and validate on a compute node at preset S, with the campaign's affinity |
| `verify_autoopt.sbatch`, `compare_autoopt.py` | equivalence check that `dace_cpu_autoopt` reproduces the `auto_opt` rows of the stock `dace_cpu`, which is what justifies using it as the DaCe column |

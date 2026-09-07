# Thesis result artifacts

`main` contains the retained measurements and analysis outputs. All campaigns used
the `paper` preset and 50 timed repetitions per successful kernel/framework pair.

| Directory | Role |
|---|---|
| `paper-final50-4499210` | Final measurements for 23 PolyBench-derived kernels; the `*-thesis-extended.pdf` is the five-page report. |
| `nonpoly-final50-4523913` | Original 31-kernel campaign; source of the final NumPy/Pluto measurements and provenance for the original DaCe results. |
| `nonpoly-dace50-4525319` | Complete DaCe remeasurement for all 31 kernels at a newer DaCe revision. |
| `nonpoly-final50-dace-fixed` | Final derived comparison: NumPy/Pluto from job 4523913, with the entire DaCe column replaced by job 4525319. This is not a single Slurm campaign. |

Each directory contains `manifest.json`, a merged `npbench.db`, `kernels.txt`,
`stats.json`/`stats.csv`, `status.json`, and PDF/PNG reports. Dispersion reports are
included where available. Source campaigns also retain `summary.txt`, Slurm output,
and `logs/<kernel>.<framework>.log` plus `.rc` exit codes. These explain failed or
declined pairs; only validated rows contribute to the reported statistics.
The derived dataset uses the source campaigns' logs and records both sources in
its manifest and its own README.

## Provenance and environment

Manifests record measurement revisions, software versions, node, and job settings.
The two nonpoly source manifests record dirty NPBench working trees: **1 file** in
job 4523913 and **2 files** in job 4525319. They do not identify those modifications.
The manifests are preserved unchanged, so the exact dirty source state cannot be
reconstructed solely from the recorded commit. The derived dataset inherits this
limitation. The paper campaign records clean NPBench and DaCe trees.

See [the campaign guide](../slurm/README.md) for Daint's uenv, launch procedure,
validation, affinity, and statistics methodology. Use each archived `kernels.txt`
as `KERNELS_FILE` when reproducing its corpus: the current launcher's default
selection can differ from the measured revision. Use a fresh output namespace.
DaCe is a separate checkout; its recorded revision is not supplied by this repository.

Additional recorded paper-campaign environment details: `setuptools<81` supplied
the `pkg_resources` dependency; NumPy's bundled OpenBLAS was 0.3.34 (maximum 64
threads), while generated C linked uenv OpenBLAS 0.3.30. The recorded Pluto build
was `bondhugula/pluto` at `dc46216`, using Clan, with submodule revisions:
`clan=fb9bd2c`, `pet=2320f24` (unused frontend), `isl=0114734`,
`cloog-isl=d9108b9`, `candl=08b7186`, `openscop=b79af02`, `piplib=261eec0`,
`polylib=597776a`. These pins retain the environment record; toolchain binaries
and build trees are not included.

## Regenerating analysis

Run from the repository root with the Python dependencies available. Write to a
new directory to preserve the archived artifacts. For the paper campaign:

```bash
artifact_dir=results/paper-final50-4499210
analysis_dir=$(mktemp -d)
python slurm/stats.py --db "$artifact_dir/npbench.db" --preset paper \
    --kernels "$artifact_dir/kernels.txt" --block 6 --resamples 10000 --seed 20260817 \
    --json "$analysis_dir/stats.json" --csv "$analysis_dir/stats.csv"
python slurm/dispersion.py --db "$artifact_dir/npbench.db" --preset paper \
    --kernels "$artifact_dir/kernels.txt" --json "$analysis_dir/dispersion.json" \
    > "$analysis_dir/dispersion.txt"
python slurm/plot_thesis.py --db "$artifact_dir/npbench.db" \
    --status "$artifact_dir/status.json" --kernels "$artifact_dir/kernels.txt" \
    --preset paper --repeat 50 --dace-framework dace_cpu_autoopt \
    --stats "$analysis_dir/stats.json" --extra-violins \
    --output "$analysis_dir/thesis-extended.pdf" --png-prefix "$analysis_dir/thesis-extended"
```

Rebuild the derived dataset from the retained sources into a nonexistent directory:

```bash
analysis_dir=$(mktemp -d)
python slurm/build_combined_dataset.py \
    results/nonpoly-final50-4523913 results/nonpoly-dace50-4525319 "$analysis_dir/combined"
```

Then run `stats.py` and `dispersion.py` as above with that directory as
`artifact_dir`. For a nonpoly overview, replace `--extra-violins` with
`--no-page3 --categories slurm/categories_nonpolybench.json` and supply an
appropriate `--title` to `plot_thesis.py`. Rebuilding updates generated provenance
such as the build timestamp; it is not intended to reproduce identical file bytes.

## Deliberate exclusions

Git excludes `_archive` login probes, duplicate `rank-*/npbench.db` shards,
build/cache directories, compiled objects/libraries, SQLite journals/WAL files,
and unapproved campaign directories. The retained merged databases contain the
same measurement rows as their combined rank shards, ignoring row IDs.
Existing measurements are preserved without edits. No Git LFS is required.

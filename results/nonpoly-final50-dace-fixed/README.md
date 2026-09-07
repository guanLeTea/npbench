# Derived dataset: nonpoly-final50-dace-fixed

**Not a single Slurm campaign.** Assembled from two runs so the DaCe column is internally
consistent at one revision.

| rows | source job | NPBench | DaCe |
|---|---|---|---|
| NumPy, Pluto | 4523913 | `d86cc06b2` | `eb7b1352a` |
| DaCe auto_opt (all 31) | 4525319 | `d19133b44` | `b27aaed2f` |

Preset paper, REPEAT=50, 31 kernels, 4 ranks x 72 CPUs.

The DaCe column was re-measured in full rather than patched per kernel: `b27aaed2f` is a large
change over `eb7b1352a`, so mixing revisions within one column would not be comparable.
Neither source directory was modified.

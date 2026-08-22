/* Loop-nest transcription of NPBench's azimint_naive NumPy port
 * (npbench/benchmarks/azimint_naive/azimint_naive_numpy.py), statement for
 * statement. Not a PolyBench kernel.
 *
 * The port's `rmax = radius.max()` is a reduction, written first. Its per-bin
 * boolean mask (`r1 <= radius < r2`) and mean are data-dependent VALUES, not
 * subscripts, so they stay affine written as a ternary inside the accumulation
 * statement: `sum[i] += cond ? data[j] : 0.0`, `cnt[i] += cond ? 1.0 : 0.0`, then
 * `res[i] = sum[i] / cnt[i]` -- dividing by the count of matched entries, exactly
 * as `values_r12.mean()` does, not by N.
 *
 * `rmax`, `sum` and `cnt` are the port's own scalar/per-bin temporaries, passed
 * as scratch parameters rather than function locals (PolyBench/C's convention
 * for scratch). */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

void azimint_naive_fp64(int64_t N, int64_t NPT, const double data[restrict N], const double radius[restrict N],
                        double res[restrict NPT], double sum[restrict NPT], double cnt[restrict NPT],
                        double *restrict rmax) {

  int i, j;

#pragma scop
  rmax[0] = radius[0];
  for (i = 0; i < N; i++)
    rmax[0] = (radius[i] > rmax[0]) ? radius[i] : rmax[0];

  for (i = 0; i < NPT; i++) {
    sum[i] = SCALAR_VAL(0.0);
    cnt[i] = SCALAR_VAL(0.0);
    for (j = 0; j < N; j++) {
      sum[i] += ((rmax[0] * i / NPT <= radius[j]) && (radius[j] < rmax[0] * (i + 1) / NPT)) ? data[j] : SCALAR_VAL(0.0);
      cnt[i] += ((rmax[0] * i / NPT <= radius[j]) && (radius[j] < rmax[0] * (i + 1) / NPT)) ? SCALAR_VAL(1.0) : SCALAR_VAL(0.0);
    }
    res[i] = sum[i] / cnt[i];
  }
#pragma endscop
}

/* Not a PolyBench kernel: transcribed statement for statement from the NumPy port
 * (azimint_hist/azimint_hist_numpy.py: azimint_hist), the standard `np.histogram` plus
 * its weighted variant, then their ratio.
 *
 * `np.histogram(radius, npt)` uses `npt` equal-width bins spanning `[radius.min(),
 * radius.max()]`, ALL bins half-open `[lo, hi)` except the LAST, which is closed on the
 * right (`[lo, hi]`) -- transcribed below as the clamp `idx >= npt -> idx = npt - 1`,
 * which only the point(s) equal to the global max can ever hit. `histu` and `histw` are
 * scratch ARRAY parameters (the raw counts and the `data`-weighted sums); the port's
 * `histw / histu` division lands in `histw` in place, which is what the port returns.
 *
 * Attempted faithfully, with no O(npt * N) gather substituted for the histogram's own
 * scatter: each point's bin index `idx` is computed from the VALUE of `radius[n]`, a
 * quantity no scop parameter states, and `histu[idx]`/`histw[idx]` are then written
 * through that data-dependent index -- not an affine access, and not rewritten into one.
 */
#include <stdint.h>

void azimint_hist_fp64(int64_t N, int64_t npt, const double data[restrict N], const double radius[restrict N],
                       double histu[restrict npt], double histw[restrict npt]) {

  int n, k;
  double lo, hi, binw;
  int idx;

#pragma scop
  lo = radius[0];
  hi = radius[0];
  for (n = 1; n < N; n++) {
    lo = (radius[n] < lo) ? radius[n] : lo;
    hi = (radius[n] > hi) ? radius[n] : hi;
  }
  binw = (hi - lo) / (double)npt;

  for (k = 0; k < npt; k++) {
    histu[k] = 0.0;
    histw[k] = 0.0;
  }

  for (n = 0; n < N; n++) {
    idx = (int)((radius[n] - lo) / binw);
    idx = (idx >= npt) ? (int)(npt - 1) : idx;
    histu[idx] += 1.0;
    histw[idx] += data[n];
  }

  for (k = 0; k < npt; k++)
    histw[k] = histw[k] / histu[k];
#pragma endscop
}

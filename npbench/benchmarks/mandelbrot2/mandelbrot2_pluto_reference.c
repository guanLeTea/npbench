/* Not a PolyBench kernel: transcribed statement for statement from the NumPy port
 * (mandelbrot2/mandelbrot2_numpy.py: mandelbrot), the escape-time iteration with STREAM
 * COMPACTION -- the variant that drops diverged points and keeps iterating only the rest.
 *
 * Written out in full and handed to polycc on purpose, so the refusal on file is Pluto's own.
 * Two properties of the kernel are expected to stop it, and neither is rewritten away:
 *
 *  - `Z = Z[I]`, `Xi, Yi = Xi[I], Yi[I]`, `C = C[I]` SHRINK the working set by a data-dependent
 *    amount every iteration. `live` below is that length: a loop bound no polyhedron describes,
 *    since it is a count of how many points have not yet diverged. Iterating the full rectangle
 *    instead would be the OTHER benchmark (mandelbrot1), not this one.
 *  - `N_[Xi[I], Yi[I]] = i + 1` scatters through the surviving points' own coordinates, which
 *    are values carried in Xi/Yi, not functions of the iteration vector.
 *
 * The port returns `Z_.T` and `N_.T`, so the scatter writes the transposed arrays directly
 * rather than materialising a transpose the port does not compute.
 *
 * Measured, both refusals from clan itself:
 *  - as written: `non-affine expression at line 36` -- the port's own flattening `i*YN + j`
 *    multiplies a PARAMETER by an index, which the polyhedral model excludes before it ever
 *    reaches the compaction.
 *  - with the flattening removed (diagnostic copy, arrays pre-filled): `variable or array
 *    reference in an affine expression`, i.e. the `k < live` bound and the Xi/Yi scatter.
 *    Removing the first blocker does not reveal an acceptable kernel underneath.
 */
#include <stdint.h>
#include <complex.h>
#include <math.h>
#define DATA_TYPE double _Complex

void mandelbrot2_fp64(int64_t XN, int64_t YN, int64_t maxiter, double _Complex ZT[restrict YN][XN],
                      int64_t NT[restrict YN][XN], double _Complex Z[restrict XN * YN],
                      double _Complex C[restrict XN * YN], int64_t Xi[restrict XN * YN],
                      int64_t Yi[restrict XN * YN], double xmin, double xmax, double ymin, double ymax,
                      double horizon) {

  int i, j, k, n;
  int64_t live, kept;

#pragma scop
  /* Xi, Yi = mgrid[0:xn, 0:yn]; C = X[Xi] + Y[Yi]*1j; all flattened to xn*yn */
  for (i = 0; i < XN; i++)
    for (j = 0; j < YN; j++) {
      Xi[i * YN + j] = i;
      Yi[i * YN + j] = j;
      C[i * YN + j] = (xmin + (double)i * (xmax - xmin) / (double)(XN - 1)) +
                      (ymin + (double)j * (ymax - ymin) / (double)(YN - 1)) * I;
      Z[i * YN + j] = 0.0;
    }
  for (j = 0; j < YN; j++)
    for (i = 0; i < XN; i++) {
      ZT[j][i] = 0.0;
      NT[j][i] = 0;
    }

  live = XN * YN;
  for (n = 0; n < maxiter; n++) {
    /* Z = Z*Z + C, over the SURVIVING points only */
    for (k = 0; k < live; k++)
      Z[k] = Z[k] * Z[k] + C[k];

    /* diverged: record and drop; survivors are compacted to the front */
    kept = 0;
    for (k = 0; k < live; k++) {
      if (cabs(Z[k]) > horizon) {
        NT[Yi[k]][Xi[k]] = n + 1;
        ZT[Yi[k]][Xi[k]] = Z[k];
      } else {
        Z[kept] = Z[k];
        C[kept] = C[k];
        Xi[kept] = Xi[k];
        Yi[kept] = Yi[k];
        kept = kept + 1;
      }
    }
    live = kept;
  }
#pragma endscop
}

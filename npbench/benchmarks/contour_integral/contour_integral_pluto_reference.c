/* Not a PolyBench kernel: transcribed statement for statement from the NumPy port
 * (contour_integral/contour_integral_numpy.py), a contour sum of resolvents.
 *
 * Written out in full and handed to polycc on purpose, so the refusal on file is Pluto's own.
 * The loops the port DOES have (the Tz accumulation over slabs and the two contour sums) are
 * written out as loops. What stops the kernel is the middle step:
 *
 *  - `np.linalg.inv(Tz)` / `np.linalg.solve(Tz, Y)` is a DENSE FACTORIZATION, and the port
 *    has no loop nest for it -- it calls LAPACK. It appears below as the call it is. A scop
 *    is a loop nest over affine accesses; a call whose body is not in the region is not one.
 *    Substituting a hand-written elimination here would be a different computation (different
 *    pivoting, different rounding) than the one NPBench validates against, so it is not done.
 *  - `if abs(z) < 1.0: X = -X` and `if NR == NM` are value/parameter conditions on top of it.
 *  - P0, P1, Ham, int_pts, Y and Tz are complex128, and P0/P1 are the returned results, so the
 *    element type is `double _Complex` rather than split into re/im arrays.
 *
 * `np.power(z, slab_per_bc/2 - n)` is `cpow` on the same complex base and real exponent.
 */
#include <stdint.h>
#include <complex.h>
#define DATA_TYPE double _Complex

/* npbench-pluto-link: -llapacke */

/* LAPACK-backed dense solve, OUTSIDE the scop region, because that is where it is in the port
 * too: `np.linalg.inv` is LAPACK zgetrf + zgetri and `np.linalg.solve` is zgesv, and NumPy
 * calls them rather than running a loop nest. Linking the same routines keeps the computation
 * identical to the one NPBench validates; writing an elimination by hand here would pivot
 * differently and round differently. Pluto sees only the loops around this call. */
#include <lapacke.h>
#include <stdlib.h>
#include <string.h>

static void contour_dense_solve(int64_t NR, int64_t NM, double _Complex *Tz, const double _Complex *Y,
                                double _Complex *X) {
  lapack_int n = (lapack_int)NR;
  lapack_int m = (lapack_int)NM;
  lapack_int *ipiv = (lapack_int *)malloc((size_t)n * sizeof(lapack_int));

  if (NR == NM) {
    /* X = np.linalg.inv(Tz) */
    memcpy(X, Tz, (size_t)n * (size_t)n * sizeof(double _Complex));
    LAPACKE_zgetrf(LAPACK_ROW_MAJOR, n, n, (lapack_complex_double *)X, n, ipiv);
    LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, (lapack_complex_double *)X, n, ipiv);
  } else {
    /* X = np.linalg.solve(Tz, Y) -- zgesv overwrites its right-hand side with the solution */
    memcpy(X, Y, (size_t)n * (size_t)m * sizeof(double _Complex));
    LAPACKE_zgesv(LAPACK_ROW_MAJOR, n, m, (lapack_complex_double *)Tz, n, ipiv, (lapack_complex_double *)X, m);
  }
  free(ipiv);
}

void contour_integral_fp64(int64_t NR, int64_t NM, int64_t slab_per_bc, int64_t num_int_pts,
                           const double _Complex Ham[restrict slab_per_bc + 1][NR][NR],
                           const double _Complex int_pts[restrict num_int_pts],
                           const double _Complex Y[restrict NR][NM], double _Complex P0[restrict NR][NM],
                           double _Complex P1[restrict NR][NM], double _Complex Tz[restrict NR][NR],
                           double _Complex X[restrict NR][NM], double _Complex zz[restrict slab_per_bc + 1]) {

  int p, n, i, j;

#pragma scop
  for (i = 0; i < NR; i++)
    for (j = 0; j < NM; j++) {
      P0[i][j] = 0.0;
      P1[i][j] = 0.0;
    }

  for (p = 0; p < num_int_pts; p++) {
    /* Tz = sum_n z**(slab_per_bc/2 - n) * Ham[n].
     * `zz` is an ARRAY, one slot per slab, not a scalar reused across the slab loop: as a
     * scalar Pluto fuses the slab loop into a parallel tile and hoists the `zz = cpow(...)`
     * statement out of the accumulation that reads it, so every tile sees one shared, stale
     * value. Measured -- that is what the first version of this file did wrong. */
    for (i = 0; i < NR; i++)
      for (j = 0; j < NR; j++)
        Tz[i][j] = 0.0;
    for (n = 0; n < slab_per_bc + 1; n++)
      zz[n] = cpow(int_pts[p], (double)slab_per_bc / 2.0 - (double)n);
    for (n = 0; n < slab_per_bc + 1; n++)
      for (i = 0; i < NR; i++)
        for (j = 0; j < NR; j++)
          Tz[i][j] += zz[n] * Ham[n][i][j];

    /* X = inv(Tz) if NR == NM else solve(Tz, Y) -- the port's LAPACK call */
    contour_dense_solve(NR, NM, &Tz[0][0], &Y[0][0], &X[0][0]);

    /* if abs(z) < 1.0: X = -X ; P0 += X ; P1 += z * X */
    for (i = 0; i < NR; i++)
      for (j = 0; j < NM; j++) {
        X[i][j] = (cabs(int_pts[p]) < 1.0) ? -X[i][j] : X[i][j];
        P0[i][j] += X[i][j];
        P1[i][j] += int_pts[p] * X[i][j];
      }
  }
#pragma endscop
}

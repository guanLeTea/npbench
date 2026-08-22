/* PolyBench/C 4.2.1 original kernel_2mm (polybench.sourceforge.net), adapted to the
 * harness's runtime-sized VLA signature -- see npbench/infrastructure/pluto_framework.py.
 * `tmp` is PolyBench/C's own scratch parameter of kernel_2mm, kept as a caller-allocated
 * parameter here too. The NumPy port (`D[:] = alpha * A @ B @ C + beta * D`) multiplies
 * alpha into the FIRST product, exactly as kernel_2mm's `tmp[i][j] += alpha * A[i][k] * B[k][j]`
 * does. */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

void k2mm_fp64(int64_t NI, int64_t NJ, int64_t NK, int64_t NL, const double A[restrict NI][NK],
               const double B[restrict NK][NJ], const double C[restrict NJ][NL], double D[restrict NI][NL],
               double tmp[restrict NI][NJ], double alpha, double beta) {

  int i, j, k;

#pragma scop
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++) {
      tmp[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < NK; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
    }
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++) {
      D[i][j] *= beta;
      for (k = 0; k < NJ; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
    }
#pragma endscop
}

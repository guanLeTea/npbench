/* Manual verbatim transcription of the PolyBench/C 4.2.1 kernel_gesummv body
 * (polybench.sourceforge.net), adapted only in the function signature: the
 * harness's runtime-sized VLA parameters, with tmp promoted from a function
 * local to a caller-allocated scratch parameter (PolyBench/C passes it as one
 * too). */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

#define _PB_N N

void gesummv_fp64(int64_t N, const double A[restrict N][N], const double B[restrict N][N], const double *restrict x,
                  double *restrict y, double *restrict tmp, double alpha, double beta) {

  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    tmp[i] = SCALAR_VAL(0.0);
    y[i] = SCALAR_VAL(0.0);
    for (j = 0; j < _PB_N; j++) {
      tmp[i] = A[i][j] * x[j] + tmp[i];
      y[i] = B[i][j] * x[j] + y[i];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
#pragma endscop
}

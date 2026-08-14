/* Manual verbatim transcription of the PolyBench/C 4.2.1 kernel_gemver body
 * (polybench.sourceforge.net), adapted only in the function signature: the
 * harness's runtime-sized VLA parameters. */
#include <stdint.h>
#include <math.h>
#define DATA_TYPE double

#define _PB_N N

void gemver_fp64(int64_t N, double A[restrict N][N], const double *restrict u1, const double *restrict u2, const double *restrict v1, const double *restrict v2, double *restrict w, double *restrict x, const double *restrict y, const double *restrict z, double alpha, double beta) {

  int i, j;

#pragma scop

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < _PB_N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      w[i] = w[i] + alpha * A[i][j] * x[j];

#pragma endscop
}

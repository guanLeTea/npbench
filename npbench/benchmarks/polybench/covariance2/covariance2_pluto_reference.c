/* Manual verbatim transcription of the PolyBench/C 4.2.1 kernel_covariance body
 * (polybench.sourceforge.net), adapted only in the function signature: the
 * harness's runtime-sized VLA parameters. `mean` is a caller-allocated scratch
 * parameter here exactly as it is in PolyBench/C. */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

#define _PB_M M
#define _PB_N N

void covariance2_fp64(int64_t M, int64_t N, double data[restrict N][M], double cov[restrict M][M], double *restrict mean,
                  double float_n) {

  int i, j, k;

#pragma scop
  for (j = 0; j < _PB_M; j++) {
    mean[j] = SCALAR_VAL(0.0);
    for (i = 0; i < _PB_N; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < _PB_M; i++)
    for (j = i; j < _PB_M; j++) {
      cov[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < _PB_N; k++)
        cov[i][j] += data[k][i] * data[k][j];
      cov[i][j] /= (float_n - SCALAR_VAL(1.0));
      cov[j][i] = cov[i][j];
    }
#pragma endscop
}

/* Manual verbatim transcription of the PolyBench/C 4.2.1 kernel_cholesky body
 * (polybench.sourceforge.net), adapted only in the function signature: the
 * harness's runtime-sized VLA parameters. */
#include <stdint.h>
#include <math.h>
#define SQRT_FUN(x) sqrt(x)
#define DATA_TYPE double

#define _PB_N N

void cholesky_fp64(int64_t N, double A[restrict N][N]) {

  int i, j, k;

#pragma scop
  for (i = 0; i < _PB_N; i++) {
    // j<i
    for (j = 0; j < i; j++) {
      for (k = 0; k < j; k++) {
        A[i][j] -= A[i][k] * A[j][k];
      }
      A[i][j] /= A[j][j];
    }
    // i==j case
    for (k = 0; k < i; k++) {
      A[i][i] -= A[i][k] * A[i][k];
    }
    A[i][i] = SQRT_FUN(A[i][i]);
  }
#pragma endscop
}

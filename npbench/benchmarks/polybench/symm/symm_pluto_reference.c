/* Manual verbatim transcription of the PolyBench/C 4.2.1 kernel_symm body
 * (polybench.sourceforge.net), adapted only in the function signature: the
 * harness's runtime-sized VLA parameters. `temp2` stays PolyBench/C's own
 * scalar local (`DATA_TYPE temp2;`), not a scratch array, so it is declared
 * above the scop rather than passed in. */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

#define _PB_M M
#define _PB_N N

void symm_fp64(int64_t M, int64_t N, double C[restrict M][N], const double A[restrict M][M],
              const double B[restrict M][N], double alpha, double beta) {

  int i, j, k;
  DATA_TYPE temp2;

#pragma scop
  for (i = 0; i < _PB_M; i++)
    for (j = 0; j < _PB_N; j++) {
      temp2 = SCALAR_VAL(0.0);
      for (k = 0; k < i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
#pragma endscop
}

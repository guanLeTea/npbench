/* Loop-nest transcription of NPBench's numerically-stable softmax over the last
 * axis of a rank-4 float32 tensor. Not a PolyBench kernel: the scop is written
 * from the NumPy port statement for statement -- row max, exp of the shifted
 * row, row sum, divide -- in the same order and the same float32 precision, so
 * only the association inside each reduction can differ.
 *
 * tmp_max and tmp_sum are the port's own keepdims temporaries, passed as
 * caller-allocated scratch parameters (PolyBench/C's convention for scratch,
 * and what keeps them out of Pluto's scop-local dead-code elimination). */
#include <stdint.h>
#include <math.h>
#define DATA_TYPE float

void softmax_fp64(int64_t N, int64_t H, int64_t SM, const float x[restrict N][H][SM][SM],
                  float out[restrict N][H][SM][SM], float tmp_max[restrict N][H][SM],
                  float tmp_sum[restrict N][H][SM]) {

  int i, j, k, l;

#pragma scop
  for (i = 0; i < N; i++)
    for (j = 0; j < H; j++)
      for (k = 0; k < SM; k++) {
        tmp_max[i][j][k] = x[i][j][k][0];
        for (l = 0; l < SM; l++)
          tmp_max[i][j][k] = (x[i][j][k][l] > tmp_max[i][j][k]) ? x[i][j][k][l] : tmp_max[i][j][k];
        tmp_sum[i][j][k] = 0.0f;
        for (l = 0; l < SM; l++) {
          out[i][j][k][l] = expf(x[i][j][k][l] - tmp_max[i][j][k]);
          tmp_sum[i][j][k] += out[i][j][k][l];
        }
        for (l = 0; l < SM; l++)
          out[i][j][k][l] /= tmp_sum[i][j][k];
      }
#pragma endscop
}

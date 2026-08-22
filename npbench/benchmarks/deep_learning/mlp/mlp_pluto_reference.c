/* Loop-nest transcription of NPBench's mlp NumPy port
 * (npbench/benchmarks/deep_learning/mlp/mlp_numpy.py), statement for statement
 * and in the port's own float32 precision: `x = relu(input @ w1 + b1)`, then
 * `x = relu(x @ w2 + b2)`, then `x = softmax(x @ w3 + b3)`. Not a PolyBench
 * kernel. The softmax stage (row max, exp of the shifted row, row sum, divide)
 * follows the same statement order as the already-ported
 * `deep_learning/softmax/softmax_pluto_reference.c`, fused directly onto the
 * third layer's matmul+bias instead of reading a separate input tensor.
 *
 * Each matmul accumulates the dot product first and adds the bias afterward, in
 * that order, matching `x @ w + b`'s own operation order (matmul, then a
 * broadcast add) rather than seeding the accumulator with the bias.
 *
 * `x1`, `x2` (the two hidden activations) and `tmp_max`/`tmp_sum` (softmax's own
 * keepdims temporaries) are all scratch parameters, never function locals. */
#include <stdint.h>
#include <math.h>
#define DATA_TYPE float

void mlp_fp64(int64_t N, int64_t C_in, int64_t S0, int64_t S1, int64_t S2, const float input[restrict N][C_in],
             const float w1[restrict C_in][S0], const float b1[restrict S0], const float w2[restrict S0][S1],
             const float b2[restrict S1], const float w3[restrict S1][S2], const float b3[restrict S2],
             float out[restrict N][S2], float x1[restrict N][S0], float x2[restrict N][S1],
             float tmp_max[restrict N], float tmp_sum[restrict N]) {

  int i, j, k;

#pragma scop
  /* x1 = relu(input @ w1 + b1) */
  for (i = 0; i < N; i++)
    for (j = 0; j < S0; j++) {
      x1[i][j] = 0.0f;
      for (k = 0; k < C_in; k++)
        x1[i][j] += input[i][k] * w1[k][j];
      x1[i][j] += b1[j];
      x1[i][j] = (x1[i][j] > 0.0f) ? x1[i][j] : 0.0f;
    }

  /* x2 = relu(x1 @ w2 + b2) */
  for (i = 0; i < N; i++)
    for (j = 0; j < S1; j++) {
      x2[i][j] = 0.0f;
      for (k = 0; k < S0; k++)
        x2[i][j] += x1[i][k] * w2[k][j];
      x2[i][j] += b2[j];
      x2[i][j] = (x2[i][j] > 0.0f) ? x2[i][j] : 0.0f;
    }

  /* out = softmax(x2 @ w3 + b3) */
  for (i = 0; i < N; i++) {
    for (j = 0; j < S2; j++) {
      out[i][j] = 0.0f;
      for (k = 0; k < S1; k++)
        out[i][j] += x2[i][k] * w3[k][j];
      out[i][j] += b3[j];
    }
    tmp_max[i] = out[i][0];
    for (j = 0; j < S2; j++)
      tmp_max[i] = (out[i][j] > tmp_max[i]) ? out[i][j] : tmp_max[i];
    tmp_sum[i] = 0.0f;
    for (j = 0; j < S2; j++) {
      out[i][j] = expf(out[i][j] - tmp_max[i]);
      tmp_sum[i] += out[i][j];
    }
    for (j = 0; j < S2; j++)
      out[i][j] /= tmp_sum[i];
  }
#pragma endscop
}

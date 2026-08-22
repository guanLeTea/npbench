/* Loop-nest transcription of NPBench's conv2d NumPy port
 * (npbench/benchmarks/deep_learning/conv2d_bias/conv2d_numpy.py), statement for statement and
 * in its own float32 precision: the benchmarked function is `conv2d_bias`, a stride-1 NHWC
 * convolution (`conv2d`) followed by a broadcast bias add, computed in that order -- the
 * accumulator is filled by the K*K*C_in reduction first and the bias is added afterward, the
 * same order `conv2d(input, weights) + bias` evaluates in and the same order already used for
 * mlp's and gesummv's own matmul-then-bias statements. Not a PolyBench kernel. The benchmark
 * key is `conv2d_bias`, but `module_name` in bench_info is `conv2d`, so this file and the
 * exported symbol are named `conv2d`.
 *
 * N, H, W, C_in come from input's own shape; K and C_out come from weights' shape (K appears
 * twice there, same value both times); H_out = H - K + 1 and W_out = W - K + 1 are DERIVED
 * extents with no array of their own, so they are declared as their own scop symbols and sized
 * by the adapter rather than written as an in-place expression. */
#include <stdint.h>

void conv2d_fp64(int64_t N, int64_t H, int64_t W, int64_t C_in, int64_t K, int64_t C_out, int64_t H_out,
                 int64_t W_out, const float input[restrict N][H][W][C_in],
                 const float weights[restrict K][K][C_in][C_out], const float bias[restrict C_out],
                 float output[restrict N][H_out][W_out][C_out]) {

  int n, i, j, co, kh, kw, ci;

#pragma scop
  for (n = 0; n < N; n++)
    for (i = 0; i < H_out; i++)
      for (j = 0; j < W_out; j++)
        for (co = 0; co < C_out; co++) {
          output[n][i][j][co] = 0.0f;
          for (kh = 0; kh < K; kh++)
            for (kw = 0; kw < K; kw++)
              for (ci = 0; ci < C_in; ci++)
                output[n][i][j][co] += input[n][i + kh][j + kw][ci] * weights[kh][kw][ci][co];
          output[n][i][j][co] += bias[co];
        }
#pragma endscop
}

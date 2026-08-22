/* Not a PolyBench kernel: transcribed statement for statement from the NumPy port
 * (deep_learning/lenet/lenet_numpy.py: relu, conv2d, maxpool2d, lenet5), float32
 * throughout, same statement order as the port.
 *
 * Adaptations, all read off the port rather than chosen to make numbers match:
 *  - Every intermediate the port allocates (both conv2d outputs, both maxpool2d outputs,
 *    and the three dense-layer activations) is a caller-allocated scratch PARAMETER, never
 *    a function local, matching this column's other deep_learning ports.
 *  - `H1`, `W1`, `PH1`, `PW1`, `H2`, `W2`, `PH2`, `PW2` are the post-conv1/post-pool1/
 *    post-conv2/post-pool2 spatial sizes. No real array's shape states them (the port
 *    computes them once, inline, from `H` and `W`), so they arrive as extra `int64_t`
 *    parameters; the adapter derives each with the exact arithmetic
 *    `deep_learning/lenet/lenet.py`'s `initialize()` uses to build `C_before_fc1`.
 *  - conv2d is `output[:, i, j, :] = sum(input[:, i:i+K, j:j+K, :, None] * weights, axis=(1,2,3))`
 *    for every output position, K = 5 both times: accumulate the raw sum into the scratch
 *    output (zero-initialized by the adapter), then a separate statement adds the bias and
 *    applies relu, matching the port's `relu(conv2d(...) + bias)` as two operations in that
 *    order.
 *  - maxpool2d's `np.max(x[:, 2i:2i+2, 2j:2j+2, :], axis=(1,2))` over each 2x2 window is
 *    written as nested `fmaxf` of the four window cells, one statement, no locals.
 *  - `x = np.reshape(x, (N, C_before_fc1))` before fc1 is pure index re-linearization: NumPy's
 *    reshape is row-major, so flattening (PH2, PW2, 16) into `C_before_fc1` gives exactly the
 *    row index `fc1w` is a matmul operand `x_flat[n, idx] @ fc1w[idx, j]` over. Since `fc1w`'s
 *    real buffer is C-contiguous with that same row-major layout, this scop reinterprets it as
 *    a `[PH2][PW2][16][120]` VLA over the SAME memory (no separate flattened copy of the pooled
 *    activation, no hand-computed linear index) and the fc1 statement below sums over the three
 *    un-flattened axes directly -- algebraically identical to reshape-then-matmul.
 *  - fc1/fc2/fc3 each accumulate the dot product first and add the bias afterward, in that
 *    order, matching `x @ w + b`'s own operation order, as this column's `mlp` reference does.
 */
#include <stdint.h>
#include <math.h>
#define DATA_TYPE float

void lenet_fp64(int64_t N, int64_t H, int64_t W, int64_t H1, int64_t W1, int64_t PH1, int64_t PW1,
                int64_t H2, int64_t W2, int64_t PH2, int64_t PW2, const float input[restrict N][H][W][1],
                const float conv1[restrict 5][5][1][6], const float conv1bias[restrict 6],
                const float conv2[restrict 5][5][6][16], const float conv2bias[restrict 16],
                const float fc1w[restrict PH2][PW2][16][120], const float fc1b[restrict 120],
                const float fc2w[restrict 120][84], const float fc2b[restrict 84],
                const float fc3w[restrict 84][10], const float fc3b[restrict 10],
                float conv1out[restrict N][H1][W1][6], float pool1out[restrict N][PH1][PW1][6],
                float conv2out[restrict N][H2][W2][16], float pool2out[restrict N][PH2][PW2][16],
                float x1[restrict N][120], float x2[restrict N][84], float out[restrict N][10]) {

  int n, i, j, cout, kh, kw, cin, c, h, w, k;

#pragma scop
  /* conv1out = conv2d(input, conv1), K = 5, Cin = 1, Cout = 6 */
  for (n = 0; n < N; n++)
    for (i = 0; i < H1; i++)
      for (j = 0; j < W1; j++)
        for (cout = 0; cout < 6; cout++)
          for (kh = 0; kh < 5; kh++)
            for (kw = 0; kw < 5; kw++)
              for (cin = 0; cin < 1; cin++)
                conv1out[n][i][j][cout] += input[n][i + kh][j + kw][cin] * conv1[kh][kw][cin][cout];

  /* conv1out = relu(conv1out + conv1bias) */
  for (n = 0; n < N; n++)
    for (i = 0; i < H1; i++)
      for (j = 0; j < W1; j++)
        for (cout = 0; cout < 6; cout++)
          conv1out[n][i][j][cout] = fmaxf(conv1out[n][i][j][cout] + conv1bias[cout], 0.0f);

  /* pool1out = maxpool2d(conv1out) */
  for (n = 0; n < N; n++)
    for (i = 0; i < PH1; i++)
      for (j = 0; j < PW1; j++)
        for (c = 0; c < 6; c++)
          pool1out[n][i][j][c] =
              fmaxf(fmaxf(conv1out[n][2 * i][2 * j][c], conv1out[n][2 * i][2 * j + 1][c]),
                    fmaxf(conv1out[n][2 * i + 1][2 * j][c], conv1out[n][2 * i + 1][2 * j + 1][c]));

  /* conv2out = conv2d(pool1out, conv2), K = 5, Cin = 6, Cout = 16 */
  for (n = 0; n < N; n++)
    for (i = 0; i < H2; i++)
      for (j = 0; j < W2; j++)
        for (cout = 0; cout < 16; cout++)
          for (kh = 0; kh < 5; kh++)
            for (kw = 0; kw < 5; kw++)
              for (cin = 0; cin < 6; cin++)
                conv2out[n][i][j][cout] += pool1out[n][i + kh][j + kw][cin] * conv2[kh][kw][cin][cout];

  /* conv2out = relu(conv2out + conv2bias) */
  for (n = 0; n < N; n++)
    for (i = 0; i < H2; i++)
      for (j = 0; j < W2; j++)
        for (cout = 0; cout < 16; cout++)
          conv2out[n][i][j][cout] = fmaxf(conv2out[n][i][j][cout] + conv2bias[cout], 0.0f);

  /* pool2out = maxpool2d(conv2out) */
  for (n = 0; n < N; n++)
    for (i = 0; i < PH2; i++)
      for (j = 0; j < PW2; j++)
        for (c = 0; c < 16; c++)
          pool2out[n][i][j][c] =
              fmaxf(fmaxf(conv2out[n][2 * i][2 * j][c], conv2out[n][2 * i][2 * j + 1][c]),
                    fmaxf(conv2out[n][2 * i + 1][2 * j][c], conv2out[n][2 * i + 1][2 * j + 1][c]));

  /* x1 = relu(reshape(pool2out) @ fc1w + fc1b), fc1w read as [PH2][PW2][16][120] */
  for (n = 0; n < N; n++)
    for (j = 0; j < 120; j++) {
      x1[n][j] = 0.0f;
      for (h = 0; h < PH2; h++)
        for (w = 0; w < PW2; w++)
          for (c = 0; c < 16; c++)
            x1[n][j] += pool2out[n][h][w][c] * fc1w[h][w][c][j];
      x1[n][j] += fc1b[j];
      x1[n][j] = fmaxf(x1[n][j], 0.0f);
    }

  /* x2 = relu(x1 @ fc2w + fc2b) */
  for (n = 0; n < N; n++)
    for (j = 0; j < 84; j++) {
      x2[n][j] = 0.0f;
      for (k = 0; k < 120; k++)
        x2[n][j] += x1[n][k] * fc2w[k][j];
      x2[n][j] += fc2b[j];
      x2[n][j] = fmaxf(x2[n][j], 0.0f);
    }

  /* out = x2 @ fc3w + fc3b */
  for (n = 0; n < N; n++)
    for (j = 0; j < 10; j++) {
      out[n][j] = 0.0f;
      for (k = 0; k < 84; k++)
        out[n][j] += x2[n][k] * fc3w[k][j];
      out[n][j] += fc3b[j];
    }
#pragma endscop
}

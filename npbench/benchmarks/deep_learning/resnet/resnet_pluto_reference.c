/* Not a PolyBench kernel: transcribed statement for statement from the NumPy port
 * (deep_learning/resnet/resnet_numpy.py: relu, conv2d, batchnorm2d, resnet_basicblock),
 * same order and same per-stage precision.
 *
 * Adaptations, all read off the port rather than chosen to make numbers match:
 *  - Every intermediate the port allocates (`padded`, both conv2d outputs, and every
 *    batchnorm mean/std reduction) is a caller-allocated scratch PARAMETER, never a
 *    function local -- a data-typed local inside the scop is what the framework's
 *    dead-code screen refuses.
 *  - `padded` matches the port's own `np.zeros((...))` with NO dtype argument: it stays
 *    double (float64) even though every conv2d output is float32, and its one-cell
 *    border is left at the caller's zero fill, exactly as the port never writes it
 *    either. The first conv2d's inputs (`input`, `conv1`) are both float32, so summing
 *    directly into `padded` (double) is at least as precise as the port's float32 sum,
 *    never less.
 *  - The second conv2d genuinely straddles a precision boundary: its input (`padded`,
 *    after batchnorm1+relu1) is double and its weights (`conv2`) are float32, so NumPy's
 *    elementwise product and reduction run in double and the result is rounded to
 *    float32 exactly ONCE, when it lands in conv2d's own `np.zeros(..., dtype=np.float32)`
 *    output array. `acc2` reproduces that: a double accumulator, cast down into `buf2`
 *    (float) in one separate statement, rather than rounding after every partial sum.
 *    The third conv2d's operands are both float32 (`buf2` after batchnorm2+relu2, and
 *    `conv3`), so it accumulates directly into `out` at float precision -- matching
 *    NumPy's own per-term precision there, same as this column's softmax reference does
 *    for its running sum.
 *  - `batchnorm2d`'s `(x - mean) / np.sqrt(std + eps)` -- NOT `sqrt(var + eps)` -- and
 *    `np.std`'s population definition (`sqrt(mean((x-mean)^2))`, divisor N) are
 *    transcribed exactly. `eps=1e-5` is written `1e-5` where the buffer is double and
 *    `1e-5f` where it is float: checked against the installed NumPy (2.5.2), a Python
 *    float scalar added to a float32 array stays float32 there (NEP 50 weak-scalar
 *    promotion), it is not assumed.
 *  - Each of the three batchnorms gets its own mean/std scratch pair because the three
 *    stages run at different shapes and precisions ((H+2,W+2,C2) double; (H,W,C2)
 *    float; (H,W,C1) float).
 *  - `out` triples as the raw third conv2d accumulator, the batchnorm3 target, and the
 *    port's returned `relu(x + input)` -- all three have the same (N,H,W,C1) shape as
 *    the residual add, so no separate buffer is needed for any of them.
 */
#include <stdint.h>
#include <math.h>

void resnet_fp64(int64_t N, int64_t H, int64_t W, int64_t C1, int64_t C2,
                 const float input[restrict N][H][W][C1],
                 const float conv1[restrict 1][1][C1][C2],
                 const float conv2[restrict 3][3][C2][C2],
                 const float conv3[restrict 1][1][C2][C1],
                 double padded[restrict N][H + 2][W + 2][C2],
                 double mean1[restrict H + 2][W + 2][C2],
                 double std1[restrict H + 2][W + 2][C2],
                 float buf2[restrict N][H][W][C2],
                 double acc2[restrict N][H][W][C2],
                 float mean2[restrict H][W][C2],
                 float std2[restrict H][W][C2],
                 float out[restrict N][H][W][C1],
                 float mean3[restrict H][W][C1],
                 float std3[restrict H][W][C1]) {

  int n, h, w, cout, cin, kh, kw;

#pragma scop
  /* padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1); K = 1, border stays at its zero fill */
  for (n = 0; n < N; n++)
    for (h = 0; h < H; h++)
      for (w = 0; w < W; w++)
        for (cout = 0; cout < C2; cout++)
          for (cin = 0; cin < C1; cin++)
            padded[n][h + 1][w + 1][cout] += input[n][h][w][cin] * conv1[0][0][cin][cout];

  /* x = batchnorm2d(padded): mean over axis 0, keepdims */
  for (h = 0; h < H + 2; h++)
    for (w = 0; w < W + 2; w++)
      for (cout = 0; cout < C2; cout++)
        for (n = 0; n < N; n++)
          mean1[h][w][cout] += padded[n][h][w][cout];
  for (h = 0; h < H + 2; h++)
    for (w = 0; w < W + 2; w++)
      for (cout = 0; cout < C2; cout++)
        mean1[h][w][cout] /= (double)N;

  /* np.std: population std, divisor N */
  for (h = 0; h < H + 2; h++)
    for (w = 0; w < W + 2; w++)
      for (cout = 0; cout < C2; cout++)
        for (n = 0; n < N; n++)
          std1[h][w][cout] +=
              (padded[n][h][w][cout] - mean1[h][w][cout]) * (padded[n][h][w][cout] - mean1[h][w][cout]);
  for (h = 0; h < H + 2; h++)
    for (w = 0; w < W + 2; w++)
      for (cout = 0; cout < C2; cout++)
        std1[h][w][cout] = sqrt(std1[h][w][cout] / (double)N);

  /* x = relu(x): (x - mean) / sqrt(std + eps), then max with 0, in place on padded */
  for (n = 0; n < N; n++)
    for (h = 0; h < H + 2; h++)
      for (w = 0; w < W + 2; w++)
        for (cout = 0; cout < C2; cout++)
          padded[n][h][w][cout] =
              fmax((padded[n][h][w][cout] - mean1[h][w][cout]) / sqrt(std1[h][w][cout] + 1e-5), 0.0);

  /* x = conv2d(x, conv2): K = 3, double accumulate then round once into buf2 */
  for (n = 0; n < N; n++)
    for (h = 0; h < H; h++)
      for (w = 0; w < W; w++)
        for (cout = 0; cout < C2; cout++)
          for (kh = 0; kh < 3; kh++)
            for (kw = 0; kw < 3; kw++)
              for (cin = 0; cin < C2; cin++)
                acc2[n][h][w][cout] += padded[n][h + kh][w + kw][cin] * conv2[kh][kw][cin][cout];
  for (n = 0; n < N; n++)
    for (h = 0; h < H; h++)
      for (w = 0; w < W; w++)
        for (cout = 0; cout < C2; cout++)
          buf2[n][h][w][cout] = (float)acc2[n][h][w][cout];

  /* x = batchnorm2d(x) at float32 */
  for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
      for (cout = 0; cout < C2; cout++)
        for (n = 0; n < N; n++)
          mean2[h][w][cout] += buf2[n][h][w][cout];
  for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
      for (cout = 0; cout < C2; cout++)
        mean2[h][w][cout] /= (float)N;

  for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
      for (cout = 0; cout < C2; cout++)
        for (n = 0; n < N; n++)
          std2[h][w][cout] += (buf2[n][h][w][cout] - mean2[h][w][cout]) * (buf2[n][h][w][cout] - mean2[h][w][cout]);
  for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
      for (cout = 0; cout < C2; cout++)
        std2[h][w][cout] = sqrtf(std2[h][w][cout] / (float)N);

  /* x = relu(x), in place on buf2 */
  for (n = 0; n < N; n++)
    for (h = 0; h < H; h++)
      for (w = 0; w < W; w++)
        for (cout = 0; cout < C2; cout++)
          buf2[n][h][w][cout] =
              fmaxf((buf2[n][h][w][cout] - mean2[h][w][cout]) / sqrtf(std2[h][w][cout] + 1e-5f), 0.0f);

  /* x = conv2d(x, conv3): K = 1, both float32, accumulate directly in out */
  for (n = 0; n < N; n++)
    for (h = 0; h < H; h++)
      for (w = 0; w < W; w++)
        for (cout = 0; cout < C1; cout++)
          for (cin = 0; cin < C2; cin++)
            out[n][h][w][cout] += buf2[n][h][w][cin] * conv3[0][0][cin][cout];

  /* x = batchnorm2d(x) at float32, no relu here -- relu comes after the residual add */
  for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
      for (cout = 0; cout < C1; cout++)
        for (n = 0; n < N; n++)
          mean3[h][w][cout] += out[n][h][w][cout];
  for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
      for (cout = 0; cout < C1; cout++)
        mean3[h][w][cout] /= (float)N;

  for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
      for (cout = 0; cout < C1; cout++)
        for (n = 0; n < N; n++)
          std3[h][w][cout] += (out[n][h][w][cout] - mean3[h][w][cout]) * (out[n][h][w][cout] - mean3[h][w][cout]);
  for (h = 0; h < H; h++)
    for (w = 0; w < W; w++)
      for (cout = 0; cout < C1; cout++)
        std3[h][w][cout] = sqrtf(std3[h][w][cout] / (float)N);

  for (n = 0; n < N; n++)
    for (h = 0; h < H; h++)
      for (w = 0; w < W; w++)
        for (cout = 0; cout < C1; cout++)
          out[n][h][w][cout] = (out[n][h][w][cout] - mean3[h][w][cout]) / sqrtf(std3[h][w][cout] + 1e-5f);

  /* return relu(x + input), in place on out */
  for (n = 0; n < N; n++)
    for (h = 0; h < H; h++)
      for (w = 0; w < W; w++)
        for (cout = 0; cout < C1; cout++)
          out[n][h][w][cout] = fmaxf(out[n][h][w][cout] + input[n][h][w][cout], 0.0f);
#pragma endscop
}

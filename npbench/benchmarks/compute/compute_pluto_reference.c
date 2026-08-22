/* Loop-nest transcription of the NumPy port `np.clip(array_1, 2, 10) * a + array_2 * b + c`.
 * Not a PolyBench kernel: there is no PolyBench original for this microbench, so the scop is
 * written from the NumPy port statement for statement. `clip` is a per-element ternary rather
 * than a guarding `if`, which keeps the loop domain affine. array_1/array_2/a/b/c are int64
 * in the initializer (`.astype(np.int64)`, `np.int64(...)`), so every element type here is
 * int64_t to match exactly. */
#include <stdint.h>

void compute_fp64(int64_t M, int64_t N, const int64_t array_1[restrict M][N], const int64_t array_2[restrict M][N],
                  int64_t out[restrict M][N], int64_t a, int64_t b, int64_t c) {

  int i, j;

#pragma scop
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      out[i][j] = (array_1[i][j] < 2 ? 2 : (array_1[i][j] > 10 ? 10 : array_1[i][j])) * a + array_2[i][j] * b + c;
#pragma endscop
}

/* Not a PolyBench kernel: transcribed statement for statement from the NumPy port
 * (spmv/spmv_numpy.py: spmv), the standard CSR sparse matrix-vector product.
 *
 * `A_row` and `A_col` are declared `uint32_t` because that is what `spmv/spmv.py`'s
 * `initialize()` actually produces (`np.uint32(matrix.indptr)`, `np.uint32(matrix.indices)`)
 * -- not cast to a wider or signed type to make declaring this easier. `y` is a scratch
 * output PARAMETER, since the port returns it and `output_args` is empty.
 *
 * Attempted faithfully, with no densification: `A_row[i]`/`A_row[i + 1]` are the inner
 * loop's bounds, so the trip count of that loop is a VALUE read out of `A_row` at run time,
 * not an affine function of `i` and the scop's parameters; `A_col[j]` is then used to
 * subscript `x`, a second data-dependent access on top of the first. Neither is rewritten
 * to something affine -- that would be a different algorithm, not this one, transcribed.
 */
#include <stdint.h>

void spmv_fp64(int64_t Mp1, int64_t NNZ, int64_t XN, const uint32_t A_row[restrict Mp1],
               const uint32_t A_col[restrict NNZ], const double A_val[restrict NNZ], const double x[restrict XN],
               double y[restrict Mp1 - 1]) {

  int i, j;

#pragma scop
  for (i = 0; i < Mp1 - 1; i++) {
    y[i] = 0.0;
    for (j = A_row[i]; j < A_row[i + 1]; j++)
      y[i] += A_val[j] * x[A_col[j]];
  }
#pragma endscop
}

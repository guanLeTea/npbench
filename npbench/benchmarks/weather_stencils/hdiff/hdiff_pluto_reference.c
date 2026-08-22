/* Loop-nest transcription of NPBench's hdiff NumPy port
 * (npbench/benchmarks/weather_stencils/hdiff/hdiff_numpy.py), statement for statement and in
 * its own float64 precision: lap_field over a one-cell-wider halo, then flx_field and fly_field
 * through np.where written as a ternary INSIDE the statement (kept out of an `if` so the domain
 * stays affine), then out_field. Not a PolyBench kernel.
 *
 * I, J, K come from out_field's own shape; in_field is (I+4, J+4, K) -- only the trailing two
 * extents of a VLA parameter matter for its addressing, the leading one is decorative, so
 * `I + 4` is written honestly rather than reused as a plain `I`. out_field is both an input and
 * the sole output (the port overwrites it in place after only reading its shape), so it stays
 * one non-const parameter rather than a separate scratch buffer. lap_field, flx_field and
 * fly_field are the port's own temporaries, shaped exactly as its slice arithmetic implies
 * ((I+2, J+2, K), (I+1, J, K) and (I, J+1, K) respectively) and passed as caller-allocated
 * scratch parameters rather than function locals. */
#include <stdint.h>

void hdiff_fp64(int64_t I, int64_t J, int64_t K, const double in_field[restrict I + 4][J + 4][K],
                double out_field[restrict I][J][K], const double coeff[restrict I][J][K],
                double lap_field[restrict I + 2][J + 2][K], double flx_field[restrict I + 1][J][K],
                double fly_field[restrict I][J + 1][K]) {

  int i, j, k;

#pragma scop
  for (i = 0; i < I + 2; i++)
    for (j = 0; j < J + 2; j++)
      for (k = 0; k < K; k++)
        lap_field[i][j][k] = 4.0 * in_field[i + 1][j + 1][k] -
                             (in_field[i + 2][j + 1][k] + in_field[i][j + 1][k] +
                              in_field[i + 1][j + 2][k] + in_field[i + 1][j][k]);

  for (i = 0; i < I + 1; i++)
    for (j = 0; j < J; j++)
      for (k = 0; k < K; k++)
        flx_field[i][j][k] = ((lap_field[i + 1][j + 1][k] - lap_field[i][j + 1][k]) *
                              (in_field[i + 2][j + 2][k] - in_field[i + 1][j + 2][k])) > 0
                                 ? 0.0
                                 : (lap_field[i + 1][j + 1][k] - lap_field[i][j + 1][k]);

  for (i = 0; i < I; i++)
    for (j = 0; j < J + 1; j++)
      for (k = 0; k < K; k++)
        fly_field[i][j][k] = ((lap_field[i + 1][j + 1][k] - lap_field[i + 1][j][k]) *
                              (in_field[i + 2][j + 2][k] - in_field[i + 2][j + 1][k])) > 0
                                 ? 0.0
                                 : (lap_field[i + 1][j + 1][k] - lap_field[i + 1][j][k]);

  for (i = 0; i < I; i++)
    for (j = 0; j < J; j++)
      for (k = 0; k < K; k++)
        out_field[i][j][k] = in_field[i + 2][j + 2][k] -
                             coeff[i][j][k] * (flx_field[i + 1][j][k] - flx_field[i][j][k] +
                                              fly_field[i][j + 1][k] - fly_field[i][j][k]);
#pragma endscop
}

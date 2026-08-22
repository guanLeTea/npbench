/* Not a PolyBench kernel: transcribed statement for statement from the NumPy port
 * (stockham_fft/stockham_fft_numpy.py: stockham_fft), the Stockham radix-R FFT.
 *
 * Written out in full and handed to polycc on purpose, so the refusal on file is Pluto's
 * own and not this column declining to try. Two properties of the kernel are expected to
 * stop it, and neither is rewritten away:
 *
 *  - The stage extents are POWERS OF THE LOOP INDEX: stage `i` views `y` as
 *    (R**i, R, R**(K-i-1)), so the inner trip counts are `Li = R**i` and `Mi = R**(K-i-1)`.
 *    Those are multiplicative in `i`, not affine in it, so `Li`/`Mi` have to be carried in
 *    variables the loop updates -- and a variable trip count is not an affine bound.
 *  - Every array is complex128 in the port and `y` is RETURNED complex, so the element type
 *    is `double _Complex` here rather than split into re/im arrays, which would change what
 *    this column hands back. That part is NOT a blocker: this column marshals complex128
 *    (see `_ELEM_TYPES`), and `contour_integral` and `mandelbrot1` both validate complex
 *    results through it.
 *
 * Measured: `[Clan] Error: variable or array reference in an affine expression at line 48`,
 * which is `for (p = 0; p < Li; p++)` -- the stage extent carried in a variable.
 *
 * The twiddle factor `exp(-2*pi*i*r*p / R**(i+1))` is written with `cexp` on an imaginary
 * argument, exactly as the port writes `np.exp(-2.0j * np.pi * ...)`.
 */
#include <stdint.h>
#include <complex.h>
#include <math.h>
#define DATA_TYPE double _Complex

void stockham_fft_fp64(int64_t N, int64_t R, int64_t K, const double _Complex x[restrict N],
                       double _Complex y[restrict N], double _Complex dft_mat[restrict R][R],
                       double _Complex tmp_twid[restrict N]) {

  int i, n, a, b, c, r, p, m;
  int64_t Li, Mi, RK1;
  double _Complex acc;

#pragma scop
  /* dft_mat = np.exp(-2.0j * np.pi * i_coord * j_coord / R) */
  for (a = 0; a < R; a++)
    for (b = 0; b < R; b++)
      dft_mat[a][b] = cexp(-2.0 * M_PI * I * (double)(a * b) / (double)R);

  /* y[:] = x[:] */
  for (n = 0; n < N; n++)
    y[n] = x[n];

  Li = 1;
  Mi = N / R;
  RK1 = N / R;
  for (i = 0; i < K; i++) {
    /* tmp_twid = permute(y) * D, with D the stage-i twiddle matrix */
    for (r = 0; r < R; r++)
      for (p = 0; p < Li; p++)
        for (m = 0; m < Mi; m++)
          tmp_twid[r * Li * Mi + p * Mi + m] =
              y[p * R * Mi + r * Mi + m] * cexp(-2.0 * M_PI * I * (double)(r * p) / (double)(Li * R));

    /* y = reshape(dft_mat @ reshape(tmp_twid, (R, R**(K-1))), (N,)) */
    for (a = 0; a < R; a++)
      for (c = 0; c < RK1; c++) {
        acc = 0.0;
        for (b = 0; b < R; b++)
          acc += dft_mat[a][b] * tmp_twid[b * RK1 + c];
        y[a * RK1 + c] = acc;
      }

    Li = Li * R;
    Mi = Mi / R;
  }
#pragma endscop
}

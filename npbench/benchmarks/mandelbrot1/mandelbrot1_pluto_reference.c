/* Not a PolyBench kernel: transcribed statement for statement from the NumPy port
 * (mandelbrot1/mandelbrot1_numpy.py: mandelbrot), the escape-time iteration over a fixed grid.
 *
 * The port's masked updates `N[I] = n` and `Z[I] = Z[I]**2 + C[I]`, with `I = abs(Z) < horizon`,
 * are VALUE conditions, not data-dependent subscripts: every cell is visited every iteration and
 * the mask only decides what is stored there. They are written as ternaries so the write set
 * stays the full rectangle, which is what the port's masked assignment does too.
 *
 * `C` is the port's own transient grid, lifted to a parameter. Z and N are returned, so their
 * element types are the port's: `double _Complex` and int64.
 *
 * `np.linspace(a, b, n)` is `a + i*(b - a)/(n - 1)` with the LAST point set to `b` exactly,
 * which is what numpy does (it does not let rounding decide the endpoint).
 */
#include <stdint.h>
#include <complex.h>
#include <math.h>
#define DATA_TYPE double _Complex

void mandelbrot1_fp64(int64_t XN, int64_t YN, int64_t maxiter, double _Complex C[restrict YN][XN],
                      double _Complex Z[restrict YN][XN], int64_t N[restrict YN][XN], double xmin, double xmax,
                      double ymin, double ymax, double horizon) {

  int i, j, n;

#pragma scop
  /* C = X + Y[:, None] * 1j over the linspace grid */
  for (j = 0; j < YN; j++)
    for (i = 0; i < XN; i++) {
      C[j][i] = (xmin + (double)i * (xmax - xmin) / (double)(XN - 1)) +
                (ymin + (double)j * (ymax - ymin) / (double)(YN - 1)) * I;
      Z[j][i] = 0.0;
      N[j][i] = 0;
    }
  for (j = 0; j < YN; j++)
    C[j][XN - 1] = xmax + cimag(C[j][XN - 1]) * I;
  for (i = 0; i < XN; i++)
    C[YN - 1][i] = creal(C[YN - 1][i]) + ymax * I;

  /* for n in range(maxiter): I = |Z| < horizon; N[I] = n; Z[I] = Z[I]**2 + C[I] */
  for (n = 0; n < maxiter; n++)
    for (j = 0; j < YN; j++)
      for (i = 0; i < XN; i++) {
        N[j][i] = (cabs(Z[j][i]) < horizon) ? n : N[j][i];
        Z[j][i] = (cabs(Z[j][i]) < horizon) ? Z[j][i] * Z[j][i] + C[j][i] : Z[j][i];
      }

  /* N[N == maxiter - 1] = 0 */
  for (j = 0; j < YN; j++)
    for (i = 0; i < XN; i++)
      N[j][i] = (N[j][i] == maxiter - 1) ? 0 : N[j][i];
#pragma endscop
}

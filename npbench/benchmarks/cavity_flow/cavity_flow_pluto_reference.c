/* Not a PolyBench kernel. Transcribed from the NumPy port
 * (cavity_flow_numpy.py: build_up_b, pressure_poisson, cavity_flow) statement for
 * statement, in the port's own order and float64 precision. The three functions
 * are inlined into one loop nest because every trip count (nt, nit, ny, nx) is a
 * scop parameter and every subscript is affine, so the whole solver is one scop.
 *
 * b, pn, un and vn are the port's own temporaries (np.zeros((ny, nx)),
 * np.zeros_like(p)/(u)/(v)) and are passed as caller-allocated scratch
 * parameters, PolyBench/C's own convention for scratch and what keeps them out
 * of Pluto's scop-local dead-code elimination.
 *
 * `pn = p.copy()` at the top of every q iteration is kept as a real full-array
 * copy statement (not aliased to p) because the interior stencil below reads pn
 * one cell outside the interior, including cells adjacent to the boundary; and
 * every boundary assignment after each stencil update is kept in the port's own
 * order, since later assignments overwrite corner cells set by earlier ones. */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

void cavity_flow_fp64(int64_t ny, int64_t nx, int64_t nt, int64_t nit, double u[restrict ny][nx],
                      double v[restrict ny][nx], double p[restrict ny][nx], double b[restrict ny][nx],
                      double pn[restrict ny][nx], double un[restrict ny][nx], double vn[restrict ny][nx], double dt,
                      double dx, double dy, double rho, double nu) {

  int i, j, n, q;

#pragma scop
  for (n = 0; n < nt; n++) {
    for (i = 0; i < ny; i++)
      for (j = 0; j < nx; j++) {
        un[i][j] = u[i][j];
        vn[i][j] = v[i][j];
      }

    /* build_up_b */
    for (i = 1; i < ny - 1; i++)
      for (j = 1; j < nx - 1; j++)
        b[i][j] = rho * (1.0 / dt *
                             ((u[i][j + 1] - u[i][j - 1]) / (2.0 * dx) + (v[i + 1][j] - v[i - 1][j]) / (2.0 * dy)) -
                         ((u[i][j + 1] - u[i][j - 1]) / (2.0 * dx)) * ((u[i][j + 1] - u[i][j - 1]) / (2.0 * dx)) -
                         2.0 * ((u[i + 1][j] - u[i - 1][j]) / (2.0 * dy) *
                                (v[i][j + 1] - v[i][j - 1]) / (2.0 * dx)) -
                         ((v[i + 1][j] - v[i - 1][j]) / (2.0 * dy)) * ((v[i + 1][j] - v[i - 1][j]) / (2.0 * dy)));

    /* pressure_poisson */
    for (q = 0; q < nit; q++) {
      for (i = 0; i < ny; i++)
        for (j = 0; j < nx; j++)
          pn[i][j] = p[i][j];

      for (i = 1; i < ny - 1; i++)
        for (j = 1; j < nx - 1; j++)
          p[i][j] = ((pn[i][j + 1] + pn[i][j - 1]) * dy * dy + (pn[i + 1][j] + pn[i - 1][j]) * dx * dx) /
                        (2.0 * (dx * dx + dy * dy)) -
                    dx * dx * dy * dy / (2.0 * (dx * dx + dy * dy)) * b[i][j];

      for (i = 0; i < ny; i++)
        p[i][nx - 1] = p[i][nx - 2]; /* dp/dx = 0 at x = 2 */
      for (j = 0; j < nx; j++)
        p[0][j] = p[1][j]; /* dp/dy = 0 at y = 0 */
      for (i = 0; i < ny; i++)
        p[i][0] = p[i][1]; /* dp/dx = 0 at x = 0 */
      for (j = 0; j < nx; j++)
        p[ny - 1][j] = 0.0; /* p = 0 at y = 2 */
    }

    for (i = 1; i < ny - 1; i++)
      for (j = 1; j < nx - 1; j++)
        u[i][j] = un[i][j] - un[i][j] * dt / dx * (un[i][j] - un[i][j - 1]) -
                  vn[i][j] * dt / dy * (un[i][j] - un[i - 1][j]) -
                  dt / (2.0 * rho * dx) * (p[i][j + 1] - p[i][j - 1]) +
                  nu * (dt / (dx * dx) * (un[i][j + 1] - 2.0 * un[i][j] + un[i][j - 1]) +
                       dt / (dy * dy) * (un[i + 1][j] - 2.0 * un[i][j] + un[i - 1][j]));

    for (i = 1; i < ny - 1; i++)
      for (j = 1; j < nx - 1; j++)
        v[i][j] = vn[i][j] - un[i][j] * dt / dx * (vn[i][j] - vn[i][j - 1]) -
                  vn[i][j] * dt / dy * (vn[i][j] - vn[i - 1][j]) -
                  dt / (2.0 * rho * dy) * (p[i + 1][j] - p[i - 1][j]) +
                  nu * (dt / (dx * dx) * (vn[i][j + 1] - 2.0 * vn[i][j] + vn[i][j - 1]) +
                       dt / (dy * dy) * (vn[i + 1][j] - 2.0 * vn[i][j] + vn[i - 1][j]));

    for (j = 0; j < nx; j++)
      u[0][j] = 0.0;
    for (i = 0; i < ny; i++)
      u[i][0] = 0.0;
    for (i = 0; i < ny; i++)
      u[i][nx - 1] = 0.0;
    for (j = 0; j < nx; j++)
      u[ny - 1][j] = 1.0; /* set velocity on cavity lid equal to 1 */
    for (j = 0; j < nx; j++)
      v[0][j] = 0.0;
    for (j = 0; j < nx; j++)
      v[ny - 1][j] = 0.0;
    for (i = 0; i < ny; i++)
      v[i][0] = 0.0;
    for (i = 0; i < ny; i++)
      v[i][nx - 1] = 0.0;
  }
#pragma endscop
}

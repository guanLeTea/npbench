/* Not a PolyBench kernel. Transcribed from the NumPy port
 * (channel_flow_numpy.py: build_up_b, pressure_poisson_periodic, channel_flow)
 * statement for statement, in the port's own order and float64 precision.
 *
 * The port's outer loop is `while udiff > .001` -- a DATA-DEPENDENT trip count,
 * not affine, so it cannot be part of a scop. It is kept as an ordinary C
 * `while` around the scop: polycc processes a translation unit whose
 * the scop pragma sits inside a `while` loop, it simply does not model the
 * `while` itself.
 *
 * OUTSIDE the scop (ordinary C, inside the `while`): the `un = u.copy()` /
 * `vn = v.copy()` snapshot taken at the top of every iteration, and, at the
 * bottom, the `udiff = (sum(u) - sum(un)) / sum(u)` reduction and the
 * `stepcount` increment that the `while` condition and the port's return value
 * depend on.
 *
 * INSIDE the scop: build_up_b (interior plus the two periodic-in-x pressure
 * source columns), pressure_poisson_periodic in full (its own `nit`-trip q
 * loop, with `pn = p.copy()` kept as a real full-array copy statement every
 * iteration -- the periodic column update reads pn one cell outside the
 * interior, wrapping to column 0/nx-1 -- then the interior/periodic/wall
 * pressure updates in the port's order), and the u/v interior update, the four
 * periodic-in-x velocity BCs (u then v, x=2 then x=0) and the two wall BCs
 * (u, v both zero at y=0 and y=2), all in the port's own order since later
 * boundary assignments can overwrite cells an earlier one just set.
 *
 * b, pn, un and vn are the port's own temporaries (np.zeros_like(u)/(p)/(u)/(v))
 * and are passed as caller-allocated scratch parameters, PolyBench/C's own
 * convention for scratch and what keeps them out of Pluto's scop-local
 * dead-code elimination. `stepcount` is the port's own return value; there is
 * no way to return a scalar through this column's `void <base>_fp64` symbol,
 * so it is written through a one-element pointer instead. */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

void channel_flow_fp64(int64_t ny, int64_t nx, int64_t nit, double u[restrict ny][nx], double v[restrict ny][nx],
                       double p[restrict ny][nx], double b[restrict ny][nx], double pn[restrict ny][nx],
                       double un[restrict ny][nx], double vn[restrict ny][nx], int64_t *restrict stepcount, double dt,
                       double dx, double dy, double rho, double nu, double F) {

  int i, j, q;
  double udiff = 1.0;
  int64_t count = 0;
  double usum, unsum;

  while (udiff > 0.001) {
    for (i = 0; i < ny; i++)
      for (j = 0; j < nx; j++) {
        un[i][j] = u[i][j];
        vn[i][j] = v[i][j];
      }

#pragma scop
    /* build_up_b: interior */
    for (i = 1; i < ny - 1; i++)
      for (j = 1; j < nx - 1; j++)
        b[i][j] = rho * (1.0 / dt *
                             ((u[i][j + 1] - u[i][j - 1]) / (2.0 * dx) + (v[i + 1][j] - v[i - 1][j]) / (2.0 * dy)) -
                         ((u[i][j + 1] - u[i][j - 1]) / (2.0 * dx)) * ((u[i][j + 1] - u[i][j - 1]) / (2.0 * dx)) -
                         2.0 * ((u[i + 1][j] - u[i - 1][j]) / (2.0 * dy) *
                                (v[i][j + 1] - v[i][j - 1]) / (2.0 * dx)) -
                         ((v[i + 1][j] - v[i - 1][j]) / (2.0 * dy)) * ((v[i + 1][j] - v[i - 1][j]) / (2.0 * dy)));

    /* build_up_b: periodic BC pressure source @ x = 2 (column nx-1) */
    for (i = 1; i < ny - 1; i++)
      b[i][nx - 1] =
          rho * (1.0 / dt *
                     ((u[i][0] - u[i][nx - 2]) / (2.0 * dx) + (v[i + 1][nx - 1] - v[i - 1][nx - 1]) / (2.0 * dy)) -
                 ((u[i][0] - u[i][nx - 2]) / (2.0 * dx)) * ((u[i][0] - u[i][nx - 2]) / (2.0 * dx)) -
                 2.0 * ((u[i + 1][nx - 1] - u[i - 1][nx - 1]) / (2.0 * dy) *
                        (v[i][0] - v[i][nx - 2]) / (2.0 * dx)) -
                 ((v[i + 1][nx - 1] - v[i - 1][nx - 1]) / (2.0 * dy)) *
                     ((v[i + 1][nx - 1] - v[i - 1][nx - 1]) / (2.0 * dy)));

    /* build_up_b: periodic BC pressure source @ x = 0 (column 0) */
    for (i = 1; i < ny - 1; i++)
      b[i][0] =
          rho * (1.0 / dt * ((u[i][1] - u[i][nx - 1]) / (2.0 * dx) + (v[i + 1][0] - v[i - 1][0]) / (2.0 * dy)) -
                 ((u[i][1] - u[i][nx - 1]) / (2.0 * dx)) * ((u[i][1] - u[i][nx - 1]) / (2.0 * dx)) -
                 2.0 * ((u[i + 1][0] - u[i - 1][0]) / (2.0 * dy) * (v[i][1] - v[i][nx - 1]) / (2.0 * dx)) -
                 ((v[i + 1][0] - v[i - 1][0]) / (2.0 * dy)) * ((v[i + 1][0] - v[i - 1][0]) / (2.0 * dy)));

    /* pressure_poisson_periodic */
    for (q = 0; q < nit; q++) {
      for (i = 0; i < ny; i++)
        for (j = 0; j < nx; j++)
          pn[i][j] = p[i][j];

      for (i = 1; i < ny - 1; i++)
        for (j = 1; j < nx - 1; j++)
          p[i][j] = ((pn[i][j + 1] + pn[i][j - 1]) * dy * dy + (pn[i + 1][j] + pn[i - 1][j]) * dx * dx) /
                        (2.0 * (dx * dx + dy * dy)) -
                    dx * dx * dy * dy / (2.0 * (dx * dx + dy * dy)) * b[i][j];

      /* periodic BC pressure @ x = 2 (column nx-1) */
      for (i = 1; i < ny - 1; i++)
        p[i][nx - 1] = ((pn[i][0] + pn[i][nx - 2]) * dy * dy + (pn[i + 1][nx - 1] + pn[i - 1][nx - 1]) * dx * dx) /
                           (2.0 * (dx * dx + dy * dy)) -
                       dx * dx * dy * dy / (2.0 * (dx * dx + dy * dy)) * b[i][nx - 1];

      /* periodic BC pressure @ x = 0 (column 0) */
      for (i = 1; i < ny - 1; i++)
        p[i][0] = ((pn[i][1] + pn[i][nx - 1]) * dy * dy + (pn[i + 1][0] + pn[i - 1][0]) * dx * dx) /
                      (2.0 * (dx * dx + dy * dy)) -
                  dx * dx * dy * dy / (2.0 * (dx * dx + dy * dy)) * b[i][0];

      /* wall boundary conditions, pressure */
      for (j = 0; j < nx; j++)
        p[ny - 1][j] = p[ny - 2][j]; /* dp/dy = 0 at y = 2 */
      for (j = 0; j < nx; j++)
        p[0][j] = p[1][j]; /* dp/dy = 0 at y = 0 */
    }

    for (i = 1; i < ny - 1; i++)
      for (j = 1; j < nx - 1; j++)
        u[i][j] = un[i][j] - un[i][j] * dt / dx * (un[i][j] - un[i][j - 1]) -
                  vn[i][j] * dt / dy * (un[i][j] - un[i - 1][j]) -
                  dt / (2.0 * rho * dx) * (p[i][j + 1] - p[i][j - 1]) +
                  nu * (dt / (dx * dx) * (un[i][j + 1] - 2.0 * un[i][j] + un[i][j - 1]) +
                       dt / (dy * dy) * (un[i + 1][j] - 2.0 * un[i][j] + un[i - 1][j])) +
                  F * dt;

    for (i = 1; i < ny - 1; i++)
      for (j = 1; j < nx - 1; j++)
        v[i][j] = vn[i][j] - un[i][j] * dt / dx * (vn[i][j] - vn[i][j - 1]) -
                  vn[i][j] * dt / dy * (vn[i][j] - vn[i - 1][j]) -
                  dt / (2.0 * rho * dy) * (p[i + 1][j] - p[i - 1][j]) +
                  nu * (dt / (dx * dx) * (vn[i][j + 1] - 2.0 * vn[i][j] + vn[i][j - 1]) +
                       dt / (dy * dy) * (vn[i + 1][j] - 2.0 * vn[i][j] + vn[i - 1][j]));

    /* periodic BC u @ x = 2 (column nx-1) */
    for (i = 1; i < ny - 1; i++)
      u[i][nx - 1] = un[i][nx - 1] - un[i][nx - 1] * dt / dx * (un[i][nx - 1] - un[i][nx - 2]) -
                     vn[i][nx - 1] * dt / dy * (un[i][nx - 1] - un[i - 1][nx - 1]) -
                     dt / (2.0 * rho * dx) * (p[i][0] - p[i][nx - 2]) +
                     nu * (dt / (dx * dx) * (un[i][0] - 2.0 * un[i][nx - 1] + un[i][nx - 2]) +
                          dt / (dy * dy) * (un[i + 1][nx - 1] - 2.0 * un[i][nx - 1] + un[i - 1][nx - 1])) +
                     F * dt;

    /* periodic BC u @ x = 0 (column 0) */
    for (i = 1; i < ny - 1; i++)
      u[i][0] = un[i][0] - un[i][0] * dt / dx * (un[i][0] - un[i][nx - 1]) -
                vn[i][0] * dt / dy * (un[i][0] - un[i - 1][0]) -
                dt / (2.0 * rho * dx) * (p[i][1] - p[i][nx - 1]) +
                nu * (dt / (dx * dx) * (un[i][1] - 2.0 * un[i][0] + un[i][nx - 1]) +
                     dt / (dy * dy) * (un[i + 1][0] - 2.0 * un[i][0] + un[i - 1][0])) +
                F * dt;

    /* periodic BC v @ x = 2 (column nx-1) */
    for (i = 1; i < ny - 1; i++)
      v[i][nx - 1] = vn[i][nx - 1] - un[i][nx - 1] * dt / dx * (vn[i][nx - 1] - vn[i][nx - 2]) -
                     vn[i][nx - 1] * dt / dy * (vn[i][nx - 1] - vn[i - 1][nx - 1]) -
                     dt / (2.0 * rho * dy) * (p[i + 1][nx - 1] - p[i - 1][nx - 1]) +
                     nu * (dt / (dx * dx) * (vn[i][0] - 2.0 * vn[i][nx - 1] + vn[i][nx - 2]) +
                          dt / (dy * dy) * (vn[i + 1][nx - 1] - 2.0 * vn[i][nx - 1] + vn[i - 1][nx - 1]));

    /* periodic BC v @ x = 0 (column 0) */
    for (i = 1; i < ny - 1; i++)
      v[i][0] = vn[i][0] - un[i][0] * dt / dx * (vn[i][0] - vn[i][nx - 1]) -
                vn[i][0] * dt / dy * (vn[i][0] - vn[i - 1][0]) -
                dt / (2.0 * rho * dy) * (p[i + 1][0] - p[i - 1][0]) +
                nu * (dt / (dx * dx) * (vn[i][1] - 2.0 * vn[i][0] + vn[i][nx - 1]) +
                     dt / (dy * dy) * (vn[i + 1][0] - 2.0 * vn[i][0] + vn[i - 1][0]));

    /* wall BC: u, v = 0 @ y = 0, 2 */
    for (j = 0; j < nx; j++)
      u[0][j] = 0.0;
    for (j = 0; j < nx; j++)
      u[ny - 1][j] = 0.0;
    for (j = 0; j < nx; j++)
      v[0][j] = 0.0;
    for (j = 0; j < nx; j++)
      v[ny - 1][j] = 0.0;
#pragma endscop

    usum = 0.0;
    unsum = 0.0;
    for (i = 0; i < ny; i++)
      for (j = 0; j < nx; j++) {
        usum += u[i][j];
        unsum += un[i][j];
      }
    udiff = (usum - unsum) / usum;
    count++;
  }

  *stepcount = count;
}

/* Not a PolyBench kernel: transcribed statement for statement from the NumPy port
 * (scattering_self_energies/scattering_self_energies_numpy.py), an 8-deep nest of Norb x Norb
 * matrix products accumulated into Sigma.
 *
 * Written out in full and handed to polycc on purpose, so the refusal on file is Pluto's own.
 * Two properties of the kernel are expected to stop it, and neither is rewritten away:
 *
 *  - `G[k, E - w, neigh_idx[a, b]]` is a DATA-DEPENDENT SUBSCRIPT: which slab of G a statement
 *    reads is a value stored in neigh_idx, not a function of the iteration vector. No exact
 *    polyhedral representation of that access exists. Replacing neigh_idx by the closed form
 *    its initializer happens to use (`(a - NB/2 + b) % NA`) would be a different kernel -- the
 *    port reads the array.
 *  - Every array but neigh_idx is complex128, and Sigma is the returned result, so the element
 *    type is `double _Complex` rather than split into re/im arrays. That part is NOT a
 *    blocker: this column marshals complex128, and `contour_integral` and `mandelbrot1`
 *    validate complex results through it. The indirect subscript is the whole reason.
 *
 * Measured: `[Clan] Error: syntax error at line 49, column 61`, which is the
 * `neigh_idx[a][b]` inside G's subscript -- clan's grammar does not admit it at all.
 *
 * `if E - w >= 0` is an affine guard on the iteration vector and is kept as a guard.
 * `dHG` and `dHD` are the port's two per-iteration temporaries, lifted to parameters.
 */
#include <stdint.h>
#include <complex.h>
#define DATA_TYPE double _Complex

void scattering_self_energies_fp64(int64_t Nkz, int64_t NE, int64_t Nqz, int64_t Nw, int64_t N3D, int64_t NA,
                                   int64_t NB, int64_t Norb, const int32_t neigh_idx[restrict NA][NB],
                                   const double _Complex dH[restrict NA][NB][N3D][Norb][Norb],
                                   const double _Complex G[restrict Nkz][NE][NA][Norb][Norb],
                                   const double _Complex D[restrict Nqz][Nw][NA][NB][N3D][N3D],
                                   double _Complex Sigma[restrict Nkz][NE][NA][Norb][Norb],
                                   double _Complex dHG[restrict Norb][Norb],
                                   double _Complex dHD[restrict Norb][Norb]) {

  int k, E, q, w, i, j, a, b, r, c, t;

#pragma scop
  for (k = 0; k < Nkz; k++)
    for (E = 0; E < NE; E++)
      for (q = 0; q < Nqz; q++)
        for (w = 0; w < Nw; w++)
          for (i = 0; i < N3D; i++)
            for (j = 0; j < N3D; j++)
              for (a = 0; a < NA; a++)
                for (b = 0; b < NB; b++)
                  if (E - w >= 0) {
                    /* dHG = G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i] */
                    for (r = 0; r < Norb; r++)
                      for (c = 0; c < Norb; c++) {
                        dHG[r][c] = 0.0;
                        for (t = 0; t < Norb; t++)
                          dHG[r][c] += G[k][E - w][neigh_idx[a][b]][r][t] * dH[a][b][i][t][c];
                      }
                    /* dHD = dH[a, b, j] * D[q, w, a, b, i, j] */
                    for (r = 0; r < Norb; r++)
                      for (c = 0; c < Norb; c++)
                        dHD[r][c] = dH[a][b][j][r][c] * D[q][w][a][b][i][j];
                    /* Sigma[k, E, a] += dHG @ dHD */
                    for (r = 0; r < Norb; r++)
                      for (c = 0; c < Norb; c++)
                        for (t = 0; t < Norb; t++)
                          Sigma[k][E][a][r][c] += dHG[r][t] * dHD[t][c];
                  }
#pragma endscop
}

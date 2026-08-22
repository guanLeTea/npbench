/* Not a PolyBench kernel: transcribed statement for statement from the NumPy port
 * (nbody/nbody_numpy.py: getAcc, getEnergy, nbody), float64 throughout, same statement
 * order and same outer time-step count as the port.
 *
 * Adaptations, all read off the port rather than chosen to make numbers match:
 *  - `acc` (the leapfrog's carried acceleration, N x 3) and `corr` (the center-of-mass
 *    velocity correction, a length-3 vector) are scratch ARRAY parameters, never locals,
 *    per this column's contract.
 *  - `ddx`/`ddy`/`ddz`/`r2`/`invr3`/`invr`/`sum_mass` are scalar accumulators, declared
 *    once above the scop region and reused across statements exactly like PolyBench's own
 *    `symm` reuses `temp2` -- not scratch arrays, so they stay locals.
 *  - `getAcc` and `getEnergy` both build a full N x N pairwise matrix in the port
 *    (`dx = x.T - x`, etc.) only to reduce it immediately (`@ mass`, or a triu sum); no
 *    intermediate of either survives past that reduction, so this scop computes each
 *    pairwise term inline in the same statement that consumes it (scalars, not an N x N
 *    scratch array) rather than materializing and re-reading a matrix nothing else reads.
 *  - `inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)` and `inv_r[inv_r > 0] = 1.0/inv_r[...]`
 *    are the ternaries `(r2 > 0.0) ? pow(r2, -1.5) : r2` and `(invr > 0.0) ? 1.0/invr : invr`,
 *    keeping the value condition inside the statement rather than an `if` around it.
 *  - `PE`'s `np.triu(-(mass*mass.T)*inv_r, 1)` sums the STRICT upper triangle (i < j) only;
 *    summing `j` from `i + 1` instead of `0` and skipping the masked-to-zero entries is the
 *    same sum (adding an exact zero changes nothing), not a reordering of it.
 *  - `t += dt` is dropped: the port increments it every iteration but never reads it again,
 *    so it cannot affect `KE`/`PE`/`pos`/`vel`.
 *  - `KE[i+1]`/`PE[i+1]` and the initial `KE[0]`/`PE[0]` write into the SAME output arrays
 *    the port returns; each cell is written exactly once, so an explicit `= 0.0` reset per
 *    cell before its accumulation loop is used (matching this column's `mlp` reference)
 *    rather than relying on the adapter's zero-fill.
 */
#include <stdint.h>
#include <math.h>
#define DATA_TYPE double

void nbody_fp64(int64_t N, int64_t Nt, const double mass[restrict N][1], double pos[restrict N][3],
                double vel[restrict N][3], double acc[restrict N][3], double corr[restrict 3],
                double KE[restrict Nt + 1], double PE[restrict Nt + 1], double dt, double G, double softening) {

  int n, i, j, t, d;
  double ddx, ddy, ddz, r2, invr3, invr, sum_mass;

#pragma scop
  /* Convert to Center-of-Mass frame: vel -= mean(mass * vel, axis=0) / mean(mass) */
  for (d = 0; d < 3; d++)
    corr[d] = 0.0;
  sum_mass = 0.0;
  for (n = 0; n < N; n++) {
    sum_mass += mass[n][0];
    for (d = 0; d < 3; d++)
      corr[d] += mass[n][0] * vel[n][d];
  }
  for (d = 0; d < 3; d++)
    corr[d] /= sum_mass;
  for (n = 0; n < N; n++)
    for (d = 0; d < 3; d++)
      vel[n][d] -= corr[d];

  /* acc = getAcc(pos, mass, G, softening) */
  for (i = 0; i < N; i++) {
    acc[i][0] = 0.0;
    acc[i][1] = 0.0;
    acc[i][2] = 0.0;
    for (j = 0; j < N; j++) {
      ddx = pos[j][0] - pos[i][0];
      ddy = pos[j][1] - pos[i][1];
      ddz = pos[j][2] - pos[i][2];
      r2 = ddx * ddx + ddy * ddy + ddz * ddz + softening * softening;
      invr3 = (r2 > 0.0) ? pow(r2, -1.5) : r2;
      acc[i][0] += G * (ddx * invr3) * mass[j][0];
      acc[i][1] += G * (ddy * invr3) * mass[j][0];
      acc[i][2] += G * (ddz * invr3) * mass[j][0];
    }
  }

  /* KE[0], PE[0] = getEnergy(pos, vel, mass, G) */
  KE[0] = 0.0;
  for (n = 0; n < N; n++)
    for (d = 0; d < 3; d++)
      KE[0] += 0.5 * mass[n][0] * vel[n][d] * vel[n][d];

  PE[0] = 0.0;
  for (i = 0; i < N; i++)
    for (j = i + 1; j < N; j++) {
      ddx = pos[j][0] - pos[i][0];
      ddy = pos[j][1] - pos[i][1];
      ddz = pos[j][2] - pos[i][2];
      invr = sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
      invr = (invr > 0.0) ? (1.0 / invr) : invr;
      PE[0] += G * (-(mass[i][0] * mass[j][0]) * invr);
    }

  /* Simulation Main Loop */
  for (t = 0; t < Nt; t++) {
    /* (1/2) kick */
    for (n = 0; n < N; n++)
      for (d = 0; d < 3; d++)
        vel[n][d] += acc[n][d] * dt / 2.0;

    /* drift */
    for (n = 0; n < N; n++)
      for (d = 0; d < 3; d++)
        pos[n][d] += vel[n][d] * dt;

    /* acc = getAcc(pos, mass, G, softening) */
    for (i = 0; i < N; i++) {
      acc[i][0] = 0.0;
      acc[i][1] = 0.0;
      acc[i][2] = 0.0;
      for (j = 0; j < N; j++) {
        ddx = pos[j][0] - pos[i][0];
        ddy = pos[j][1] - pos[i][1];
        ddz = pos[j][2] - pos[i][2];
        r2 = ddx * ddx + ddy * ddy + ddz * ddz + softening * softening;
        invr3 = (r2 > 0.0) ? pow(r2, -1.5) : r2;
        acc[i][0] += G * (ddx * invr3) * mass[j][0];
        acc[i][1] += G * (ddy * invr3) * mass[j][0];
        acc[i][2] += G * (ddz * invr3) * mass[j][0];
      }
    }

    /* (1/2) kick */
    for (n = 0; n < N; n++)
      for (d = 0; d < 3; d++)
        vel[n][d] += acc[n][d] * dt / 2.0;

    /* KE[t+1], PE[t+1] = getEnergy(pos, vel, mass, G) */
    KE[t + 1] = 0.0;
    for (n = 0; n < N; n++)
      for (d = 0; d < 3; d++)
        KE[t + 1] += 0.5 * mass[n][0] * vel[n][d] * vel[n][d];

    PE[t + 1] = 0.0;
    for (i = 0; i < N; i++)
      for (j = i + 1; j < N; j++) {
        ddx = pos[j][0] - pos[i][0];
        ddy = pos[j][1] - pos[i][1];
        ddz = pos[j][2] - pos[i][2];
        invr = sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
        invr = (invr > 0.0) ? (1.0 / invr) : invr;
        PE[t + 1] += G * (-(mass[i][0] * mass[j][0]) * invr);
      }
  }
#pragma endscop
}

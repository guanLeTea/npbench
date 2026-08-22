/* Loop-nest transcription of NPBench's vadv NumPy port
 * (npbench/benchmarks/weather_stencils/vadv/vadv_numpy.py), statement for statement and in its
 * own float64 precision: five loop nests in the port's own order -- k=0, k=1..K-2, k=K-1, then
 * the two backward Thomas-solver sweeps k=K-1 and k=K-2..0. Not a PolyBench kernel.
 *
 * BET_M/BET_P are the port's own module-level constants (both 0.5), never passed as arguments,
 * so they are `#define`d literals here rather than scop parameters. GAV/GCV/ACOL/AS/CCOL_RAW/CS
 * are the port's own per-statement scalar temporaries (gav, gcv, acol, as_, cs and ccol's
 * not-yet-divided value); each is a function-like macro expanding to the same arithmetic the
 * port evaluates, not a stored variable -- the same role SQRT_FUN/EXP_FUN play in the tracked
 * PolyBench scops (gemm, gesummv). A data-typed local declared inside the scop is scratch
 * Pluto's dead-code screen can eliminate, so none of the port's scalar temporaries becomes one;
 * these macros are how the arithmetic is shared across statements without that.
 *
 * I, J, K come from utens_stage's own shape; wcon is (I+1, J, K), and as with hdiff's in_field
 * only the trailing two extents of a VLA parameter matter for its addressing, so its leading
 * extent is written honestly as `I + 1`. ccol, dcol and data_col are the port's own
 * Thomas-solver temporaries and are passed as caller-allocated scratch parameters. utens_stage
 * is both an input and the sole output (already `output_args`), so it stays one non-const
 * parameter.
 *
 * One adaptation, forced by C's in-place update semantics vs. the port's snapshot semantics,
 * not changing any value computed: in the k=0 and k=1..K-2 loops the port computes `divided`
 * once and applies it to both ccol and dcol, but `divided` depends on ccol's PRE-division
 * value. Applying it to dcol first and ccol second (as done below) reads that same pre-division
 * value in both places -- they are disjoint arrays, so nothing else depends on this order --
 * whereas the port's own ccol-then-dcol order would, in C, have the ccol store overwrite the
 * value the dcol expression still needs.
 *
 * The k=K-2..0 backward sweep carries a genuine k-to-k dependence through data_col, so an
 * honest Pluto result here is `PlutoSequential` -- no loop marked parallel -- not a failure.
 *
 * Each of the port's `for k in range(a, a+1)` single-value ranges (its k=0 and k=K-1 blocks)
 * is written as a degenerate `for (k = a; k < a+1; k++)` rather than a plain `k = a;`
 * assignment: Clan rejects a bare assignment to a variable later used as an array subscript
 * ("variable or array reference in an affine expression"), since only a genuine loop iterator
 * or a scop parameter is affine in its model. Still exactly one value of k, exactly as the
 * port ranges over. */
#include <stdint.h>

#define BET_M 0.5
#define BET_P 0.5

#define GAV(i, j, k) (-0.25 * (wcon[i + 1][j][k] + wcon[i][j][k]))
#define GCV(i, j, k) (0.25 * (wcon[i + 1][j][k + 1] + wcon[i][j][k + 1]))
#define ACOL(i, j, k) (GAV(i, j, k) * BET_P)
#define AS(i, j, k) (GAV(i, j, k) * BET_M)
#define CCOL_RAW(i, j, k) (GCV(i, j, k) * BET_P)
#define CS(i, j, k) (GCV(i, j, k) * BET_M)

void vadv_fp64(int64_t I, int64_t J, int64_t K, double utens_stage[restrict I][J][K],
              const double u_stage[restrict I][J][K], const double wcon[restrict I + 1][J][K],
              const double u_pos[restrict I][J][K], const double utens[restrict I][J][K],
              double ccol[restrict I][J][K], double dcol[restrict I][J][K], double data_col[restrict I][J],
              double dtr_stage) {

  int i, j, k;

#pragma scop
  /* k = 0 */
  for (k = 0; k < 1; k++)
    for (i = 0; i < I; i++)
      for (j = 0; j < J; j++) {
        ccol[i][j][k] = CCOL_RAW(i, j, k);
        dcol[i][j][k] = dtr_stage * u_pos[i][j][k] + utens[i][j][k] + utens_stage[i][j][k] -
                       CS(i, j, k) * (u_stage[i][j][k + 1] - u_stage[i][j][k]);
        dcol[i][j][k] = dcol[i][j][k] / (dtr_stage - ccol[i][j][k]);
        ccol[i][j][k] = ccol[i][j][k] / (dtr_stage - ccol[i][j][k]);
      }

  /* k = 1 .. K-2 */
  for (k = 1; k < K - 1; k++)
    for (i = 0; i < I; i++)
      for (j = 0; j < J; j++) {
        ccol[i][j][k] = CCOL_RAW(i, j, k);
        dcol[i][j][k] = dtr_stage * u_pos[i][j][k] + utens[i][j][k] + utens_stage[i][j][k] -
                       AS(i, j, k) * (u_stage[i][j][k - 1] - u_stage[i][j][k]) -
                       CS(i, j, k) * (u_stage[i][j][k + 1] - u_stage[i][j][k]);
        dcol[i][j][k] = (dcol[i][j][k] - dcol[i][j][k - 1] * ACOL(i, j, k)) /
                       (dtr_stage - ACOL(i, j, k) - ccol[i][j][k] - ccol[i][j][k - 1] * ACOL(i, j, k));
        ccol[i][j][k] = ccol[i][j][k] /
                       (dtr_stage - ACOL(i, j, k) - ccol[i][j][k] - ccol[i][j][k - 1] * ACOL(i, j, k));
      }

  /* k = K-1 */
  for (k = K - 1; k < K; k++)
    for (i = 0; i < I; i++)
      for (j = 0; j < J; j++) {
        dcol[i][j][k] = dtr_stage * u_pos[i][j][k] + utens[i][j][k] + utens_stage[i][j][k] -
                       AS(i, j, k) * (u_stage[i][j][k - 1] - u_stage[i][j][k]);
        dcol[i][j][k] = (dcol[i][j][k] - dcol[i][j][k - 1] * ACOL(i, j, k)) /
                       (dtr_stage - ACOL(i, j, k) - ccol[i][j][k - 1] * ACOL(i, j, k));
      }

  /* backward sweep, k = K-1 */
  for (k = K - 1; k < K; k++)
    for (i = 0; i < I; i++)
      for (j = 0; j < J; j++) {
        data_col[i][j] = dcol[i][j][k];
        utens_stage[i][j][k] = dtr_stage * (data_col[i][j] - u_pos[i][j][k]);
      }

  /* backward sweep, k = K-2 .. 0 */
  for (k = K - 2; k >= 0; k--)
    for (i = 0; i < I; i++)
      for (j = 0; j < J; j++) {
        data_col[i][j] = dcol[i][j][k] - ccol[i][j][k] * data_col[i][j];
        utens_stage[i][j][k] = dtr_stage * (data_col[i][j] - u_pos[i][j][k]);
      }
#pragma endscop
}

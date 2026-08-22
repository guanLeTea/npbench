/* Loop-nest transcription of NPBench's go_fast NumPy port
 * (npbench/benchmarks/go_fast/go_fast_numpy.py), statement for statement: the
 * diagonal-tanh reduction into `trace`, then the elementwise `a + trace`. Not a
 * PolyBench kernel.
 *
 * `trace` is the port's own scalar reduction; it is passed as a one-element
 * scratch parameter rather than a function local, the same convention PolyBench/C
 * uses for its own scratch (see gesummv's `tmp`, covariance's `mean`). */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

void go_fast_fp64(int64_t N, const double a[restrict N][N], double out[restrict N][N], double *restrict trace) {

  int i, j;

#pragma scop
  trace[0] = SCALAR_VAL(0.0);
  for (i = 0; i < N; i++)
    trace[0] += tanh(a[i][i]);
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      out[i][j] = a[i][j] + trace[0];
#pragma endscop
}

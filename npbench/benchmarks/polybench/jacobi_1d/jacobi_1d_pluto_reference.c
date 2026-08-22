/* Manual verbatim transcription of the PolyBench/C 4.2.1 kernel_jacobi_1d body
 * (polybench.sourceforge.net), adapted only in the function signature: the
 * harness's runtime-sized VLA parameters. PolyBench sweeps
 * `for (t = 0; t < TSTEPS; t++)`; NPBench's port sweeps `for t in range(1, TSTEPS)`,
 * one fewer -- corrected via this sidecar's ARG_OVERRIDES, not by touching the
 * loop bound here. */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

#define _PB_TSTEPS TSTEPS
#define _PB_N N

void jacobi_1d_fp64(int64_t TSTEPS, int64_t N, double A[restrict N], double B[restrict N]) {

  int t, i;

#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++) {
    for (i = 1; i < _PB_N - 1; i++)
      B[i] = SCALAR_VAL(0.33333) * (A[i - 1] + A[i] + A[i + 1]);
    for (i = 1; i < _PB_N - 1; i++)
      A[i] = SCALAR_VAL(0.33333) * (B[i - 1] + B[i] + B[i + 1]);
  }
#pragma endscop
}

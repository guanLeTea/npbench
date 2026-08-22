/* Manual verbatim transcription of the PolyBench/C 4.2.1 kernel_doitgen body
 * (polybench.sourceforge.net), adapted only in the function signature: the
 * harness's runtime-sized VLA parameters. `sum` is a caller-allocated scratch
 * parameter here exactly as it is in PolyBench/C. */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

#define _PB_NR NR
#define _PB_NQ NQ
#define _PB_NP NP

void doitgen_fp64(int64_t NR, int64_t NQ, int64_t NP, double A[restrict NR][NQ][NP],
                  const double C4[restrict NP][NP], double *restrict sum) {

  int r, q, p, s;

#pragma scop
  for (r = 0; r < _PB_NR; r++)
    for (q = 0; q < _PB_NQ; q++) {
      for (p = 0; p < _PB_NP; p++) {
        sum[p] = SCALAR_VAL(0.0);
        for (s = 0; s < _PB_NP; s++)
          sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < _PB_NP; p++)
        A[r][q][p] = sum[p];
    }
#pragma endscop
}

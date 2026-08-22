/* PolyBench/C 4.2.1 original kernel_3mm (polybench.sourceforge.net), adapted to the
 * harness's runtime-sized VLA signature -- see npbench/infrastructure/pluto_framework.py.
 * `E` and `F` are PolyBench/C's own scratch parameters of kernel_3mm (E := A*B, F := C*D),
 * kept as caller-allocated parameters here too; `G := E*F` is the result. The NumPy port
 * (`return A @ B @ C @ D`) has no alpha/beta, unlike kernel_2mm -- kernel_3mm has none
 * either, so the two already agree. */
#include <stdint.h>
#include <math.h>
#define SCALAR_VAL(x) x
#define DATA_TYPE double

void k3mm_fp64(int64_t NI, int64_t NJ, int64_t NK, int64_t NL, int64_t NM, const double A[restrict NI][NK],
               const double B[restrict NK][NJ], const double C[restrict NJ][NM], const double D[restrict NM][NL],
               double E[restrict NI][NJ], double F[restrict NJ][NL], double G[restrict NI][NL]) {

  int i, j, k;

#pragma scop
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++) {
      E[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < NK; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++) {
      F[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < NM; ++k)
        F[i][j] += C[i][k] * D[k][j];
    }
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++) {
      G[i][j] = SCALAR_VAL(0.0);
      for (k = 0; k < NJ; ++k)
        G[i][j] += E[i][k] * F[k][j];
    }
#pragma endscop
}

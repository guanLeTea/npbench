/* Loop-nest transcription of the NumPy port's single elementwise expression over four rank-1
 * float64 arrays. Not a PolyBench kernel: written from the NumPy port statement for statement.
 * `x**2` is written as `x*x`, matching NumPy's own handling of an integer exponent 2, not `pow`.
 * `temp` is the port's own intermediate, promoted from a NumPy temporary into a caller-allocated
 * scratch parameter, the same PolyBench/C convention used for scratch elsewhere in this column --
 * a data-typed function local inside the scop is what the framework's dead-code screen refuses. */
#include <stdint.h>
#include <math.h>

void arc_distance_fp64(int64_t N, const double *restrict theta_1, const double *restrict phi_1,
                       const double *restrict theta_2, const double *restrict phi_2, double *restrict out,
                       double *restrict temp) {

  int i;

#pragma scop
  for (i = 0; i < N; i++) {
    temp[i] = sin((theta_2[i] - theta_1[i]) / 2) * sin((theta_2[i] - theta_1[i]) / 2) +
             cos(theta_1[i]) * cos(theta_2[i]) * sin((phi_2[i] - phi_1[i]) / 2) * sin((phi_2[i] - phi_1[i]) / 2);
    out[i] = 2 * atan2(sqrt(temp[i]), sqrt(1 - temp[i]));
  }
#pragma endscop
}

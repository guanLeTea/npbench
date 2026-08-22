/* Loop-nest transcription of NPBench's crc16 NumPy port
 * (npbench/benchmarks/crc16/crc16_numpy.py), statement for statement: the CRC-16-CCITT
 * shift-and-conditional-xor recurrence over N bytes, 8 bit-iterations each, then the
 * final complement/byte-swap/mask. Not a PolyBench kernel; the domain (N x 8) is affine,
 * but the accumulator `out[0]` is a loop-carried recurrence across both loops, so it is
 * inherently sequential -- polycc marking no loop parallel here is an honest answer, not
 * a defect.
 *
 * The port's `crc` accumulator never goes negative (it only ever holds `(crc >> 1) ^ poly`
 * or `crc >> 1` starting from 0xFFFF, both non-negative for a 16-bit poly), so an int32_t
 * right-shift here reproduces the port's `& 0xFFFF` masked semantics exactly, without any
 * implementation-defined negative shift.
 *
 * `out` doubles as the port's `crc` scalar (a caller-allocated scratch/output parameter,
 * PolyBench/C's convention, rather than a function local); `cur_byte` is an int, not a
 * DATA_TYPE, so it is exempt from the scratch-local rule and stays a plain counter beside
 * `i`/`bit`.
 *
 * The port's input `data` is a NumPy uint8 array (crc16.py's initializer:
 * `rng.integers(0, 256, size=(N,), dtype=np.uint8)`), and is declared `uint8_t` here to
 * match it exactly, now that `_ELEM_TYPES` in pluto_framework.py carries that type. `data[i]`
 * promotes to a plain (signed) `int` under the usual arithmetic conversions before the `&
 * 0xFF` mask -- every value 0..255 is representable in that `int`, so the promotion changes
 * no bit and `cur_byte`'s arithmetic is exactly what it was with an int32_t `data`. The `&
 * 0xFF` is a no-op now that the element itself cannot exceed 0xFF, and is kept so the C reads
 * as the port's own `0xFF & b` does. */
#include <stdint.h>
#define POLY 0x8408

void crc16_fp64(int64_t N, const uint8_t data[restrict N], int32_t *restrict out) {

  int i, bit;
  int cur_byte;

#pragma scop
  out[0] = 0xFFFF;
  for (i = 0; i < N; i++) {
    cur_byte = data[i] & 0xFF;
    for (bit = 0; bit < 8; bit++) {
      out[0] = ((out[0] & 0x0001) ^ (cur_byte & 0x0001)) ? ((out[0] >> 1) ^ POLY) : (out[0] >> 1);
      cur_byte >>= 1;
    }
  }
  out[0] = (~out[0]) & 0xFFFF;
  out[0] = (out[0] << 8) | ((out[0] >> 8) & 0xFF);
  out[0] = out[0] & 0xFFFF;
#pragma endscop
}

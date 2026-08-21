/**
 * @file dot_avx2.cpp
 * @brief AVX2 + FMA vectorized dot product kernel.
 *
 * This is the only translation unit compiled with -mavx2 -mfma (see
 * src/CMakeLists.txt). Isolating the flags here guarantees no other part of
 * the binary contains AVX2 instructions, which makes the runtime dispatch in
 * dot_dispatch.cpp safe on CPUs without AVX2.
 */

// [C#→C++] Intrinsics are the manual counterpart of System.Runtime.Intrinsics
// (Vector256<float> / Avx2.* in .NET). Difference: the C# JIT picks the ISA at
// runtime; here the compiler bakes AVX2 instructions into the binary, so CPU
// dispatch must be done by hand in a separate translation unit.
#include "core/dot.hpp"

#include <cassert>
#include <immintrin.h>  // umbrella header for all x86 intrinsics

namespace vecdb {

float dot_avx2(std::span<const float> a, std::span<const float> b) {
    assert(a.size() == b.size());

    const std::size_t n = a.size();
    const float* pa = a.data();
    const float* pb = b.data();

    // Two independent accumulators hide FMA latency (~4 cycles on most
    // microarchitectures): while one fmadd is in flight, the other chain can
    // dispatch. With a single accumulator every FMA depends on the previous
    // one and the pipeline serializes.
    __m256 acc0 = _mm256_setzero_ps();  // __m256 = one YMM register: 8 floats
    __m256 acc1 = _mm256_setzero_ps();

    std::size_t i = 0;

    // Main body: 16 floats per iteration (two 8-wide registers).
    // _mm256_loadu_ps is an unaligned load: it does not require 32-byte
    // alignment. On modern CPUs the penalty vs. aligned loads is ~zero unless
    // the load splits a cache line; aligning the index storage to 32/64 bytes
    // is a future optimization.
    for (; i + 16 <= n; i += 16) {
        // fmadd(x, y, acc) = acc + x*y in ONE instruction with ONE rounding
        // step (the intermediate product is not rounded to float). This is
        // why results differ from the scalar version in the last bits — tests
        // must compare with a tolerance, not exact equality.
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa + i),     _mm256_loadu_ps(pb + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa + i + 8), _mm256_loadu_ps(pb + i + 8), acc1);
    }

    // One leftover 8-wide block, if any.
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa + i), _mm256_loadu_ps(pb + i), acc0);
    }

    // Horizontal reduction: collapse 8 lanes into one scalar.
    // YMM = [high 128-bit lane | low 128-bit lane] → sum halves successively.
    const __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc),      // low 4 floats
                          _mm256_extractf128_ps(acc, 1));   // high 4 floats
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));                 // [0+2, 1+3, ...]
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));          // [0+2+1+3, ...]
    float total = _mm_cvtss_f32(s);

    // Scalar tail: the last n % 8 elements.
    for (; i < n; ++i) {
        total += pa[i] * pb[i];
    }
    return total;
}

}  // namespace vecdb

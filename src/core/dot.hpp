/**
 * @file dot.hpp
 * @brief Dot product kernels: scalar reference, AVX2 SIMD, and runtime dispatch.
 *
 * The AVX2 kernel lives in its own translation unit (dot_avx2.cpp), which is
 * the only one compiled with -mavx2 -mfma. This keeps AVX2 instructions out
 * of the rest of the binary so the runtime dispatcher can safely fall back to
 * the scalar path on older CPUs.
 */

// [C#→C++] A header is textually pasted into every .cpp that #includes it —
// there are no assemblies or metadata. The declaration/definition split is
// manual. #pragma once prevents double inclusion within one translation unit.
#pragma once

#include <cstddef>
#include <span>

namespace vecdb {

// [C#→C++] std::span<const float> ≈ ReadOnlySpan<float>: pointer + length,
// no ownership, no copy. Unlike C#, there is no GC keeping the underlying
// buffer alive — the caller must guarantee the memory outlives the span,
// otherwise it dangles (undefined behavior).

/**
 * @brief Scalar reference implementation of the dot product.
 *
 * Simple sequential accumulation. Serves as the ground truth for tests and
 * as the fallback on CPUs without AVX2/FMA support.
 *
 * @param a First vector.
 * @param b Second vector.
 * @return Sum of element-wise products, accumulated in float.
 * @pre a.size() == b.size()
 */
float dot_scalar(std::span<const float> a, std::span<const float> b);

/**
 * @brief AVX2 + FMA vectorized dot product (8 floats per instruction).
 *
 * Processes 16 floats per main-loop iteration using two independent
 * accumulators, then reduces horizontally and handles the tail scalarly.
 *
 * @param a First vector.
 * @param b Second vector.
 * @return Sum of element-wise products.
 * @pre a.size() == b.size()
 * @warning Calling this on a CPU without AVX2/FMA raises an illegal
 *          instruction fault. Use dot() for safe runtime dispatch.
 * @note Results differ from dot_scalar() in the last bits: float addition is
 *       not associative (different summation order) and FMA rounds once per
 *       multiply-add instead of twice. Compare with a tolerance, never ==.
 */
float dot_avx2(std::span<const float> a, std::span<const float> b);

/**
 * @brief Dot product with runtime CPU dispatch.
 *
 * Selects dot_avx2() when the CPU reports AVX2 and FMA support (checked once
 * via CPUID at startup), otherwise dot_scalar().
 *
 * @param a First vector.
 * @param b Second vector.
 * @return Sum of element-wise products.
 * @pre a.size() == b.size()
 */
float dot(std::span<const float> a, std::span<const float> b);

}  // namespace vecdb

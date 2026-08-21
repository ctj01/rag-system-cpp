/**
 * @file dot_dispatch.cpp
 * @brief Runtime CPU-feature dispatch for the dot product.
 *
 * Queries CPUID once at startup and routes dot() to the AVX2 kernel when the
 * CPU supports both AVX2 and FMA, otherwise to the scalar reference.
 */
#include "core/dot.hpp"

namespace vecdb {
namespace {
// [C#→C++] An anonymous namespace ≈ C# internal: everything inside has
// internal linkage, invisible outside this translation unit.

/// @brief Returns true if the CPU supports both AVX2 and FMA.
/// Both are required because dot_avx2() uses fused multiply-add; rare CPUs
/// expose AVX2 without FMA.
bool detect_avx2_fma() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;  // unknown compiler: scalar fallback is always correct
#endif
}

// [C#→C++] A global initialized exactly once before main() (static
// initialization) — the role of a static readonly field in C#. CPUID is
// queried once, never in the hot path.
const bool g_use_avx2 = detect_avx2_fma();

}  // namespace

float dot(std::span<const float> a, std::span<const float> b) {
    return g_use_avx2 ? dot_avx2(a, b) : dot_scalar(a, b);
}

}  // namespace vecdb

/**
 * @file test_dot.cpp
 * @brief Verifies dot_avx2 ≡ dot_scalar on random vectors, and that both stay
 *        close to a double-precision reference.
 *
 * @details
 * SIMD and scalar results are NOT bit-identical, by design:
 *  1. Summation order differs: the scalar version accumulates sequentially;
 *     the SIMD version keeps 16 partial sums collapsed at the end. Float
 *     addition is not associative: (a+b)+c != a+(b+c) in the last bits.
 *  2. FMA rounds once per multiply-add (the intermediate product is exact);
 *     the scalar path rounds the product AND the sum.
 * Hence the tests compare with a relative tolerance, never with ==. The SIMD
 * result is typically MORE accurate (fewer roundings, partial sums).
 */
#include "core/dot.hpp"

#include <cstddef>
#include <random>
#include <vector>

#include "test_framework.hpp"

namespace {

/// Double-precision reference: accumulating in double makes the reference's
/// own error negligible next to that of the float versions under test.
double dot_ref(const std::vector<float>& a, const std::vector<float>& b) {
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return acc;
}

}  // namespace

int main() {
    // [C#→C++] mt19937 with a fixed seed == new Random(42): reproducible.
    // <random> separates the generator (bits) from the distribution (mapping
    // to a range), unlike .NET's monolithic Random.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Sizes chosen to exercise every path of the AVX2 kernel:
    //  - 0, 1..7      → scalar tail only
    //  - 8, 15        → one 8-wide block + tail
    //  - 16, 17, 31   → 16-wide body (+ 8-wide block, + tail)
    //  - 384/768/1536 → real embedding dimensions (MiniLM, BERT, ada)
    //  - 1000         → large and not a multiple of 8
    const std::size_t sizes[] = {0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32,
                                 100, 384, 768, 1000, 1536};

    for (const std::size_t n : sizes) {
        for (int rep = 0; rep < 20; ++rep) {
            std::vector<float> a(n), b(n);
            for (auto& x : a) x = dist(rng);
            for (auto& x : b) x = dist(rng);

            const double ref = dot_ref(a, b);
            const float s = vecdb::dot_scalar(a, b);
            const float v = vecdb::dot_avx2(a, b);
            const float d = vecdb::dot(a, b);

            // Tolerance: the error of an n-term float sum scales roughly as
            // O(n·eps·|magnitude|); 1e-4 relative is generous for n <= 1536
            // and still catches real bugs (a dropped lane or a miscounted
            // tail shows up as a huge relative error).
            CHECK(approx_eq(s, ref, 1e-4, 1e-5));
            CHECK(approx_eq(v, ref, 1e-4, 1e-5));
            CHECK(approx_eq(v, s, 1e-4, 1e-5));
            CHECK(d == v || d == s);  // dispatch returns one of the two
        }
    }

    // Directed cases: exactly orthogonal and exactly parallel vectors.
    {
        std::vector<float> e1(16, 0.0f), e2(16, 0.0f);
        e1[0] = 1.0f;
        e2[1] = 1.0f;
        CHECK(vecdb::dot_avx2(e1, e2) == 0.0f);
        CHECK(vecdb::dot_avx2(e1, e1) == 1.0f);
        CHECK(vecdb::dot_scalar(e1, e2) == 0.0f);
    }

    return test_summary("test_dot");
}

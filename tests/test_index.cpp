/**
 * @file test_index.cpp
 * @brief VectorIndex sanity tests: normalization, top-k vs. a naive
 *        independent computation, and input validation.
 */
#include "core/vector_index.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/dot.hpp"
#include "test_framework.hpp"

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const std::size_t dim = 384;
    const std::size_t n = 200;

    vecdb::VectorIndex index(dim);

    std::vector<std::vector<float>> raw;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        raw.push_back(v);
        const auto id = index.add(v);
        CHECK(id == i);
    }
    CHECK(index.size() == n);

    // Stored vectors must be unit-norm.
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto v = index.vector(i);
        const float norm = std::sqrt(vecdb::dot_scalar(v, v));
        CHECK(approx_eq(norm, 1.0, 1e-4, 0.0));
    }

    // Searching with a vector ALREADY in the index must return it first with
    // score ~1 (cosine with itself).
    {
        const auto results = index.search(raw[17], 5);
        CHECK(results.size() == 5);
        CHECK(results[0].id == 17);
        CHECK(approx_eq(results[0].score, 1.0, 1e-4, 0.0));
        // Descending order.
        for (std::size_t j = 1; j < results.size(); ++j) {
            CHECK(results[j - 1].score >= results[j].score);
        }
    }

    // Index top-1 must match a naive cosine computed outside the index.
    {
        std::vector<float> q(dim);
        for (auto& x : q) x = dist(rng);

        const auto results = index.search(q, 10);
        CHECK(results.size() == 10);

        float best = -2.0f;
        std::uint32_t best_id = 0;
        const float qn = std::sqrt(vecdb::dot_scalar(q, q));
        for (std::uint32_t i = 0; i < n; ++i) {
            const float vn = std::sqrt(vecdb::dot_scalar(raw[i], raw[i]));
            const float cos = vecdb::dot_scalar(q, raw[i]) / (qn * vn);
            if (cos > best) {
                best = cos;
                best_id = i;
            }
        }
        CHECK(results[0].id == best_id);
        CHECK(approx_eq(results[0].score, best, 1e-3, 1e-5));
    }

    // k > size() is clamped; k == 0 returns empty.
    CHECK(index.search(raw[0], 10'000).size() == n);
    CHECK(index.search(raw[0], 0).empty());

    // Input validation.
    CHECK_THROWS(vecdb::VectorIndex(0), std::invalid_argument);
    CHECK_THROWS(index.add(std::vector<float>(dim + 1, 1.0f)), std::invalid_argument);
    CHECK_THROWS(index.add(std::vector<float>(dim, 0.0f)), std::invalid_argument);
    CHECK_THROWS(index.search(std::vector<float>(dim - 1, 1.0f), 3), std::invalid_argument);
    CHECK_THROWS(index.vector(static_cast<std::uint32_t>(n)), std::out_of_range);

    return test_summary("test_index");
}

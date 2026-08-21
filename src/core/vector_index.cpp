/**
 * @file vector_index.cpp
 * @brief VectorIndex implementation: normalized insertion and brute-force top-k.
 */
#include "core/vector_index.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/dot.hpp"

namespace vecdb {

namespace {
/// Below this L2 norm a vector is considered to have no usable direction.
constexpr float kMinNorm = 1e-6f;
}

VectorIndex::VectorIndex(std::size_t dim) : dim_(dim) {
    if (dim == 0) {
        throw std::invalid_argument("VectorIndex: dim must be > 0");
    }
}

std::uint32_t VectorIndex::add(std::span<const float> vec) {
    if (vec.size() != dim_) {
        throw std::invalid_argument("VectorIndex::add: dimension mismatch");
    }

    const float norm = std::sqrt(dot(vec, vec));
    if (!(norm > kMinNorm)) {  // also catches NaN (any comparison with NaN is false)
        throw std::invalid_argument("VectorIndex::add: zero or invalid norm");
    }

    // Normalize ON INSERT, once, so that at query time cosine(q, v) is a bare
    // dot product — the hot path pays no divisions.
    const float inv = 1.0f / norm;

    // Growth is left to push_back's amortized doubling (same policy as
    // List<T>). Do NOT "help" with reserve(size + dim) here: an explicit
    // reserve allocates EXACTLY the requested capacity, defeating geometric
    // growth — every insert then reallocates and copies the whole buffer,
    // turning n inserts into O(n^2) bytes moved. Measured: ~15 TB of memcpy
    // for 100k adds at dim 768. Reallocation is also why spans returned by
    // vector() are invalidated after an add().
    for (const float x : vec) {
        data_.push_back(x * inv);
    }

    return static_cast<std::uint32_t>(count_++);
}

std::span<const float> VectorIndex::vector(std::uint32_t id) const {
    if (id >= count_) {
        throw std::out_of_range("VectorIndex::vector: bad id");
    }
    // [C#→C++] Constructing a span is just {pointer, length} — zero copy,
    // like new ReadOnlySpan<float>(ptr, len). subspan == Slice().
    return std::span<const float>(data_).subspan(id * dim_, dim_);
}

std::vector<SearchResult> VectorIndex::search(std::span<const float> query,
                                              std::size_t k) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("VectorIndex::search: dimension mismatch");
    }
    k = std::min(k, count_);
    if (k == 0) {
        return {};
    }

    // Normalize the query into a local buffer (stored vectors are already
    // unit-norm; the query arrives raw).
    const float qnorm = std::sqrt(dot(query, query));
    if (!(qnorm > kMinNorm)) {
        throw std::invalid_argument("VectorIndex::search: zero or invalid query norm");
    }
    std::vector<float> q(query.begin(), query.end());
    const float inv = 1.0f / qnorm;
    for (float& x : q) {
        x *= inv;
    }

    // Linear scan: one dot per stored vector, reading data_ sequentially.
    // This loop is exactly what HNSW will replace (O(n·d) → ~O(log n · d)).
    std::vector<SearchResult> results;
    results.reserve(count_);
    const float* base = data_.data();
    for (std::size_t i = 0; i < count_; ++i) {
        const float score = dot(q, std::span<const float>(base + i * dim_, dim_));
        results.push_back({static_cast<std::uint32_t>(i), score});
    }

    // Top-k selection: nth_element is average O(n) quickselect — it puts the
    // k best in the prefix without sorting the rest; we then sort only those
    // k. For large n and small k this beats a full sort by a wide margin.
    const auto by_score_desc = [](const SearchResult& x, const SearchResult& y) {
        return x.score > y.score;
    };
    std::nth_element(results.begin(), results.begin() + k, results.end(), by_score_desc);
    results.resize(k);  // drop the tail; adjusts size without reallocating
    std::sort(results.begin(), results.end(), by_score_desc);

    // [C#→C++] Returning a std::vector by value does NOT copy: move semantics
    // (or RVO — the compiler constructs the result directly in the caller's
    // slot) apply. A move steals the internal buffer pointer in O(1). This is
    // why modern C++ returns containers by value guilt-free, where in C# you
    // would return a reference to the heap object.
    return results;
}

}  // namespace vecdb

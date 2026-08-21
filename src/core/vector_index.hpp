/**
 * @file vector_index.hpp
 * @brief Brute-force vector index with cosine similarity over contiguous storage.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vecdb {

/**
 * @brief One search hit: vector id and its cosine similarity to the query.
 */
struct SearchResult {
    std::uint32_t id;   ///< Insertion index (0-based).
    float score;        ///< Cosine similarity in [-1, 1] (vectors are unit-norm).
};

/**
 * @brief Brute-force cosine-similarity index.
 *
 * @details
 * Core memory decision: ALL vectors live in ONE contiguous buffer, row-major:
 *
 *   data_ = [ v0[0..dim) | v1[0..dim) | v2[0..dim) | ... ]
 *
 * The search scan therefore reads physically sequential memory, letting the
 * CPU prefetcher stream cache lines ahead of the loop, and letting the SIMD
 * kernel load 8 adjacent floats per instruction. Brute-force search is memory
 * bandwidth bound, so the layout IS the optimization — same reason numpy uses
 * a single C buffer instead of nested lists.
 *
 * Vectors are L2-normalized once at insertion, so cosine similarity at query
 * time reduces to a plain dot product (no divisions or square roots in the
 * hot path).
 */
// [C#→C++] In C#, List<float[]> would be the natural shape — but each float[]
// is a separate heap object at an arbitrary address, so scanning becomes
// pointer chasing where every hop risks a cache miss (~100ns vs ~1ns L1).
// A single std::vector<float> guarantees physical contiguity.
//
// [C#→C++] std::vector<float> ≈ List<float>: dynamic array with amortized
// growth. Key differences: (1) it is a value type — copying it copies ALL the
// data, not a reference; (2) its memory is freed deterministically by the
// destructor (RAII), not when a GC decides; (3) operator[] does no bounds
// check in Release.
class VectorIndex {
public:
    /**
     * @brief Constructs an empty index for vectors of dimension @p dim.
     * @param dim Vector dimensionality; must be > 0.
     * @throws std::invalid_argument if dim == 0.
     */
    // [C#→C++] explicit forbids the implicit size_t → VectorIndex conversion.
    // In C++ single-argument constructors act as implicit conversions unless
    // marked explicit (C# constructors never do). Rule of thumb: explicit on
    // every single-argument constructor.
    explicit VectorIndex(std::size_t dim);

    /**
     * @brief Copies @p vec into internal storage, L2-normalized.
     * @param vec Input vector; must have size() == dim().
     * @return The id assigned to the vector (sequential, 0-based).
     * @throws std::invalid_argument on dimension mismatch or if the vector
     *         has (near-)zero or non-finite norm — such vectors have no
     *         defined direction.
     */
    std::uint32_t add(std::span<const float> vec);

    /**
     * @brief Returns the top-k most similar vectors to @p query.
     * @param query Raw (un-normalized) query vector; must have size() == dim().
     * @param k Number of results requested; clamped to size().
     * @return Results sorted by descending cosine similarity.
     * @throws std::invalid_argument on dimension mismatch or zero-norm query.
     * @details Linear O(n·dim) scan — exactly the loop HNSW will later
     *          replace with ~O(log n · dim).
     */
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k) const;

    /// @brief Number of vectors stored.
    // [C#→C++] Trailing const = "this method does not mutate the object",
    // compiler-enforced (no C# equivalent). noexcept = promise not to throw;
    // enables optimizations and doubles as executable documentation.
    std::size_t size() const noexcept { return count_; }

    /// @brief Vector dimensionality this index was built for.
    std::size_t dim() const noexcept { return dim_; }

    /**
     * @brief Read-only view of the stored (normalized) vector @p id. No copy.
     * @param id Vector id returned by add().
     * @return Span over the vector's dim() floats inside internal storage.
     * @throws std::out_of_range if id >= size().
     * @warning The span is invalidated by any subsequent add(): growth may
     *          reallocate the buffer — same hazard as a Span<T> over a
     *          List<T> that grows.
     */
    std::span<const float> vector(std::uint32_t id) const;

private:
    std::size_t dim_;
    std::size_t count_ = 0;      ///< Number of vectors (== data_.size() / dim_).
    std::vector<float> data_;    ///< Contiguous row-major storage.

    // [C#→C++] No Dispose, no finalizer: when a VectorIndex goes out of
    // scope, its (compiler-generated) destructor destroys data_, which frees
    // its heap. That is RAII — resource lifetime tied to object lifetime,
    // deterministic, no GC. C#'s "using" approximates manually what is the
    // default behavior here.
};

}  // namespace vecdb

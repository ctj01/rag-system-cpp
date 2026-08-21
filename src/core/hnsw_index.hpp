/**
 * @file hnsw_index.hpp
 * @brief HNSW (Hierarchical Navigable Small World) approximate nearest
 *        neighbor index over cosine similarity.
 *
 * Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
 * neighbor search using Hierarchical Navigable Small World graphs" (2018).
 *
 * Design decisions (see project README / design notes):
 *  - Flat adjacency storage: layer-0 links live in one contiguous array with
 *    fixed-capacity slots per node; upper-layer links (only ~1/M of nodes
 *    have any) live in a separate flat block, allocated once per node at
 *    insertion (a node's level never changes).
 *  - Neighbor selection uses the diversity heuristic (Algorithm 4) by
 *    default, with keep-pruned refill; the naive "M closest" selection is
 *    available behind a flag for recall comparison benchmarks.
 *  - Reverse-edge overflow is re-pruned with the same heuristic, never by
 *    dropping the farthest neighbor.
 *  - Visited tracking uses epoch-stamped arrays from a pool (no hashing or
 *    per-query allocation in the hot path); the pool makes read-only
 *    searches safe to run concurrently on a frozen graph.
 *  - Concurrency phase 1: single-threaded build, then freeze; searches are
 *    const and thread-safe against each other (NOT against add()).
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <vector>

#include "core/vector_index.hpp"  // SearchResult

namespace vecdb {

/**
 * @brief Tunables for HnswIndex.
 *
 * @details Trade-offs:
 *  - M: graph degree (layers >= 1); layer 0 caps at 2*M. Higher M = better
 *    recall and more memory; structural, cannot change after build.
 *  - ef_construction: beam width during insertion. Buys graph quality once,
 *    at build time; diminishing returns past ~200.
 *  - The search-time beam (ef_search) is a search() parameter, not stored
 *    config — it is the runtime recall/latency knob.
 */
// [C#→C++] Default member initializers work like C# auto-property
// initializers; brace-init `HnswConfig{.M = 32}` (designated initializers,
// C++20) is the closest thing to named arguments.
struct HnswConfig {
    std::size_t M = 16;                ///< Max degree per node, layers >= 1.
    std::size_t ef_construction = 200; ///< Beam width during insertion.
    std::uint64_t seed = 42;           ///< RNG seed for level assignment (reproducible builds).
    bool use_heuristic = true;         ///< Diversity heuristic vs naive M-closest selection.
    bool keep_pruned = true;           ///< Refill up to M with pruned candidates.
};

/**
 * @brief Approximate nearest-neighbor index; drop-in alternative to the
 *        brute-force VectorIndex for large n.
 *
 * Vectors are L2-normalized on insertion (cosine == dot). Internally
 * distances are d = 1 - dot, so "smaller is closer" like the paper's
 * pseudocode; scores returned to callers are cosine similarities.
 */
class HnswIndex {
public:
    /**
     * @brief Constructs an empty index.
     * @param dim Vector dimensionality; must be > 0.
     * @param config Tunables; defaults are sane for text embeddings.
     * @throws std::invalid_argument if dim == 0 or config.M < 2.
     */
    explicit HnswIndex(std::size_t dim, HnswConfig config = {});

    /**
     * @brief Inserts a vector (copied, L2-normalized). Single-threaded.
     * @return The id assigned (insertion order, 0-based).
     * @throws std::invalid_argument on dimension mismatch or zero norm.
     */
    std::uint32_t add(std::span<const float> vec);

    /**
     * @brief Approximate top-k by cosine similarity.
     * @param query Raw (un-normalized) query vector.
     * @param k Number of results; clamped to size().
     * @param ef_search Beam width at layer 0; clamped up to k. Higher =
     *        better recall, ~linearly more latency. 50-200 typical.
     * @return Results sorted by descending score. Approximate: may miss
     *         true neighbors (measure recall against VectorIndex).
     * @note Thread-safe against other search() calls once the build is
     *       finished (frozen graph); NOT safe concurrently with add().
     */
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                     std::size_t ef_search = 100) const;

    std::size_t size() const noexcept { return count_; }
    std::size_t dim() const noexcept { return dim_; }
    int max_level() const noexcept { return max_level_; }

private:
    /// One entry in the search beams: distance (1 - dot) plus node id.
    struct Candidate {
        float d;
        std::uint32_t id;
    };

    /**
     * @brief Pool of epoch-stamped visited lists.
     *
     * Marking a node visited is one store; checking is one compare;
     * "clearing" between queries is ++epoch (O(1), no memset). The pool
     * hands each concurrent search its own list.
     */
    class VisitedPool {
    public:
        struct List {
            std::vector<std::uint32_t> stamp;
            std::uint32_t epoch = 0;

            bool test_and_set(std::uint32_t id) {
                if (stamp[id] == epoch) return true;
                stamp[id] = epoch;
                return false;
            }
        };

        // [C#→C++] Move-only RAII lease (cf. SafeHandle + using in C#): the
        // destructor returns the list to the pool automatically, even on
        // exceptions. Copying is implicitly disabled because unique_ptr is
        // move-only — the compiler enforces single ownership at compile time.
        class Lease {
        public:
            Lease(VisitedPool* pool, std::unique_ptr<List> list)
                : pool_(pool), list_(std::move(list)) {}
            ~Lease() {
                if (list_) pool_->release(std::move(list_));
            }
            Lease(Lease&&) = default;
            Lease& operator=(Lease&&) = delete;

            List& operator*() { return *list_; }

        private:
            VisitedPool* pool_;
            std::unique_ptr<List> list_;
        };

        /// Returns a list sized for n nodes with a fresh epoch.
        Lease acquire(std::size_t n);

    private:
        void release(std::unique_ptr<List> list);

        std::mutex mu_;
        std::vector<std::unique_ptr<List>> free_;
    };

    std::span<const float> vec(std::uint32_t id) const {
        return std::span<const float>(data_.data() + static_cast<std::size_t>(id) * dim_, dim_);
    }
    float dist_to(std::span<const float> q, std::uint32_t id) const;

    /// Adjacency slot for @p id at @p level: pointer to [count, id0, id1...].
    std::uint32_t* links(std::uint32_t id, int level);
    const std::uint32_t* links(std::uint32_t id, int level) const;
    std::size_t max_degree(int level) const {
        return level == 0 ? mmax0_ : cfg_.M;
    }

    int random_level();

    /// Greedy ef=1 descent within one layer; returns the closest node found.
    std::uint32_t greedy_closest(std::span<const float> q, std::uint32_t ep, int level) const;

    /// Beam search within one layer; returns candidates sorted ascending by d.
    std::vector<Candidate> search_layer(std::span<const float> q, std::uint32_t ep,
                                        std::size_t ef, int level) const;

    /// Selects up to m neighbors from candidates (sorted ascending) for a
    /// node whose vector is @p base — heuristic or simple per config.
    std::vector<Candidate> select_neighbors(std::span<const float> base,
                                            const std::vector<Candidate>& candidates,
                                            std::size_t m) const;

    /// Adds edge node->cand; if the slot is full, re-prunes with the heuristic.
    void add_link(std::uint32_t node, std::uint32_t cand, int level);

    std::size_t dim_;
    HnswConfig cfg_;
    std::size_t mmax0_;   ///< Layer-0 degree cap: 2*M.
    double ml_;           ///< Level normalization: 1/ln(M).

    std::size_t count_ = 0;
    std::vector<float> data_;            ///< Contiguous row-major vectors (unit norm).
    std::vector<int> levels_;            ///< Max level per node.

    // Layer 0: fixed slots of (1 + mmax0_) uint32 per node: [count, ids...].
    std::vector<std::uint32_t> links0_;
    // Layers >= 1: per node with level L >= 1, a block of L*(1+M) uint32,
    // appended once at insertion. upper_offset_[id] indexes into upper_
    // (npos when the node lives only in layer 0).
    static constexpr std::size_t kNoUpper = static_cast<std::size_t>(-1);
    std::vector<std::size_t> upper_offset_;
    std::vector<std::uint32_t> upper_;

    std::uint32_t entry_ = 0;
    int max_level_ = -1;  ///< -1 while empty.
    std::mt19937_64 rng_;

    // [C#→C++] mutable = "this member may change inside const methods".
    // search() is logically read-only (the graph doesn't change) but needs
    // scratch space; mutable is the idiom for caches/scratch that don't
    // affect observable state. No C# equivalent — const-correctness itself
    // has no C# counterpart.
    mutable VisitedPool visited_pool_;
};

}  // namespace vecdb

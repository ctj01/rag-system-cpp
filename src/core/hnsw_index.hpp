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
 *    have any) live in a separate flat block.
 *  - Neighbor selection uses the diversity heuristic (Algorithm 4) by
 *    default, with keep-pruned refill; naive "M closest" behind a flag.
 *  - Reverse-edge overflow is re-pruned with the same heuristic.
 *  - Visited tracking uses epoch-stamped arrays from a pool.
 *
 * Concurrency model:
 *  - add(): single writer.
 *  - add_batch(): parallel build. All storage is pre-allocated up front (no
 *    reallocation while threads run), levels are a pure function of
 *    (seed, id) so no shared RNG, adjacency lists are protected by striped
 *    per-node mutexes, and the entry point sits behind its own mutex taken
 *    only when an insert introduces a new top layer (rare: ~1/M^level).
 *  - search(): lock-free; safe from any number of threads once the build is
 *    finished (frozen graph). NOT safe concurrently with add()/add_batch().
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "core/vector_index.hpp"  // SearchResult

namespace vecdb {

/**
 * @brief Tunables for HnswIndex.
 */
struct HnswConfig {
    std::size_t M = 16;                ///< Max degree per node, layers >= 1.
    std::size_t ef_construction = 200; ///< Beam width during insertion.
    std::uint64_t seed = 42;           ///< Level-assignment seed (levels are f(seed, id)).
    bool use_heuristic = true;         ///< Diversity heuristic vs naive M-closest selection.
    bool keep_pruned = true;           ///< Refill up to M with pruned candidates.
};

/**
 * @brief Approximate nearest-neighbor index; drop-in alternative to the
 *        brute-force VectorIndex for large n.
 *
 * Vectors are L2-normalized on insertion (cosine == dot). Internally
 * distances are d = 1 - dot ("smaller is closer"); scores returned to
 * callers are cosine similarities.
 */
class HnswIndex {
public:
    explicit HnswIndex(std::size_t dim, HnswConfig config = {});

    /**
     * @brief Inserts one vector (copied, L2-normalized). Single-threaded.
     * @return The id assigned (insertion order, 0-based).
     * @throws std::invalid_argument on dimension mismatch or zero norm.
     */
    std::uint32_t add(std::span<const float> vec);

    /**
     * @brief Inserts a batch of vectors, building the graph on @p n_threads.
     *
     * Validates every vector first (strong guarantee: on throw, the index is
     * unchanged), pre-allocates all storage, then links nodes in parallel.
     * Ids are assigned contiguously starting at the current size().
     *
     * @param vectors Batch to insert.
     * @param n_threads 0 = hardware concurrency.
     * @return Number of vectors inserted.
     * @throws std::invalid_argument on any dimension mismatch or zero norm.
     * @note The resulting graph depends on thread interleaving (documented
     *       trade-off); node LEVELS are still deterministic per (seed, id).
     */
    std::size_t add_batch(std::span<const std::vector<float>> vectors, int n_threads = 0);

    /**
     * @brief Approximate top-k by cosine similarity.
     * @param query Raw (un-normalized) query vector.
     * @param k Number of results; clamped to size().
     * @param ef_search Beam width at layer 0; clamped up to k.
     * @return Results sorted by descending score.
     * @note Lock-free and thread-safe against other search() calls on a
     *       frozen graph; NOT safe concurrently with add()/add_batch().
     */
    std::vector<SearchResult> search(std::span<const float> query, std::size_t k,
                                     std::size_t ef_search = 100) const;

    std::size_t size() const noexcept { return count_; }
    std::size_t dim() const noexcept { return dim_; }
    int max_level() const noexcept { return max_level_; }

    /**
     * @brief Serializes the full index (config, vectors, graph) to a binary
     *        file. The format is little-endian/x86-64 native (documented
     *        non-portability; this is a local cache, not an exchange format).
     * @throws std::runtime_error on I/O failure.
     */
    void save(const std::string& path) const;

    /**
     * @brief Loads an index written by save(). Full graph state is restored:
     *        searches return bit-identical results to the saved instance.
     * @throws std::runtime_error on I/O failure, bad magic, or version
     *         mismatch.
     */
    // [C#→C++] Returning HnswIndex BY VALUE despite it containing
    // std::mutex members (which can be neither copied nor moved) works
    // because of C++17 guaranteed copy elision: `return HnswIndex(...)` is a
    // prvalue, constructed directly in the caller's storage — no move ever
    // happens. Naming the local first would not compile.
    static HnswIndex load(const std::string& path);

private:
    /// Deserializing constructor: header already validated/consumed by load().
    HnswIndex(std::size_t dim, HnswConfig config, std::istream& in);
    /// One entry in the search beams: distance (1 - dot) plus node id.
    struct Candidate {
        float d;
        std::uint32_t id;
    };

    /**
     * @brief Pool of epoch-stamped visited lists (one per in-flight search).
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

        // [C#→C++] Move-only RAII lease (cf. SafeHandle + using): the
        // destructor returns the list to the pool automatically, even on
        // exceptions. Copying is a compile error (unique_ptr is move-only).
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

    std::uint32_t* links(std::uint32_t id, int level);
    const std::uint32_t* links(std::uint32_t id, int level) const;
    std::size_t max_degree(int level) const {
        return level == 0 ? mmax0_ : cfg_.M;
    }

    /// Level as a pure function of (seed, id): no shared RNG state, so
    /// parallel builders need no synchronization, and levels are
    /// reproducible regardless of insertion interleaving.
    int level_for(std::uint32_t id) const;

    std::mutex& stripe(std::uint32_t id) const {
        return link_locks_[id % kStripes];
    }

    std::uint32_t greedy_closest(std::span<const float> q, std::uint32_t ep, int level,
                                 bool locked) const;
    std::vector<Candidate> search_layer(std::span<const float> q, std::uint32_t ep,
                                        std::size_t ef, int level, bool locked) const;
    std::vector<Candidate> select_neighbors(std::span<const float> base,
                                            const std::vector<Candidate>& candidates,
                                            std::size_t m) const;
    void add_link(std::uint32_t node, std::uint32_t cand, int level, bool locked);

    /// Wires node @p id into the graph (both edge directions, all layers).
    /// Storage for the node must already exist. locked = parallel build.
    void insert_node(std::uint32_t id, bool locked);

    /// Appends storage (vector data, level, zeroed link slots) for one node.
    std::uint32_t append_node_storage(std::span<const float> v, float inv_norm);

    std::size_t dim_;
    HnswConfig cfg_;
    std::size_t mmax0_;   ///< Layer-0 degree cap: 2*M.
    double ml_;           ///< Level normalization: 1/ln(M).

    std::size_t count_ = 0;
    std::vector<float> data_;            ///< Contiguous row-major vectors (unit norm).
    std::vector<int> levels_;            ///< Max level per node.

    // Layer 0: fixed slots of (1 + mmax0_) uint32 per node: [count, ids...].
    std::vector<std::uint32_t> links0_;
    // Layers >= 1: per node with level L >= 1, a block of L*(1+M) uint32.
    static constexpr std::size_t kNoUpper = static_cast<std::size_t>(-1);
    std::vector<std::size_t> upper_offset_;
    std::vector<std::uint32_t> upper_;

    std::uint32_t entry_ = 0;
    int max_level_ = -1;  ///< -1 while empty.

    // Striped adjacency locks: 64 mutexes shared across all nodes (id % 64).
    // Full per-node mutexes would cost 40 bytes each; striping trades a
    // little false contention for constant memory. Only the build path locks
    // them — frozen-graph searches never touch these.
    static constexpr std::size_t kStripes = 64;
    mutable std::array<std::mutex, kStripes> link_locks_;
    std::mutex entry_mutex_;  ///< Guards entry_/max_level_ during parallel build.

    // [C#→C++] mutable: scratch that may change inside const methods without
    // affecting observable state (no C# equivalent; C# lacks const methods).
    mutable VisitedPool visited_pool_;
};

}  // namespace vecdb

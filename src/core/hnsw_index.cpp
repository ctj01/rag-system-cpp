/**
 * @file hnsw_index.cpp
 * @brief HNSW implementation: insertion with heuristic neighbor selection,
 *        layered beam search, reverse-edge re-pruning, parallel batch build.
 */
#include "core/hnsw_index.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <exception>
#include <queue>
#include <stdexcept>
#include <thread>
#include <xmmintrin.h>  // _mm_prefetch (SSE, baseline on x86-64)

#include "core/dot.hpp"

namespace vecdb {

namespace {
constexpr float kMinNorm = 1e-6f;

// [C#→C++] std::priority_queue comparators are TYPES resolved at compile
// time — the compare code inlines into the heap operations, no virtual calls.
struct CloserFirst {  // min-heap by distance (candidates to expand)
    bool operator()(const auto& a, const auto& b) const { return a.d > b.d; }
};
struct FartherFirst {  // max-heap by distance (bounded result set, worst on top)
    bool operator()(const auto& a, const auto& b) const { return a.d < b.d; }
};
}  // namespace

// ---------------------------------------------------------------- VisitedPool

HnswIndex::VisitedPool::Lease HnswIndex::VisitedPool::acquire(std::size_t n) {
    std::unique_ptr<List> list;
    {
        // [C#→C++] lock_guard = C#'s lock(obj){} as RAII: locks in the
        // constructor, unlocks in the destructor, exception-safe.
        std::lock_guard<std::mutex> lock(mu_);
        if (!free_.empty()) {
            list = std::move(free_.back());
            free_.pop_back();
        }
    }
    if (!list) list = std::make_unique<List>();
    list->stamp.resize(n, 0);
    ++list->epoch;
    if (list->epoch == 0) {  // uint32 wraparound: reset stamps once per ~4B queries
        std::fill(list->stamp.begin(), list->stamp.end(), 0);
        list->epoch = 1;
    }
    return Lease(this, std::move(list));
}

void HnswIndex::VisitedPool::release(std::unique_ptr<List> list) {
    std::lock_guard<std::mutex> lock(mu_);
    free_.push_back(std::move(list));
}

// ------------------------------------------------------------------ HnswIndex

HnswIndex::HnswIndex(std::size_t dim, HnswConfig config)
    : dim_(dim),
      cfg_(config),
      mmax0_(2 * config.M),
      ml_(1.0 / std::log(static_cast<double>(config.M))) {
    if (dim == 0) throw std::invalid_argument("HnswIndex: dim must be > 0");
    if (cfg_.M < 2) throw std::invalid_argument("HnswIndex: M must be >= 2");
}

float HnswIndex::dist_to(std::span<const float> q, std::uint32_t id) const {
    // Unit-norm vectors: cosine distance = 1 - dot (monotone in angle).
    return 1.0f - dot(q, vec(id));
}

std::uint32_t* HnswIndex::links(std::uint32_t id, int level) {
    if (level == 0) {
        return links0_.data() + static_cast<std::size_t>(id) * (1 + mmax0_);
    }
    assert(upper_offset_[id] != kNoUpper && level <= levels_[id]);
    return upper_.data() + upper_offset_[id] +
           static_cast<std::size_t>(level - 1) * (1 + cfg_.M);
}

const std::uint32_t* HnswIndex::links(std::uint32_t id, int level) const {
    // [C#→C++] const_cast to share the implementation between const and
    // non-const overloads — the classic idiom; the const overload never
    // writes through the pointer.
    return const_cast<HnswIndex*>(this)->links(id, level);
}

int HnswIndex::level_for(std::uint32_t id) const {
    // splitmix64 of (seed, id) -> u in (0,1] -> geometric level. Stateless:
    // any thread can compute any node's level with no synchronization, and
    // the value never depends on insertion order.
    std::uint64_t z = cfg_.seed + 0x9e3779b97f4a7c15ull * (static_cast<std::uint64_t>(id) + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    const double u = (static_cast<double>(z >> 11) + 1.0) * 0x1.0p-53;  // (0,1]
    return static_cast<int>(-std::log(u) * ml_);
}

// -------------------------------------------------------------------- search

std::uint32_t HnswIndex::greedy_closest(std::span<const float> q, std::uint32_t ep,
                                        int level, bool locked) const {
    std::vector<std::uint32_t> copy;
    float best = dist_to(q, ep);
    bool improved = true;
    while (improved) {
        improved = false;
        const std::uint32_t* ids;
        std::uint32_t n;
        if (locked) {
            // Copy the neighbor list under the node's stripe lock: a writer
            // may be re-pruning it concurrently during a parallel build.
            std::lock_guard<std::mutex> g(stripe(ep));
            const std::uint32_t* l = links(ep, level);
            n = l[0];
            copy.assign(l + 1, l + 1 + n);
            ids = copy.data();
        } else {
            const std::uint32_t* l = links(ep, level);
            n = l[0];
            ids = l + 1;
        }
        for (std::uint32_t j = 0; j < n; ++j) {
            const float d = dist_to(q, ids[j]);
            if (d < best) {
                best = d;
                ep = ids[j];
                improved = true;
            }
        }
    }
    return ep;
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer(std::span<const float> q,
                                                          std::uint32_t ep,
                                                          std::size_t ef, int level,
                                                          bool locked) const {
    auto lease = visited_pool_.acquire(count_);
    auto& visited = *lease;

    std::priority_queue<Candidate, std::vector<Candidate>, CloserFirst> frontier;
    std::priority_queue<Candidate, std::vector<Candidate>, FartherFirst> best;

    const float d0 = dist_to(q, ep);
    frontier.push({d0, ep});
    best.push({d0, ep});
    visited.test_and_set(ep);

    std::vector<std::uint32_t> copy;  // reused: one allocation per call, not per hop

    while (!frontier.empty()) {
        const Candidate c = frontier.top();
        // Stop: the closest unexpanded candidate is worse than the worst
        // kept result and the result set is full. (Exact on an ideal
        // navigable graph — HNSW's approximation lives exactly here.)
        if (c.d > best.top().d && best.size() >= ef) break;
        frontier.pop();

        const std::uint32_t* ids;
        std::uint32_t n;
        if (locked) {
            std::lock_guard<std::mutex> g(stripe(c.id));
            const std::uint32_t* l = links(c.id, level);
            n = l[0];
            copy.assign(l + 1, l + 1 + n);
            ids = copy.data();
        } else {
            const std::uint32_t* l = links(c.id, level);
            n = l[0];
            ids = l + 1;
        }

        for (std::uint32_t j = 0; j < n; ++j) {
            // Overlap the next neighbor's vector fetch with this distance.
            if (j + 1 < n) {
                _mm_prefetch(reinterpret_cast<const char*>(vec(ids[j + 1]).data()),
                             _MM_HINT_T0);
            }
            const std::uint32_t nb = ids[j];
            if (visited.test_and_set(nb)) continue;
            const float d = dist_to(q, nb);
            if (best.size() < ef || d < best.top().d) {
                frontier.push({d, nb});
                best.push({d, nb});
                if (best.size() > ef) best.pop();
            }
        }
    }

    std::vector<Candidate> out;
    out.reserve(best.size());
    while (!best.empty()) {
        out.push_back(best.top());
        best.pop();
    }
    std::reverse(out.begin(), out.end());  // ascending by distance
    return out;
}

std::vector<SearchResult> HnswIndex::search(std::span<const float> query, std::size_t k,
                                            std::size_t ef_search) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("HnswIndex::search: dimension mismatch");
    }
    if (count_ == 0 || k == 0) return {};
    k = std::min(k, count_);
    ef_search = std::max(ef_search, k);

    const float qnorm = std::sqrt(dot(query, query));
    if (!(qnorm > kMinNorm)) {
        throw std::invalid_argument("HnswIndex::search: zero or invalid query norm");
    }
    std::vector<float> q(query.begin(), query.end());
    const float inv = 1.0f / qnorm;
    for (float& x : q) x *= inv;

    std::uint32_t ep = entry_;
    for (int level = max_level_; level >= 1; --level) {
        ep = greedy_closest(q, ep, level, /*locked=*/false);
    }
    const auto found = search_layer(q, ep, ef_search, 0, /*locked=*/false);

    std::vector<SearchResult> results;
    results.reserve(std::min(k, found.size()));
    for (std::size_t i = 0; i < found.size() && i < k; ++i) {
        results.push_back({found[i].id, 1.0f - found[i].d});
    }
    return results;
}

// ----------------------------------------------------------------- insertion

std::vector<HnswIndex::Candidate> HnswIndex::select_neighbors(
    std::span<const float> base, const std::vector<Candidate>& candidates,
    std::size_t m) const {
    if (!cfg_.use_heuristic || candidates.size() <= m) {
        return {candidates.begin(),
                candidates.begin() + static_cast<std::ptrdiff_t>(
                                         std::min(m, candidates.size()))};
    }

    // Heuristic (Algorithm 4): accept c only if it is closer to base than to
    // every already-selected neighbor. The surviving set is directionally
    // diverse — it creates the long bridge edges that keep clusters
    // mutually reachable.
    std::vector<Candidate> selected;
    std::vector<Candidate> pruned;
    selected.reserve(m);
    for (const Candidate& c : candidates) {
        if (selected.size() >= m) break;
        bool keep = true;
        for (const Candidate& s : selected) {
            if (dist_to(vec(c.id), s.id) < c.d) {  // c sits "behind" s
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.push_back(c);
        } else if (cfg_.keep_pruned) {
            pruned.push_back(c);
        }
    }
    for (std::size_t i = 0; i < pruned.size() && selected.size() < m; ++i) {
        selected.push_back(pruned[i]);
    }
    return selected;
}

void HnswIndex::add_link(std::uint32_t node, std::uint32_t cand, int level, bool locked) {
    // [C#→C++] unique_lock vs lock_guard: unique_lock can be constructed
    // unlocked/deferred and conditionally engaged — the idiom for "lock only
    // on the concurrent path". Slightly heavier than lock_guard, fine off
    // the frozen-search hot path. Holding a single stripe at a time also
    // makes deadlock impossible (no lock ordering to violate).
    std::unique_lock<std::mutex> guard;
    if (locked) guard = std::unique_lock<std::mutex>(stripe(node));

    std::uint32_t* l = links(node, level);
    const std::size_t cap = max_degree(level);
    if (l[0] < cap) {
        l[++l[0]] = cand;
        return;
    }
    // Slot full: re-prune with the SAME heuristic used at selection time.
    // Dropping "the farthest" instead silently destroys the diverse edges.
    std::vector<Candidate> candidates;
    candidates.reserve(cap + 1);
    const auto base = vec(node);
    candidates.push_back({dist_to(base, cand), cand});
    for (std::uint32_t j = 1; j <= l[0]; ++j) {
        candidates.push_back({dist_to(base, l[j]), l[j]});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.d < b.d; });
    const auto selected = select_neighbors(base, candidates, cap);
    l[0] = static_cast<std::uint32_t>(selected.size());
    for (std::size_t j = 0; j < selected.size(); ++j) {
        l[1 + j] = selected[j].id;
    }
}

void HnswIndex::insert_node(std::uint32_t id, bool locked) {
    const int level = levels_[id];
    const auto q = vec(id);

    // Snapshot the entry point. Under parallel build another thread may
    // update it concurrently — a slightly stale snapshot only costs a few
    // extra greedy hops, never correctness.
    std::uint32_t ep;
    int top;
    {
        std::unique_lock<std::mutex> guard;
        if (locked) guard = std::unique_lock<std::mutex>(entry_mutex_);
        ep = entry_;
        top = max_level_;
    }

    for (int lc = top; lc > level; --lc) {
        ep = greedy_closest(q, ep, lc, locked);
    }

    for (int lc = std::min(level, top); lc >= 0; --lc) {
        const auto candidates = search_layer(q, ep, cfg_.ef_construction, lc, locked);
        const auto neighbors = select_neighbors(q, candidates, cfg_.M);

        {
            // Own outgoing edges first, under our own stripe: after the
            // first reverse edge below publishes this node, other builders
            // may traverse INTO it and even re-prune it concurrently.
            std::unique_lock<std::mutex> guard;
            if (locked) guard = std::unique_lock<std::mutex>(stripe(id));
            std::uint32_t* l = links(id, lc);
            l[0] = static_cast<std::uint32_t>(neighbors.size());
            for (std::size_t j = 0; j < neighbors.size(); ++j) {
                l[1 + j] = neighbors[j].id;
            }
        }
        // Publication ordering: outgoing edges are written BEFORE any
        // reverse edge makes the node reachable. The mutexes provide the
        // release/acquire ordering; a traversal can never see a node whose
        // adjacency at this layer is still unwritten. (It may see zeroed
        // lists on LOWER layers the builder hasn't reached yet — that reads
        // as a dead end, costing recall transiently, never correctness.)
        for (const Candidate& nb : neighbors) {
            add_link(nb.id, id, lc, locked);
        }

        ep = candidates.front().id;
    }

    if (level > top) {  // node introduces a new top layer (rare: ~1/M^level)
        std::unique_lock<std::mutex> guard;
        if (locked) guard = std::unique_lock<std::mutex>(entry_mutex_);
        if (level > max_level_) {  // recheck: another thread may have won
            entry_ = id;
            max_level_ = level;
        }
    }
}

std::uint32_t HnswIndex::append_node_storage(std::span<const float> v, float inv_norm) {
    const auto id = static_cast<std::uint32_t>(count_);
    const int level = level_for(id);
    for (const float x : v) data_.push_back(x * inv_norm);
    levels_.push_back(level);
    links0_.resize(links0_.size() + 1 + mmax0_, 0);
    if (level >= 1) {
        upper_offset_.push_back(upper_.size());
        upper_.resize(upper_.size() + static_cast<std::size_t>(level) * (1 + cfg_.M), 0);
    } else {
        upper_offset_.push_back(kNoUpper);
    }
    ++count_;
    return id;
}

std::uint32_t HnswIndex::add(std::span<const float> v) {
    if (v.size() != dim_) {
        throw std::invalid_argument("HnswIndex::add: dimension mismatch");
    }
    const float norm = std::sqrt(dot(v, v));
    if (!(norm > kMinNorm)) {
        throw std::invalid_argument("HnswIndex::add: zero or invalid norm");
    }

    const auto id = append_node_storage(v, 1.0f / norm);
    if (count_ == 1) {  // first node: entry point, no edges yet
        entry_ = id;
        max_level_ = levels_[id];
        return id;
    }
    insert_node(id, /*locked=*/false);
    return id;
}

std::size_t HnswIndex::add_batch(std::span<const std::vector<float>> vectors,
                                 int n_threads) {
    if (vectors.empty()) return 0;

    // Validate EVERYTHING before mutating anything: if this loop throws,
    // the index is untouched (strong exception guarantee).
    std::vector<float> inv_norms;
    inv_norms.reserve(vectors.size());
    for (const auto& v : vectors) {
        if (v.size() != dim_) {
            throw std::invalid_argument("HnswIndex::add_batch: dimension mismatch");
        }
        const float norm = std::sqrt(dot(v, v));
        if (!(norm > kMinNorm)) {
            throw std::invalid_argument("HnswIndex::add_batch: zero or invalid norm");
        }
        inv_norms.push_back(1.0f / norm);
    }

    // Pre-allocate ALL storage before any thread starts: the flat arrays
    // never reallocate during the parallel phase, so raw pointers taken by
    // concurrent builders stay valid for the whole build.
    const auto base = static_cast<std::uint32_t>(count_);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        append_node_storage(vectors[i], inv_norms[i]);
    }

    std::size_t first = 0;
    if (max_level_ == -1) {  // empty index: seed the graph with one node
        entry_ = base;
        max_level_ = levels_[base];
        first = 1;
    }
    const auto total = static_cast<std::uint32_t>(count_);
    if (base + first >= total) return vectors.size();

    // Sequential warmup: with a tiny graph, concurrent inserters cannot see
    // each other (a node becomes visible only after its reverse edges land),
    // so the earliest nodes end up permanently under-connected — measured as
    // a 2-5% self-retrieval loss concentrated in the first few hundred ids.
    // Building the initial scaffold single-threaded removes that phase; at
    // real corpus sizes its cost is noise.
    constexpr std::uint32_t kWarmup = 256;
    std::uint32_t start = base + static_cast<std::uint32_t>(first);
    const std::uint32_t warm_end = std::min<std::uint32_t>(total, start + kWarmup);
    for (; start < warm_end; ++start) {
        insert_node(start, /*locked=*/false);
    }
    if (start >= total) return vectors.size();

    unsigned t = n_threads > 0 ? static_cast<unsigned>(n_threads)
                               : std::max(1u, std::thread::hardware_concurrency());
    t = std::min<unsigned>(t, total - start);

    // Work dispenser: threads pull the next id with one atomic increment.
    std::atomic<std::uint32_t> next{start};

    // [C#→C++] Exceptions do NOT cross thread boundaries (no Task machinery
    // rethrowing on await): one escaping a std::thread calls std::terminate.
    // The idiom is to capture the first one as std::exception_ptr and
    // rethrow it on the joining thread.
    std::exception_ptr error;
    std::mutex error_mu;

    auto worker = [&] {
        try {
            while (true) {
                const std::uint32_t id = next.fetch_add(1);
                if (id >= total) break;
                insert_node(id, /*locked=*/true);
            }
        } catch (...) {
            std::lock_guard<std::mutex> g(error_mu);
            if (!error) error = std::current_exception();
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(t);
    for (unsigned i = 0; i < t; ++i) workers.emplace_back(worker);
    for (auto& w : workers) w.join();

    if (error) std::rethrow_exception(error);
    return vectors.size();
}

}  // namespace vecdb

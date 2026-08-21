/**
 * @file hnsw_index.cpp
 * @brief HNSW implementation: insertion with heuristic neighbor selection,
 *        layered beam search, reverse-edge re-pruning.
 */
#include "core/hnsw_index.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <xmmintrin.h>  // _mm_prefetch (SSE, baseline on x86-64 — no special flags)

#include "core/dot.hpp"

namespace vecdb {

namespace {
constexpr float kMinNorm = 1e-6f;

// Comparators for the two beams. Candidate keeps "smaller d = closer".
// [C#→C++] std::priority_queue ≈ PriorityQueue<TElement,TPriority>, but the
// comparator is a TYPE (usually a struct with operator(), or a lambda's
// type), resolved at compile time — no virtual calls per comparison, the
// compare code inlines into the heap operations.
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
        // [C#→C++] lock_guard = the `lock (obj) {}` block of C#, as RAII: it
        // locks in the constructor and unlocks in the destructor at scope
        // end, exception-safe by construction.
        std::lock_guard<std::mutex> lock(mu_);
        if (!free_.empty()) {
            list = std::move(free_.back());
            free_.pop_back();
        }
    }
    if (!list) list = std::make_unique<List>();
    list->stamp.resize(n, 0);
    ++list->epoch;
    if (list->epoch == 0) {  // uint32 wraparound (once per ~4B queries): reset
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
      ml_(1.0 / std::log(static_cast<double>(config.M))),
      rng_(config.seed) {
    if (dim == 0) throw std::invalid_argument("HnswIndex: dim must be > 0");
    if (cfg_.M < 2) throw std::invalid_argument("HnswIndex: M must be >= 2");
}

float HnswIndex::dist_to(std::span<const float> q, std::uint32_t id) const {
    // Unit-norm vectors: cosine distance = 1 - dot. Monotone in the true
    // angular distance, which is all a comparison-based search needs.
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
    // [C#→C++] const_cast to share the implementation between the const and
    // non-const overload — the classic idiom. Safe here: we cast a genuinely
    // non-const object, and the const overload never writes through it.
    return const_cast<HnswIndex*>(this)->links(id, level);
}

int HnswIndex::random_level() {
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    double u = dist01(rng_);
    if (u <= 0.0) u = 1e-12;  // log(0) guard
    return static_cast<int>(-std::log(u) * ml_);
}

// -------------------------------------------------------------------- search

std::uint32_t HnswIndex::greedy_closest(std::span<const float> q, std::uint32_t ep,
                                        int level) const {
    float best = dist_to(q, ep);
    bool improved = true;
    while (improved) {
        improved = false;
        const std::uint32_t* l = links(ep, level);
        const std::uint32_t n = l[0];
        for (std::uint32_t j = 1; j <= n; ++j) {
            const float d = dist_to(q, l[j]);
            if (d < best) {
                best = d;
                ep = l[j];
                improved = true;
            }
        }
    }
    return ep;
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer(std::span<const float> q,
                                                          std::uint32_t ep,
                                                          std::size_t ef,
                                                          int level) const {
    auto lease = visited_pool_.acquire(count_);
    auto& visited = *lease;

    std::priority_queue<Candidate, std::vector<Candidate>, CloserFirst> frontier;
    std::priority_queue<Candidate, std::vector<Candidate>, FartherFirst> best;

    const float d0 = dist_to(q, ep);
    frontier.push({d0, ep});
    best.push({d0, ep});
    visited.test_and_set(ep);

    while (!frontier.empty()) {
        const Candidate c = frontier.top();
        // Stop condition: the closest unexpanded candidate is already worse
        // than the worst kept result and the result set is full — expanding
        // further cannot improve (exact on an ideal navigable graph; the
        // approximation of HNSW lives exactly here).
        if (c.d > best.top().d && best.size() >= ef) break;
        frontier.pop();

        const std::uint32_t* l = links(c.id, level);
        const std::uint32_t n = l[0];
        for (std::uint32_t j = 1; j <= n; ++j) {
            // Overlap the next neighbor's vector fetch (~80ns from RAM) with
            // the current distance computation.
            if (j < n) {
                _mm_prefetch(reinterpret_cast<const char*>(vec(l[j + 1]).data()),
                             _MM_HINT_T0);
            }
            const std::uint32_t nb = l[j];
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

    // Descend: greedy hops through the upper layers only pick the entry
    // point; the real beam search happens once, at layer 0.
    std::uint32_t ep = entry_;
    for (int level = max_level_; level >= 1; --level) {
        ep = greedy_closest(q, ep, level);
    }
    const auto found = search_layer(q, ep, ef_search, 0);

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
        // Simple selection: the m closest (candidates arrive sorted ascending).
        return {candidates.begin(),
                candidates.begin() + static_cast<std::ptrdiff_t>(
                                         std::min(m, candidates.size()))};
    }

    // Heuristic (Algorithm 4): accept c only if it is closer to base than to
    // every neighbor already selected. Rejects candidates "behind" an
    // existing edge; the surviving set is directionally diverse and creates
    // the long bridge edges that keep clusters mutually reachable.
    std::vector<Candidate> selected;
    std::vector<Candidate> pruned;
    selected.reserve(m);
    for (const Candidate& c : candidates) {
        if (selected.size() >= m) break;
        bool keep = true;
        for (const Candidate& s : selected) {
            // d(c, s) < d(c, base) means c sits behind s as seen from base.
            if (dist_to(vec(c.id), s.id) < c.d) {
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
    // keepPrunedConnections: full degree helps navigability; refill with the
    // closest rejected candidates.
    for (std::size_t i = 0; i < pruned.size() && selected.size() < m; ++i) {
        selected.push_back(pruned[i]);
    }
    return selected;
}

void HnswIndex::add_link(std::uint32_t node, std::uint32_t cand, int level) {
    std::uint32_t* l = links(node, level);
    const std::size_t cap = max_degree(level);
    if (l[0] < cap) {
        l[++l[0]] = cand;
        return;
    }
    // Slot full: re-prune with the SAME heuristic used at selection time.
    // Dropping "the farthest" instead silently destroys the diverse edges
    // and shows up weeks later as low recall, not as a failing test.
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

std::uint32_t HnswIndex::add(std::span<const float> v) {
    if (v.size() != dim_) {
        throw std::invalid_argument("HnswIndex::add: dimension mismatch");
    }
    const float norm = std::sqrt(dot(v, v));
    if (!(norm > kMinNorm)) {
        throw std::invalid_argument("HnswIndex::add: zero or invalid norm");
    }

    const auto id = static_cast<std::uint32_t>(count_);
    const int level = random_level();

    // Storage for the new node: vector data, level, adjacency slots (zeroed
    // counts). All appends — ids are stable, offsets are computed. Growth is
    // push_back/resize amortized doubling; no explicit reserve (an exact
    // reserve per insert defeats geometric growth: O(n^2) bytes copied).
    const float inv = 1.0f / norm;
    for (const float x : v) data_.push_back(x * inv);
    levels_.push_back(level);
    links0_.resize(links0_.size() + 1 + mmax0_, 0);
    if (level >= 1) {
        upper_offset_.push_back(upper_.size());
        upper_.resize(upper_.size() +
                          static_cast<std::size_t>(level) * (1 + cfg_.M),
                      0);
    } else {
        upper_offset_.push_back(kNoUpper);
    }
    ++count_;

    if (count_ == 1) {  // first node: becomes the entry point, no edges yet
        entry_ = id;
        max_level_ = level;
        return id;
    }

    const auto q = vec(id);

    // Phase 1 of the descent: from the top of the graph down to level+1,
    // greedy only — we just want a good entry point for the build beams.
    std::uint32_t ep = entry_;
    for (int lc = max_level_; lc > level; --lc) {
        ep = greedy_closest(q, ep, lc);
    }

    // Phase 2: on each layer the node lives in, run the construction beam,
    // pick M diverse neighbors, and wire both edge directions.
    for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
        const auto candidates = search_layer(q, ep, cfg_.ef_construction, lc);
        const auto neighbors = select_neighbors(q, candidates, cfg_.M);

        std::uint32_t* l = links(id, lc);
        l[0] = static_cast<std::uint32_t>(neighbors.size());
        for (std::size_t j = 0; j < neighbors.size(); ++j) {
            l[1 + j] = neighbors[j].id;
        }
        for (const Candidate& nb : neighbors) {
            add_link(nb.id, id, lc);  // reverse edge, heuristic re-prune if full
        }

        ep = candidates.front().id;  // best entry for the next layer down
    }

    if (level > max_level_) {  // node introduces a new top layer
        entry_ = id;
        max_level_ = level;
    }
    return id;
}

}  // namespace vecdb

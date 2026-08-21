/**
 * @file bench_hnsw.cpp
 * @brief HNSW at scale: build time, recall@10 vs efSearch sweep, latency vs
 *        the exact brute-force baseline, heuristic vs simple selection.
 *
 * @details
 * Corpus: mixture of Gaussians (clustered), which is what real text
 * embeddings look like and the regime where neighbor-selection quality
 * matters — uniform random data in high dimension has near-equidistant
 * neighbors and hides the difference.
 *
 * Ground truth comes from the exact VectorIndex, so recall is measured, not
 * estimated. Latency is wall-clock per query amortized over the query batch.
 *
 * Usage: vecdb_bench_hnsw [n] [dim]   (defaults: 50000 768)
 */
#include <chrono>
#include <cstdio>
#include <memory>
#include <cstdlib>
#include <random>
#include <vector>

#include "core/hnsw_index.hpp"
#include "core/vector_index.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

double recall_at_k(const std::vector<vecdb::SearchResult>& exact,
                   const std::vector<vecdb::SearchResult>& approx) {
    std::size_t hits = 0;
    for (const auto& e : exact) {
        for (const auto& a : approx) {
            if (a.id == e.id) {
                ++hits;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(exact.size());
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 50'000;
    const std::size_t dim = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 768;
    constexpr std::size_t k = 10;
    constexpr int n_queries = 100;
    constexpr std::size_t n_clusters = 256;

    // Line-buffer stdout so progress is visible when output goes to a file.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::printf("HNSW benchmark: n=%zu dim=%zu k=%zu queries=%d (clustered corpus)\n\n",
                n, dim, k, n_queries);

    // ---- corpus: 256 Gaussian clusters ------------------------------------
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> centers_dist(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.25f);

    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& c : centers) {
        c.resize(dim);
        for (auto& x : c) x = centers_dist(rng);
    }
    const auto make_point = [&](std::size_t cluster) {
        std::vector<float> v = centers[cluster];
        for (auto& x : v) x += noise(rng);
        return v;
    };

    std::vector<std::vector<float>> corpus;
    corpus.reserve(n);
    for (std::size_t i = 0; i < n; ++i) corpus.push_back(make_point(i % n_clusters));

    std::vector<std::vector<float>> queries;
    for (int i = 0; i < n_queries; ++i) {
        queries.push_back(make_point(static_cast<std::size_t>(i) % n_clusters));
    }

    // ---- build: exact oracle + two HNSW variants --------------------------
    vecdb::VectorIndex oracle(dim);
    for (const auto& v : corpus) oracle.add(v);

    vecdb::HnswConfig cfg_h;  // defaults: M=16, efC=200, heuristic on
    vecdb::HnswConfig cfg_s = cfg_h;
    cfg_s.use_heuristic = false;

    auto build = [&](const vecdb::HnswConfig& cfg, const char* label) {
        auto index = std::make_unique<vecdb::HnswIndex>(dim, cfg);
        const auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            index->add(corpus[i]);
            if ((i + 1) % 10'000 == 0) {
                std::printf("  [%s] %zu/%zu inserted (%.1fs)\n", label, i + 1, n,
                            seconds_since(t0));
            }
        }
        const double dt = seconds_since(t0);
        std::printf("build %-9s: %.1fs (%.0f inserts/s, max_level=%d)\n\n", label, dt,
                    n / dt, index->max_level());
        return index;
    };

    const auto hnsw_h = build(cfg_h, "heuristic");
    const auto hnsw_s = build(cfg_s, "simple");

    // ---- ground truth + brute-force latency baseline ----------------------
    std::vector<std::vector<vecdb::SearchResult>> exact;
    const auto t0 = Clock::now();
    for (const auto& q : queries) exact.push_back(oracle.search(q, k));
    const double brute_us = seconds_since(t0) * 1e6 / n_queries;
    std::printf("brute force (exact): %.0f us/query — the baseline HNSW must beat\n\n",
                brute_us);

    // ---- efSearch sweep ----------------------------------------------------
    std::printf("%-6s | %-22s | %-22s\n", "", "heuristic", "simple");
    std::printf("%-6s | %9s %12s | %9s %12s\n", "ef", "recall@10", "us/query",
                "recall@10", "us/query");
    std::printf("-------+------------------------+-----------------------\n");

    for (const std::size_t ef : {10u, 20u, 50u, 100u, 200u, 400u}) {
        auto run = [&](const vecdb::HnswIndex& index, double& rec, double& usq) {
            rec = 0.0;
            const auto ts = Clock::now();
            for (int i = 0; i < n_queries; ++i) {
                const auto approx = index.search(queries[i], k, ef);
                rec += recall_at_k(exact[i], approx);
            }
            usq = seconds_since(ts) * 1e6 / n_queries;
            rec /= n_queries;
        };
        double rh, uh, rs, us;
        run(*hnsw_h, rh, uh);
        run(*hnsw_s, rs, us);
        std::printf("%-6zu | %9.3f %10.0fus | %9.3f %10.0fus\n", ef, rh, uh, rs, us);
    }

    std::printf("\nspeedup at ef=100 vs exact: see table (both indexes searched the\n"
                "same %d queries; recall measured against the exact top-%zu).\n",
                n_queries, k);
    return 0;
}

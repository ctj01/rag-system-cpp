/**
 * @file test_hnsw.cpp
 * @brief HNSW tests: recall@k against the brute-force oracle, deterministic
 *        builds, self-retrieval, and heuristic-vs-simple neighbor selection.
 *
 * The brute-force VectorIndex is exact, so it serves as ground truth: HNSW
 * quality is measured, not assumed. All builds are seeded — every value
 * below is deterministic, so thresholds are tight-but-safe bounds on values
 * observed at fixed seeds, not statistical hopes.
 */
#include "core/hnsw_index.hpp"

#include <atomic>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/vector_index.hpp"
#include "test_framework.hpp"

namespace {

std::vector<float> random_vec(std::mt19937& rng, std::size_t dim) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

/// Fraction of the exact top-k also present in the approximate top-k.
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

int main() {
    constexpr std::size_t dim = 32;
    constexpr std::size_t n = 2000;
    constexpr std::size_t k = 10;
    constexpr int n_queries = 50;

    std::mt19937 rng(123);

    // Shared corpus: uniform random vectors (a hard, cluster-free case).
    std::vector<std::vector<float>> corpus;
    corpus.reserve(n);
    for (std::size_t i = 0; i < n; ++i) corpus.push_back(random_vec(rng, dim));

    vecdb::VectorIndex oracle(dim);
    vecdb::HnswIndex hnsw(dim);
    for (const auto& v : corpus) {
        oracle.add(v);
        hnsw.add(v);
    }
    CHECK(hnsw.size() == n);
    CHECK(hnsw.max_level() >= 1);  // with n=2000, M=16 the hierarchy must exist

    // Recall@10 against the exact oracle.
    {
        double total = 0.0;
        for (int qi = 0; qi < n_queries; ++qi) {
            const auto q = random_vec(rng, dim);
            const auto exact = oracle.search(q, k);
            const auto approx = hnsw.search(q, k, /*ef_search=*/100);
            CHECK(approx.size() == k);
            // Scores must be sorted descending.
            for (std::size_t j = 1; j < approx.size(); ++j) {
                CHECK(approx[j - 1].score >= approx[j].score);
            }
            total += recall_at_k(exact, approx);
        }
        const double avg = total / n_queries;
        std::printf("recall@%zu (ef=100, uniform): %.3f\n", k, avg);
        CHECK(avg >= 0.90);

        // efSearch is the recall knob: more beam must not hurt.
        double total_wide = 0.0;
        std::mt19937 rng2(123 + 1);
        (void)rng2;
        for (int qi = 0; qi < n_queries; ++qi) {
            const auto q = random_vec(rng, dim);
            total_wide += recall_at_k(oracle.search(q, k), hnsw.search(q, k, 400));
        }
        std::printf("recall@%zu (ef=400, uniform): %.3f\n", k, total_wide / n_queries);
        CHECK(total_wide / n_queries >= 0.95);
    }

    // Self-retrieval: querying with a stored vector must return it first
    // (distance 0). A failure here means an unreachable node — a
    // connectivity bug, not an approximation error.
    {
        int found = 0;
        for (std::uint32_t i = 0; i < 500; ++i) {
            const auto res = hnsw.search(corpus[i], 1, 50);
            if (!res.empty() && res[0].id == i) ++found;
        }
        std::printf("self-retrieval top-1: %d/500\n", found);
        CHECK(found >= 495);
    }

    // Deterministic build: same seed + same insertion order = same graph =
    // identical search results, bit for bit.
    {
        vecdb::HnswIndex a(dim), b(dim);
        for (std::size_t i = 0; i < 300; ++i) {
            a.add(corpus[i]);
            b.add(corpus[i]);
        }
        const auto q = corpus[7];
        const auto ra = a.search(q, 5);
        const auto rb = b.search(q, 5);
        CHECK(ra.size() == rb.size());
        for (std::size_t j = 0; j < ra.size(); ++j) {
            CHECK(ra[j].id == rb[j].id);
            CHECK(ra[j].score == rb[j].score);
        }
    }

    // Heuristic vs simple selection on CLUSTERED data — the case the
    // heuristic exists for. Simple selection spends all edges inside the
    // local cluster and can strand the greedy descent in the wrong one.
    {
        constexpr int n_clusters = 8;
        constexpr std::size_t per_cluster = 150;
        std::mt19937 crng(77);
        std::normal_distribution<float> noise(0.0f, 0.08f);

        std::vector<std::vector<float>> centers;
        for (int c = 0; c < n_clusters; ++c) centers.push_back(random_vec(crng, dim));

        vecdb::HnswConfig with_h;   // defaults: use_heuristic = true
        vecdb::HnswConfig without_h;
        without_h.use_heuristic = false;

        vecdb::HnswIndex h(dim, with_h), s(dim, without_h);
        vecdb::VectorIndex ex(dim);
        for (int c = 0; c < n_clusters; ++c) {
            for (std::size_t i = 0; i < per_cluster; ++i) {
                std::vector<float> v = centers[c];
                for (auto& x : v) x += noise(crng);
                h.add(v);
                s.add(v);
                ex.add(v);
            }
        }

        double rh = 0.0, rs = 0.0;
        constexpr int cq = 40;
        for (int qi = 0; qi < cq; ++qi) {
            std::vector<float> q = centers[qi % n_clusters];
            for (auto& x : q) x += noise(crng);
            const auto exact = ex.search(q, k);
            rh += recall_at_k(exact, h.search(q, k, 100));
            rs += recall_at_k(exact, s.search(q, k, 100));
        }
        rh /= cq;
        rs /= cq;
        std::printf("clustered recall@%zu: heuristic=%.3f simple=%.3f\n", k, rh, rs);
        CHECK(rh >= 0.90);
        CHECK(rh >= rs - 0.02);  // heuristic never meaningfully worse
    }

    // Parallel batch build. The resulting graph depends on thread
    // interleaving (documented trade-off), so we assert recall bounds and
    // connectivity, not exact structure.
    {
        vecdb::HnswIndex par(dim);
        CHECK(par.add_batch(corpus, 8) == n);
        CHECK(par.size() == n);

        std::mt19937 qrng(555);
        double total = 0.0;
        for (int qi = 0; qi < 30; ++qi) {
            const auto q = random_vec(qrng, dim);
            total += recall_at_k(oracle.search(q, k), par.search(q, k, 100));
        }
        std::printf("recall@%zu (parallel build, 8 threads): %.3f\n", k, total / 30);
        CHECK(total / 30 >= 0.90);

        // Random sample across the whole id range (early ids are warmup-built,
        // late ids are parallel-built — sample both regimes).
        std::mt19937 pick(999);
        std::uniform_int_distribution<std::uint32_t> ids(0, static_cast<std::uint32_t>(n - 1));
        int found = 0;
        for (int s = 0; s < 200; ++s) {
            const std::uint32_t i = ids(pick);
            const auto r = par.search(corpus[i], 1, 50);
            if (!r.empty() && r[0].id == i) ++found;
        }
        std::printf("self-retrieval top-1 (parallel build): %d/200\n", found);
        CHECK(found >= 192);  // small slack: the parallel graph is interleaving-dependent
    }

    // Concurrent frozen-graph search: 8 threads must produce results
    // identical to a serial pass (search is lock-free and read-only).
    {
        std::mt19937 qrng(777);
        std::vector<std::vector<float>> qs;
        for (int i = 0; i < 32; ++i) qs.push_back(random_vec(qrng, dim));
        std::vector<std::vector<vecdb::SearchResult>> serial;
        for (const auto& q : qs) serial.push_back(hnsw.search(q, k, 100));

        std::atomic<int> mismatches{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&] {
                for (std::size_t i = 0; i < qs.size(); ++i) {
                    const auto r = hnsw.search(qs[i], k, 100);
                    if (r.size() != serial[i].size()) {
                        ++mismatches;
                        continue;
                    }
                    for (std::size_t j = 0; j < r.size(); ++j) {
                        if (r[j].id != serial[i][j].id) {
                            ++mismatches;
                            break;
                        }
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
        CHECK(mismatches == 0);
    }

    // Persistence: save → load → bit-identical search results.
    {
        const char* path = "test_hnsw_index.bin";
        hnsw.save(path);
        const auto loaded = vecdb::HnswIndex::load(path);
        CHECK(loaded.size() == hnsw.size());
        CHECK(loaded.dim() == hnsw.dim());
        CHECK(loaded.max_level() == hnsw.max_level());

        std::mt19937 qrng(1234);
        for (int qi = 0; qi < 10; ++qi) {
            const auto q = random_vec(qrng, dim);
            const auto a = hnsw.search(q, k, 100);
            const auto b = loaded.search(q, k, 100);
            CHECK(a.size() == b.size());
            for (std::size_t j = 0; j < a.size(); ++j) {
                CHECK(a[j].id == b[j].id);
                CHECK(a[j].score == b[j].score);  // same graph -> bit-exact
            }
        }
        std::remove(path);

        // Corrupted file: wrong magic must throw, not crash.
        {
            std::FILE* f = std::fopen(path, "wb");
            std::fputs("garbage that is not an index", f);
            std::fclose(f);
        }
        CHECK_THROWS(vecdb::HnswIndex::load(path), std::runtime_error);
        std::remove(path);
        CHECK_THROWS(vecdb::HnswIndex::load("no_such_file.bin"), std::runtime_error);
    }

    // Validation and edge cases.
    {
        vecdb::HnswIndex empty(dim);
        CHECK(empty.search(corpus[0], 5).empty());

        vecdb::HnswIndex one(dim);
        one.add(corpus[0]);
        const auto r = one.search(corpus[0], 10);  // k > size clamps
        CHECK(r.size() == 1);
        CHECK(r[0].id == 0);

        CHECK_THROWS(vecdb::HnswIndex(0), std::invalid_argument);
        CHECK_THROWS(hnsw.search(std::vector<float>(dim + 1, 1.0f), 3),
                     std::invalid_argument);
        CHECK_THROWS(one.add(std::vector<float>(dim, 0.0f)), std::invalid_argument);
    }

    return test_summary("test_hnsw");
}

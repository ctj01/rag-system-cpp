/**
 * @file bench_dot.cpp
 * @brief Micro-benchmark: scalar vs AVX2 dot product, and full-index scans
 *        across cache levels to expose the memory-bandwidth ceiling.
 *
 * @details
 * Methodology:
 *  - Timing via std::chrono::steady_clock (monotonic; wall clocks can jump).
 *  - Each measurement is calibrated so a single sample runs >= ~20 ms, long
 *    enough to drown out timer overhead and scheduler jitter.
 *  - The MEDIAN of several samples is reported, not the mean: on a noisy
 *    host (WSL2 included) the mean is dragged by outliers, the median is not.
 *  - Results are fed into an accumulator passed through do_not_optimize();
 *    otherwise -O3 sees an unused value and deletes the entire loop.
 *  - dot_scalar/dot_avx2 live in another translation unit and LTO is off, so
 *    the compiler cannot inline or hoist them out of the timing loop.
 *
 * Note on fairness: GCC does NOT auto-vectorize the scalar accumulation loop
 * at -O3, because reordering a float reduction changes the result and that is
 * only allowed under -ffast-math. So "scalar" here is genuinely scalar.
 */
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

#include "core/dot.hpp"

namespace {

// [C#→C++] Equivalent of BenchmarkDotNet's Consume()/DoNotOptimize. The empty
// asm block is opaque to the optimizer and claims to read *p and touch
// memory, so the value feeding it must actually be computed. C# needs library
// support for this; C++ does it with one line of inline assembly.
inline void do_not_optimize(const void* p) {
    asm volatile("" : : "g"(p) : "memory");
}

using Clock = std::chrono::steady_clock;

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/**
 * @brief Measures fn(), auto-calibrated: runs fn in batches sized so one
 *        timed sample lasts >= target_ms, then returns the median ns per call.
 */
template <typename Fn>
double bench_ns_per_call(Fn&& fn, double target_ms = 20.0, int samples = 7) {
    // Warmup: fault in pages, warm caches and branch predictors.
    fn();

    // Calibrate batch size from one rough measurement.
    auto t0 = Clock::now();
    fn();
    const double once_ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    const std::uint64_t batch =
        std::max<std::uint64_t>(1, static_cast<std::uint64_t>(target_ms * 1e6 / std::max(once_ns, 1.0)));

    std::vector<double> per_call(samples);
    for (int s = 0; s < samples; ++s) {
        t0 = Clock::now();
        for (std::uint64_t i = 0; i < batch; ++i) {
            fn();
        }
        const double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
        per_call[s] = ns / static_cast<double>(batch);
    }
    return median(per_call);
}

std::vector<float> random_floats(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

/// Single-pair dot product, data hot in L1: isolates pure compute throughput.
void bench_kernels() {
    std::printf("== Kernel: single dot product, data hot in cache ==\n");
    std::printf("%-6s %14s %14s %10s\n", "dim", "scalar ns", "avx2 ns", "speedup");

    for (const std::size_t dim : {384u, 768u, 1536u}) {
        const auto a = random_floats(dim, 1);
        const auto b = random_floats(dim, 2);
        float sink = 0.0f;

        const double ns_scalar = bench_ns_per_call([&] {
            sink += vecdb::dot_scalar(a, b);
            do_not_optimize(&sink);
        });
        const double ns_avx2 = bench_ns_per_call([&] {
            sink += vecdb::dot_avx2(a, b);
            do_not_optimize(&sink);
        });

        std::printf("%-6zu %14.1f %14.1f %9.1fx\n", dim, ns_scalar, ns_avx2, ns_scalar / ns_avx2);
    }
    std::printf("\n");
}

/// Full scan over a row-major buffer (the brute-force search hot loop) at
/// working-set sizes targeting L2, L3 and RAM. As the set outgrows the
/// caches, AVX2's advantage collapses toward the memory-bandwidth ceiling.
void bench_scans() {
    constexpr std::size_t dim = 768;
    struct Case { const char* label; std::size_t rows; };
    const Case cases[] = {
        {"L2-resident", 512},      //  ~1.5 MB
        {"L3-resident", 8192},     //   ~24 MB
        {"RAM-resident", 65536},   //  ~192 MB
    };

    std::printf("== Scan: dot against every row (dim=%zu) ==\n", dim);
    std::printf("%-13s %9s %9s %13s %13s %10s %11s %11s\n",
                "working set", "rows", "MB", "scalar ns/v", "avx2 ns/v",
                "speedup", "scalar GB/s", "avx2 GB/s");

    const auto q = random_floats(dim, 3);

    for (const auto& c : cases) {
        const auto data = random_floats(c.rows * dim, 4);
        const double mb = static_cast<double>(c.rows * dim * sizeof(float)) / (1024.0 * 1024.0);
        const float* base = data.data();
        float sink = 0.0f;

        const auto scan = [&](auto&& kernel) {
            float acc = 0.0f;
            for (std::size_t i = 0; i < c.rows; ++i) {
                acc += kernel(std::span<const float>(q),
                              std::span<const float>(base + i * dim, dim));
            }
            sink += acc;
            do_not_optimize(&sink);
        };

        const double ns_scalar = bench_ns_per_call([&] { scan(vecdb::dot_scalar); }) / c.rows;
        const double ns_avx2   = bench_ns_per_call([&] { scan(vecdb::dot_avx2); }) / c.rows;

        // Bytes streamed per row: the row itself (the query stays in L1).
        const double row_bytes = static_cast<double>(dim * sizeof(float));
        std::printf("%-13s %9zu %9.1f %13.1f %13.1f %9.1fx %11.1f %11.1f\n",
                    c.label, c.rows, mb, ns_scalar, ns_avx2, ns_scalar / ns_avx2,
                    row_bytes / ns_scalar, row_bytes / ns_avx2);
    }
    std::printf("\n(GB/s = bytes of vector data streamed per second; compare against\n"
                " your platform's practical single-core RAM bandwidth to see the ceiling.)\n");
}

}  // namespace

int main() {
    std::printf("vecdb micro-benchmark (Release, median of 7 samples)\n\n");
    bench_kernels();
    bench_scans();
    return 0;
}

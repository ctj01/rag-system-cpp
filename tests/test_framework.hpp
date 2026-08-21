/**
 * @file test_framework.hpp
 * @brief Minimal zero-dependency test harness.
 *
 * CHECK() reports a failure and continues; each test's main() returns the
 * failure count (non-zero exit code => ctest marks the test FAILED).
 */
#pragma once

#include <cmath>
#include <cstdio>

// [C#→C++] inline on a global variable lets it be defined in a header
// included from several .cpp files without violating the One Definition Rule
// — think "a single shared static". Without inline, each .cpp would get its
// own copy (or a linker error if non-static).
inline int g_failures = 0;

// [C#→C++] Macros are preprocessor text substitution — no types, no scope.
// C# has no equivalent (test frameworks use reflection + attributes; C++ has
// no reflection, so the classic idiom is macro + __FILE__/__LINE__ to capture
// the call site).

/// @brief Checks a condition; on failure prints file:line and the expression.
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

/// @brief Checks that evaluating @p expr throws an exception of type @p ExType.
#define CHECK_THROWS(expr, ExType)                                         \
    do {                                                                   \
        bool caught_ = false;                                              \
        try {                                                              \
            (void)(expr);                                                  \
        } catch (const ExType&) {                                          \
            caught_ = true;                                                \
        }                                                                  \
        if (!caught_) {                                                    \
            std::printf("FAIL %s:%d: expected %s from %s\n",               \
                        __FILE__, __LINE__, #ExType, #expr);               \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

/**
 * @brief Floating-point comparison with relative tolerance and absolute floor.
 * @param a First value.
 * @param b Second value.
 * @param rel Relative tolerance (scaled by the larger magnitude).
 * @param abs_floor Absolute tolerance floor for values near zero.
 * @return true if the values are approximately equal.
 */
inline bool approx_eq(double a, double b, double rel, double abs_floor) {
    const double diff = std::fabs(a - b);
    const double scale = std::fmax(std::fabs(a), std::fabs(b));
    return diff <= std::fmax(rel * scale, abs_floor);
}

/// @brief Prints a pass/fail summary and returns the failure count.
inline int test_summary(const char* name) {
    if (g_failures == 0) {
        std::printf("%s: all checks passed\n", name);
    } else {
        std::printf("%s: %d check(s) FAILED\n", name, g_failures);
    }
    return g_failures;
}

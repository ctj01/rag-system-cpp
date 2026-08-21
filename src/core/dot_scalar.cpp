/**
 * @file dot_scalar.cpp
 * @brief Scalar reference implementation of the dot product.
 */
#include "core/dot.hpp"

#include <cassert>

namespace vecdb {

float dot_scalar(std::span<const float> a, std::span<const float> b) {
    // [C#→C++] assert() only exists in Debug builds (stripped by -DNDEBUG in
    // Release). It is a development contract, not runtime validation — C++
    // avoids paying for checks in Release hot paths, unlike C# where array
    // bounds checks are always present unless the JIT elides them.
    assert(a.size() == b.size());

    float acc = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

}  // namespace vecdb

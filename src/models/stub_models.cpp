/**
 * @file stub_models.cpp
 * @brief Deterministic stub implementations of EmbeddingModel and Generator.
 *
 * @details
 * HashEmbedder implements signed feature hashing ("the hashing trick"):
 * tokenize on non-alphanumeric boundaries, lowercase, FNV-1a hash each token,
 * and add +1/-1 (sign taken from one hash bit) into bucket (hash % dim).
 * Properties that make it a useful stand-in for a real embedding model:
 *  - Deterministic: same text → same vector, forever. Tests stay stable.
 *  - Semantically non-trivial: texts sharing tokens get correlated vectors,
 *    so cosine retrieval genuinely ranks related chunks higher. The RAG
 *    pipeline can be built and tested end-to-end before llama.cpp lands.
 *  - The random sign keeps E[dot] ≈ 0 for unrelated texts (collisions cancel
 *    out on average instead of accumulating positive bias).
 */
#include <cctype>
#include <cstdint>

#include "models/embedding_model.hpp"
#include "models/generator.hpp"

namespace vecdb {
namespace {

/// FNV-1a 64-bit: tiny, fast, deterministic non-cryptographic hash.
constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

/// splitmix64 finalizer: avalanche mixing so every output bit depends on
/// every input bit. Needed because FNV-1a's LOW bits are weak for short
/// strings, and bucket = h % dim (dim a power of two) reads only low bits —
/// without mixing, "the" and "a" landed in the same bucket at dim 256 AND
/// 1024, silently cancelling shared-token contributions in the dot product.
std::uint64_t mix(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

bool is_token_char(unsigned char c) {
    // [C#→C++] <cctype> functions require an unsigned char (passing a
    // negative plain char is UB) — hence the cast at call sites. No
    // char.IsLetterOrDigit niceties; and this is byte-wise ASCII, not
    // Unicode-aware like .NET strings.
    return std::isalnum(c) != 0;
}

// [C#→C++] Deriving with `final` = C# sealed. `override` is like C#'s, but
// here the compiler REQUIRES the base method to be virtual; writing override
// on a non-virtual is a compile error, which catches signature typos that
// would otherwise silently create a new unrelated method (C++'s equivalent
// of accidental `new`-style hiding).
class HashEmbedder final : public EmbeddingModel {
public:
    explicit HashEmbedder(std::size_t d) : dim_(d) {}

    std::size_t dim() const override { return dim_; }

    std::vector<float> embed(std::string_view text) const override {
        std::vector<float> v(dim_, 0.0f);
        std::size_t i = 0;
        const std::size_t n = text.size();
        while (i < n) {
            // Skip separators.
            while (i < n && !is_token_char(static_cast<unsigned char>(text[i]))) ++i;
            // Hash one token, lowercased, without materializing a substring.
            std::uint64_t h = kFnvOffset;
            const std::size_t start = i;
            while (i < n && is_token_char(static_cast<unsigned char>(text[i]))) {
                const auto c = static_cast<unsigned char>(
                    std::tolower(static_cast<unsigned char>(text[i])));
                h = (h ^ c) * kFnvPrime;
                ++i;
            }
            if (i > start) {
                const std::uint64_t m = mix(h);
                const std::size_t bucket = static_cast<std::size_t>(m % dim_);
                const float sign = (m >> 63) ? 1.0f : -1.0f;
                v[bucket] += sign;
            }
        }
        return v;
    }

private:
    std::size_t dim_;
};

class EchoGenerator final : public Generator {
public:
    std::string generate(std::string_view prompt) override {
        std::string out;
        out.reserve(prompt.size() + 64);
        out += "[stub generator — echoing prompt]\n";
        out += prompt;
        return out;
    }
};

}  // namespace

// Factories are the only symbols exported from this file: callers depend on
// the interfaces alone, concrete types stay private to this translation unit.
// [C#→C++] std::make_unique<T>(args) ≈ new T(args) wrapped in ownership.
// Implicit conversion unique_ptr<Derived> → unique_ptr<Base> works because
// the base destructor is virtual.
std::unique_ptr<EmbeddingModel> make_hash_embedder(std::size_t dim) {
    return std::make_unique<HashEmbedder>(dim);
}

std::unique_ptr<Generator> make_echo_generator() {
    return std::make_unique<EchoGenerator>();
}

}  // namespace vecdb

/**
 * @file embedding_model.hpp
 * @brief Abstract interface for text → vector embedding models.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

namespace vecdb {

/**
 * @brief Interface for embedding models: maps text to a fixed-dimension
 *        float vector.
 *
 * Implementations: HashEmbedder (deterministic stub, feature hashing) today;
 * a llama.cpp-backed model later. The RAG layer depends only on this
 * interface, so swapping the backend touches no orchestration code.
 */
// [C#→C++] C++ has no `interface` keyword. The idiom is a class with only
// pure virtual methods (`= 0` means "no body here, deriveds MUST implement" —
// like C# abstract). A class with at least one pure virtual method cannot be
// instantiated, exactly like a C# interface/abstract class.
class EmbeddingModel {
public:
    // [C#→C++] The virtual destructor is CRITICAL and has no C# analogue.
    // When deleting through a base pointer (unique_ptr<EmbeddingModel> owning
    // a HashEmbedder), a non-virtual destructor destroys only the base part —
    // undefined behavior, silent leaks. In C# the GC always knows the real
    // type; in C++ you must opt in. Rule: any class meant to be inherited
    // from needs `virtual ~T()`.
    virtual ~EmbeddingModel() = default;

    /// @brief Dimensionality of the vectors this model produces.
    virtual std::size_t dim() const = 0;

    /**
     * @brief Embeds @p text into a dim()-sized vector.
     * @param text Input text (UTF-8 bytes; stubs treat it as ASCII-ish).
     * @return Raw (un-normalized) embedding — VectorIndex normalizes on add.
     * @note May return an all-zero vector for texts with no usable content
     *       (e.g. empty string); VectorIndex::add rejects those.
     */
    // [C#→C++] std::string_view ≈ ReadOnlySpan<char> over a string: pointer +
    // length, no copy, no ownership — accepts std::string, string literals,
    // or substrings for free. Same lifetime caveat as std::span: no GC keeps
    // the underlying buffer alive.
    virtual std::vector<float> embed(std::string_view text) const = 0;
};

/**
 * @brief Creates the deterministic feature-hashing stub embedder.
 * @param dim Output dimensionality (e.g. 256).
 * @return Owning pointer to the model.
 */
// [C#→C++] std::unique_ptr<T> = exclusive ownership, no GC: when the
// unique_ptr goes out of scope, its destructor deletes the object (RAII).
// It is move-only — copying is a compile error, ownership must be
// transferred explicitly with std::move. Returning it by value hands
// ownership to the caller. The nearest C# mental model is "the only
// reference that exists, with a deterministic Dispose built in".
std::unique_ptr<EmbeddingModel> make_hash_embedder(std::size_t dim);

}  // namespace vecdb

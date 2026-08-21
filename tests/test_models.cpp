/**
 * @file test_models.cpp
 * @brief Stub model tests: determinism, polymorphic use through the
 *        interfaces, and end-to-end retrieval quality of the hash embedder.
 */
#include <cmath>
#include <string>
#include <vector>

#include "core/dot.hpp"
#include "core/vector_index.hpp"
#include "models/embedding_model.hpp"
#include "models/generator.hpp"
#include "test_framework.hpp"

namespace {

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    const float na = std::sqrt(vecdb::dot_scalar(a, a));
    const float nb = std::sqrt(vecdb::dot_scalar(b, b));
    return vecdb::dot_scalar(a, b) / (na * nb);
}

}  // namespace

int main() {
    // [C#→C++] `auto` = C# var. The factory returns std::unique_ptr — a
    // move-only type: assigning it elsewhere requires std::move; when
    // `embedder` goes out of scope at the end of main(), the model is
    // deleted deterministically (no GC involved).
    const auto embedder = vecdb::make_hash_embedder(256);

    CHECK(embedder->dim() == 256);

    // Determinism: same text, same vector — bit-identical.
    {
        const auto v1 = embedder->embed("kalman filter hedge ratio");
        const auto v2 = embedder->embed("kalman filter hedge ratio");
        CHECK(v1 == v2);  // std::vector compares element-wise
        CHECK(v1.size() == 256);
    }

    // Tokenization is case-insensitive and separator-agnostic.
    {
        const auto a = embedder->embed("Kalman   FILTER!!");
        const auto b = embedder->embed("kalman filter");
        CHECK(a == b);
    }

    // Empty / token-less text yields the all-zero vector (documented
    // behavior; VectorIndex::add rejects it downstream).
    {
        const auto z = embedder->embed("  ...  ");
        float mass = 0.0f;
        for (float x : z) mass += std::fabs(x);
        CHECK(mass == 0.0f);
    }

    // Shared vocabulary must produce higher cosine than unrelated text.
    {
        const auto q = embedder->embed("the kalman filter estimates the hedge ratio");
        const auto rel = embedder->embed("a kalman filter for dynamic hedge estimation");
        const auto unrel = embedder->embed("boca juniors won the match yesterday");
        CHECK(cosine(q, rel) > cosine(q, unrel));
    }

    // End-to-end: embed a tiny corpus, index it, and check retrieval ranks
    // the semantically related chunk first. This is the RAG retrieval path
    // minus chunking and prompting.
    {
        const std::string corpus[] = {
            "the kalman filter estimates a dynamic hedge ratio for pairs trading",
            "cointegration tests select pairs with a stationary spread",
            "gradient descent minimizes the loss function of a neural network",
            "the recipe requires flour, eggs and two cups of sugar",
        };

        vecdb::VectorIndex index(embedder->dim());
        for (const auto& text : corpus) {
            index.add(embedder->embed(text));
        }

        const auto hits =
            index.search(embedder->embed("kalman filter hedge ratio estimation"), 2);
        CHECK(hits.size() == 2);
        CHECK(hits[0].id == 0);                 // the kalman/hedge chunk wins
        CHECK(hits[0].score > hits[1].score);
    }

    // Generator stub: echoes the prompt so the assembled prompt is inspectable.
    {
        const auto gen = vecdb::make_echo_generator();
        const auto out = gen->generate("QUESTION: what is the hedge ratio?");
        CHECK(out.find("QUESTION: what is the hedge ratio?") != std::string::npos);
        CHECK(out.find("[stub generator") != std::string::npos);
    }

    return test_summary("test_models");
}

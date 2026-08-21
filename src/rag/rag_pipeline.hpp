/**
 * @file rag_pipeline.hpp
 * @brief RAG orchestration: chunk → embed → index → retrieve → prompt → generate.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/hnsw_index.hpp"
#include "models/embedding_model.hpp"
#include "models/generator.hpp"
#include "rag/chunker.hpp"

namespace vecdb {

/**
 * @brief One retrieved passage backing an answer.
 * @warning text/source are views into pipeline-owned storage: valid while
 *          the pipeline lives and no further documents are added.
 */
struct RetrievedChunk {
    std::uint32_t chunk_id;
    float score;              ///< Cosine similarity to the question.
    std::string_view text;
    std::string_view source;  ///< Document label the chunk came from.
};

/**
 * @brief Full result of a question: the answer, its evidence, and the exact
 *        prompt sent to the generator (auditability — with the echo stub the
 *        prompt IS the visible pipeline output).
 */
struct RagResponse {
    std::string answer;
    std::vector<RetrievedChunk> retrieved;
    std::string prompt;
};

/**
 * @brief End-to-end RAG pipeline over the HNSW index.
 *
 * Ingestion: add_document() chunks the text, embeds each chunk, and indexes
 * it. Query: ask() embeds the question, retrieves top-k chunks, assembles a
 * grounded prompt (numbered passages + instructions to answer only from
 * context), and calls the generator.
 */
class RagPipeline {
public:
    /**
     * @brief Takes ownership of both models.
     *
     * @param embedder Embedding backend (stub or llama.cpp later).
     * @param generator Generation backend.
     * @param chunk_cfg Chunking window/overlap.
     * @param index_cfg HNSW tunables.
     * @throws std::invalid_argument if either model pointer is null.
     */
    // [C#→C++] Dependency injection, C++ style: unique_ptr BY VALUE forces
    // the caller to hand over ownership explicitly —
    //   RagPipeline p(std::move(embedder), std::move(generator));
    // After the move the caller's pointers are null; the pipeline controls
    // the models' lifetime (destroyed with it, deterministically). In .NET
    // the DI container owns lifetimes; here the type signature does.
    RagPipeline(std::unique_ptr<EmbeddingModel> embedder,
                std::unique_ptr<Generator> generator,
                ChunkerConfig chunk_cfg = {},
                HnswConfig index_cfg = {});

    /**
     * @brief Chunks, embeds and indexes a document.
     * @param text Full document text.
     * @param source Label for attribution (filename, title...).
     * @return Number of chunks actually indexed (chunks whose embedding has
     *         no usable tokens are skipped).
     */
    std::size_t add_document(std::string_view text, std::string source);

    /**
     * @brief Answers a question from the indexed corpus.
     * @param question Natural-language question.
     * @param k Number of passages to retrieve (clamped to chunk_count()).
     * @return Answer, evidence and the assembled prompt.
     * @throws std::invalid_argument if the question embeds to a zero vector
     *         (no usable tokens) or the corpus is empty.
     */
    RagResponse ask(std::string_view question, std::size_t k = 4) const;

    std::size_t chunk_count() const noexcept { return chunks_.size(); }
    std::size_t document_count() const noexcept { return sources_.size(); }

private:
    std::string build_prompt(std::string_view question,
                             const std::vector<RetrievedChunk>& hits) const;

    // [C#→C++] const methods + unique_ptr: constness is SHALLOW. In ask()
    // const, generator_ (the pointer) is const but the Generator it points
    // to is not — calling the non-const generate() compiles. Same as C#
    // readonly (the reference is fixed, the object isn't), but worth knowing
    // because everything else about const here is deep.
    std::unique_ptr<EmbeddingModel> embedder_;
    std::unique_ptr<Generator> generator_;
    ChunkerConfig chunk_cfg_;
    HnswIndex index_;

    struct StoredChunk {
        std::string text;
        std::size_t source_idx;
    };
    std::vector<StoredChunk> chunks_;   ///< chunk_id == position here == index id.
    std::vector<std::string> sources_;
};

}  // namespace vecdb

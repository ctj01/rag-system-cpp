/**
 * @file rag_pipeline.cpp
 * @brief RAG orchestration implementation.
 */
#include "rag/rag_pipeline.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "core/dot.hpp"

namespace vecdb {

namespace {
bool has_signal(const std::vector<float>& v) {
    const float norm2 = dot_scalar(v, v);
    return norm2 > 1e-12f;
}
}  // namespace

RagPipeline::RagPipeline(std::unique_ptr<EmbeddingModel> embedder,
                         std::unique_ptr<Generator> generator,
                         ChunkerConfig chunk_cfg, HnswConfig index_cfg)
    // [C#→C++] Members initialize in the member-initializer list, and
    // std::move is required to transfer the unique_ptrs into the members —
    // copying them would not compile (move-only type).
    : embedder_(std::move(embedder)),
      generator_(std::move(generator)),
      chunk_cfg_(chunk_cfg),
      index_(embedder_ ? embedder_->dim() : 1, index_cfg) {
    if (!embedder_ || !generator_) {
        throw std::invalid_argument("RagPipeline: models must not be null");
    }
}

std::size_t RagPipeline::add_document(std::string_view text, std::string source) {
    const auto pieces = chunk_text(text, chunk_cfg_);
    if (pieces.empty()) return 0;

    sources_.push_back(std::move(source));
    const std::size_t source_idx = sources_.size() - 1;

    std::size_t indexed = 0;
    for (const auto& piece : pieces) {
        const auto embedding = embedder_->embed(piece);
        if (!has_signal(embedding)) continue;  // punctuation-only chunk etc.
        index_.add(embedding);
        // Index id and chunks_ position advance in lockstep: search results
        // map back to text by plain indexing.
        chunks_.push_back({piece, source_idx});
        ++indexed;
    }
    return indexed;
}

std::string RagPipeline::build_prompt(std::string_view question,
                                      const std::vector<RetrievedChunk>& hits) const {
    std::string p;
    p.reserve(512 + hits.size() * 600);
    p += "You are a retrieval-augmented assistant. Answer the question using "
         "ONLY the context passages below. If the answer is not contained in "
         "the context, say so plainly instead of guessing. Cite passages by "
         "their [number].\n\nContext:\n";
    for (std::size_t i = 0; i < hits.size(); ++i) {
        p += "[";
        p += std::to_string(i + 1);
        p += "] (source: ";
        p += hits[i].source;
        p += ")\n";
        p += hits[i].text;
        p += "\n\n";
    }
    p += "Question: ";
    p += question;
    p += "\nAnswer:";
    return p;
}

RagResponse RagPipeline::ask(std::string_view question, std::size_t k) const {
    if (chunks_.empty()) {
        throw std::invalid_argument("RagPipeline::ask: empty corpus");
    }
    const auto q_embedding = embedder_->embed(question);
    if (!has_signal(q_embedding)) {
        throw std::invalid_argument("RagPipeline::ask: question has no usable tokens");
    }

    const auto results = index_.search(q_embedding, std::min(k, chunks_.size()));

    RagResponse response;
    response.retrieved.reserve(results.size());
    for (const auto& r : results) {
        const StoredChunk& c = chunks_[r.id];
        response.retrieved.push_back(
            {r.id, r.score, c.text, sources_[c.source_idx]});
    }
    response.prompt = build_prompt(question, response.retrieved);
    response.answer = generator_->generate(response.prompt);
    return response;
}

}  // namespace vecdb

/**
 * @file test_rag.cpp
 * @brief Chunker and RAG pipeline tests (end-to-end with stub models).
 */
#include <stdexcept>
#include <string>

#include "models/embedding_model.hpp"
#include "models/generator.hpp"
#include "rag/chunker.hpp"
#include "rag/rag_pipeline.hpp"
#include "test_framework.hpp"

namespace {

std::string words(int n, const std::string& prefix) {
    std::string out;
    for (int i = 0; i < n; ++i) {
        out += prefix;
        out += std::to_string(i);
        out += ' ';
    }
    return out;
}

}  // namespace

int main() {
    // ---- chunker ----------------------------------------------------------
    {
        vecdb::ChunkerConfig cfg;
        cfg.max_words = 100;
        cfg.overlap_words = 20;

        // 300 words, stride 80: windows start at 0/80/160/240 -> 4 chunks.
        const auto chunks = vecdb::chunk_text(words(300, "w"), cfg);
        CHECK(chunks.size() == 4);
        // Overlap: chunk 0 ends with w99, chunk 1 starts at w80.
        CHECK(chunks[0].find("w99") != std::string::npos);
        CHECK(chunks[1].find("w80 ") == 0 || chunks[1].rfind("w80", 0) == 0);
        // Short text: single chunk, verbatim words.
        const auto single = vecdb::chunk_text("hello brave new world", cfg);
        CHECK(single.size() == 1);
        CHECK(single[0] == "hello brave new world");
        // No words at all.
        CHECK(vecdb::chunk_text("  \n\t  ", cfg).empty());
        // Bad config.
        vecdb::ChunkerConfig bad;
        bad.max_words = 10;
        bad.overlap_words = 10;
        CHECK_THROWS(vecdb::chunk_text("a b c", bad), std::invalid_argument);
    }

    // ---- pipeline end-to-end ----------------------------------------------
    {
        vecdb::RagPipeline pipeline(vecdb::make_hash_embedder(1024),
                                    vecdb::make_echo_generator());

        pipeline.add_document(
            "The Kalman filter estimates the hedge ratio dynamically for "
            "pairs trading. The spread z-score triggers entries beyond two "
            "standard deviations and exits at mean reversion.",
            "pairs.txt");
        pipeline.add_document(
            "Sourdough bread needs flour, water, salt and a mature starter. "
            "Fold the dough during bulk fermentation and bake in a hot dutch "
            "oven for a crisp crust.",
            "bread.txt");
        pipeline.add_document(
            "Boca Juniors played a tense final at La Bombonera and the "
            "midfield controlled the tempo of the match.",
            "football.txt");

        CHECK(pipeline.document_count() == 3);
        CHECK(pipeline.chunk_count() >= 3);

        const auto response = pipeline.ask("kalman filter hedge ratio", 2);

        // Retrieval: the pairs-trading chunk must win.
        CHECK(response.retrieved.size() == 2);
        CHECK(response.retrieved[0].source == "pairs.txt");
        CHECK(response.retrieved[0].score > response.retrieved[1].score);

        // Prompt assembly: grounding instructions, numbered passages,
        // evidence text and the question, all present.
        CHECK(response.prompt.find("[1] (source: pairs.txt)") != std::string::npos);
        CHECK(response.prompt.find("Kalman filter estimates") != std::string::npos);
        CHECK(response.prompt.find("Question: kalman filter hedge ratio") !=
              std::string::npos);
        // Echo generator: the answer IS the prompt (auditable pipeline).
        CHECK(response.answer.find(response.prompt) != std::string::npos);

        // k clamps to corpus size.
        CHECK(pipeline.ask("kalman", 50).retrieved.size() == pipeline.chunk_count());

        // Question with no usable tokens.
        CHECK_THROWS(pipeline.ask("... !!! ...", 2), std::invalid_argument);
    }

    // Empty corpus and null models.
    {
        vecdb::RagPipeline empty(vecdb::make_hash_embedder(256),
                                 vecdb::make_echo_generator());
        CHECK_THROWS(empty.ask("anything", 1), std::invalid_argument);
        CHECK_THROWS(vecdb::RagPipeline(nullptr, vecdb::make_echo_generator()),
                     std::invalid_argument);
    }

    return test_summary("test_rag");
}

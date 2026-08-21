/**
 * @file rag_demo.cpp
 * @brief Interactive RAG demo over a built-in sample corpus.
 *
 * Usage:
 *   rag_demo [--embed-model path.gguf] [--gen-model path.gguf] [question...]
 *
 *   rag_demo                          # stubs, interactive (empty line exits)
 *   rag_demo what is a hedge ratio    # stubs, one-shot
 *   rag_demo --embed-model minilm.gguf --gen-model qwen.gguf   # real models
 *
 * With the stub models the "answer" is the echoed prompt — the pipeline is
 * fully inspectable. With GGUF paths the same pipeline runs real local
 * inference through llama.cpp behind the same interfaces.
 */
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "models/embedding_model.hpp"
#include "models/generator.hpp"
#include "rag/rag_pipeline.hpp"
#ifdef VECDB_WITH_LLAMA
#include "models/llama_models.hpp"
#endif

namespace {

struct SampleDoc {
    const char* source;
    const char* text;
};

// Small built-in corpus: four documents on clearly distinct topics, so
// lexical retrieval (the hash-embedder stub) has real signal to work with.
constexpr SampleDoc kCorpus[] = {
    {"pairs_trading.txt",
     "Pairs trading is a market neutral strategy that exploits the relative "
     "mispricing between two cointegrated assets. The spread between the two "
     "legs is modeled as a mean reverting process. A Kalman filter can "
     "estimate the hedge ratio dynamically, treating it as a hidden state "
     "that evolves over time. Entry signals trigger when the z-score of the "
     "spread exceeds two standard deviations, and positions are closed when "
     "the spread reverts to its mean. Transaction costs are the main "
     "constraint on live profitability of the strategy."},
    {"options_pricing.txt",
     "The Black-Scholes model prices European options under the assumption "
     "of geometric Brownian motion and constant volatility. The Greeks "
     "measure sensitivities of the option price: delta to the underlying, "
     "gamma to delta itself, theta to time decay, vega to volatility and rho "
     "to interest rates. Implied volatility is the volatility that makes the "
     "model price match the market price, and its surface across strikes "
     "and maturities reveals where the model assumptions break."},
    {"hnsw_index.txt",
     "HNSW builds a multilayer navigable small world graph for approximate "
     "nearest neighbor search. Upper layers act as highways for greedy "
     "routing while layer zero holds every vector. The neighbor selection "
     "heuristic keeps edges directionally diverse, which preserves bridges "
     "between clusters. Recall is tuned at query time through the efSearch "
     "beam width parameter without rebuilding the graph."},
    {"swimming_notes.txt",
     "Freestyle swimming efficiency is measured with SWOLF, the sum of "
     "strokes and seconds per pool length. A long glide phase after each "
     "stroke reduces stroke count, and bilateral breathing every three "
     "strokes balances the body roll. Kick from the hips with relaxed "
     "ankles rather than from the knees."},
};

void print_response(const vecdb::RagResponse& response) {
    std::printf("\n-- retrieved --\n");
    for (std::size_t i = 0; i < response.retrieved.size(); ++i) {
        const auto& hit = response.retrieved[i];
        std::printf("[%zu] score=%.3f  source=%.*s\n", i + 1, hit.score,
                    static_cast<int>(hit.source.size()), hit.source.data());
    }
    std::printf("\n-- answer --\n%s\n\n", response.answer.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    // Minimal arg parsing: --embed-model / --gen-model, rest is the question.
    std::string embed_path, gen_path;
    std::vector<std::string> question_words;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--embed-model" && i + 1 < argc) {
            embed_path = argv[++i];
        } else if (arg == "--gen-model" && i + 1 < argc) {
            gen_path = argv[++i];
        } else {
            question_words.push_back(arg);
        }
    }

    std::unique_ptr<vecdb::EmbeddingModel> embedder;
    std::unique_ptr<vecdb::Generator> generator;
#ifdef VECDB_WITH_LLAMA
    if (!embed_path.empty()) {
        std::printf("loading embedding model: %s\n", embed_path.c_str());
        embedder = vecdb::make_llama_embedder(embed_path);
    }
    if (!gen_path.empty()) {
        std::printf("loading generation model: %s\n", gen_path.c_str());
        generator = vecdb::make_llama_generator(gen_path);
    }
#else
    if (!embed_path.empty() || !gen_path.empty()) {
        std::printf("built without VECDB_WITH_LLAMA; using stubs\n");
    }
#endif
    if (!embedder) embedder = vecdb::make_hash_embedder(1024);
    if (!generator) generator = vecdb::make_echo_generator();

    vecdb::RagPipeline pipeline(std::move(embedder), std::move(generator));

    for (const auto& doc : kCorpus) {
        const auto n = pipeline.add_document(doc.text, doc.source);
        std::printf("ingested %-22s -> %zu chunk(s)\n", doc.source, n);
    }
    std::printf("corpus ready: %zu documents, %zu chunks\n\n",
                pipeline.document_count(), pipeline.chunk_count());

    if (!question_words.empty()) {  // one-shot
        std::string question;
        for (std::size_t i = 0; i < question_words.size(); ++i) {
            if (i > 0) question += ' ';
            question += question_words[i];
        }
        print_response(pipeline.ask(question, 2));
        return 0;
    }

    std::printf("ask something (empty line to exit):\n");
    std::string line;
    while (std::printf("> "), std::getline(std::cin, line)) {
        if (line.empty()) break;
        try {
            print_response(pipeline.ask(line, 2));
        } catch (const std::exception& e) {
            // [C#→C++] catch by const reference, never by value — catching
            // by value slices a derived exception down to the base class.
            std::printf("error: %s\n", e.what());
        }
    }
    return 0;
}

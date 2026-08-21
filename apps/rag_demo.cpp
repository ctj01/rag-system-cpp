/**
 * @file rag_demo.cpp
 * @brief Interactive RAG demo over a built-in sample corpus (stub models).
 *
 * Usage:
 *   rag_demo                     # interactive: type questions, empty line exits
 *   rag_demo what is a hedge...  # one-shot: argv joined as the question
 *
 * With the stub models the "answer" is the echoed prompt — which makes the
 * whole pipeline inspectable: you see exactly the evidence and instructions
 * a real LLM would receive once llama.cpp lands behind the same interfaces.
 */
#include <cstdio>
#include <iostream>
#include <string>

#include "models/embedding_model.hpp"
#include "models/generator.hpp"
#include "rag/rag_pipeline.hpp"

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
    vecdb::RagPipeline pipeline(vecdb::make_hash_embedder(1024),
                                vecdb::make_echo_generator());

    for (const auto& doc : kCorpus) {
        const auto n = pipeline.add_document(doc.text, doc.source);
        std::printf("ingested %-22s -> %zu chunk(s)\n", doc.source, n);
    }
    std::printf("corpus ready: %zu documents, %zu chunks\n\n",
                pipeline.document_count(), pipeline.chunk_count());

    if (argc > 1) {  // one-shot: join argv into the question
        std::string question;
        for (int i = 1; i < argc; ++i) {
            if (i > 1) question += ' ';
            question += argv[i];
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

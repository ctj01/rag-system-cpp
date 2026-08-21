/**
 * @file llama_models.hpp
 * @brief llama.cpp-backed implementations of EmbeddingModel and Generator.
 *
 * Only available when the project is built with VECDB_WITH_LLAMA (the
 * default; llama.cpp is fetched and built from source, pinned tag). The RAG
 * pipeline is unchanged: these factories return the same interfaces the
 * stubs implement.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "models/embedding_model.hpp"
#include "models/generator.hpp"

namespace vecdb {

/// Tunables shared by both llama backends. Zero values mean "sensible default".
struct LlamaModelOptions {
    int n_threads = 0;             ///< 0 = half the hardware threads.
    std::uint32_t n_ctx = 0;       ///< 0 = model default (embedder) / 4096 (generator).
    int max_new_tokens = 256;      ///< Generator: completion length cap.
    float temperature = 0.0f;      ///< Generator: <= 0 means greedy decoding.
    std::uint32_t seed = 42;       ///< Generator sampling seed (when temperature > 0).
};

/**
 * @brief Loads a GGUF embedding model (e.g. all-MiniLM, nomic-embed).
 * @throws std::runtime_error if the file cannot be loaded.
 * @note embed() is serialized internally (one llama context); dim() comes
 *       from the model file.
 */
std::unique_ptr<EmbeddingModel> make_llama_embedder(const std::string& gguf_path,
                                                    LlamaModelOptions options = {});

/**
 * @brief Loads a GGUF chat/completion model (e.g. Qwen2.5-0.5B-Instruct).
 * @throws std::runtime_error if the file cannot be loaded.
 */
std::unique_ptr<Generator> make_llama_generator(const std::string& gguf_path,
                                                LlamaModelOptions options = {});

}  // namespace vecdb

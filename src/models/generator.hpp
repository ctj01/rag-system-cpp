/**
 * @file generator.hpp
 * @brief Abstract interface for text generation models.
 */
#pragma once

#include <memory>
#include <string>
#include <string_view>

namespace vecdb {

/**
 * @brief Interface for generation models: prompt in, completion out.
 *
 * Implementations: EchoGenerator (stub that reflects the prompt back) today;
 * a llama.cpp-backed model later. Kept synchronous and blocking on purpose —
 * streaming tokens can be layered on when a real backend exists.
 */
class Generator {
public:
    virtual ~Generator() = default;

    /**
     * @brief Generates a completion for @p prompt.
     * @param prompt Full prompt (instructions + retrieved context + query).
     * @return Generated text.
     */
    virtual std::string generate(std::string_view prompt) = 0;
};

/**
 * @brief Creates the stub generator, which echoes the prompt back verbatim
 *        prefixed with a marker — useful to inspect exactly what the RAG
 *        layer would send to a real LLM.
 */
std::unique_ptr<Generator> make_echo_generator();

}  // namespace vecdb

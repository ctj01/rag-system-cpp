/**
 * @file chunker.hpp
 * @brief Word-window document chunking with overlap.
 */
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace vecdb {

/**
 * @brief Chunking tunables.
 *
 * @details The central trade-off: small chunks give precise retrieval but
 * fragmented context; large chunks carry more context per hit but produce
 * diluted embeddings (one vector cannot point at fifteen topics). Overlap
 * (~10-25%) ensures an idea crossing a chunk boundary exists whole in at
 * least one chunk.
 */
struct ChunkerConfig {
    std::size_t max_words = 120;    ///< Window size, in whitespace-delimited words.
    std::size_t overlap_words = 30; ///< Words shared between consecutive chunks.
};

/**
 * @brief Splits text into overlapping word-window chunks.
 *
 * Words are whitespace-delimited; each chunk is a substring of the original
 * text (original spacing and punctuation preserved inside the window).
 *
 * @param text Input document.
 * @param cfg Window/overlap sizes; overlap must be < max_words.
 * @return Chunks in document order; empty if the text has no words.
 * @throws std::invalid_argument if cfg.overlap_words >= cfg.max_words.
 */
std::vector<std::string> chunk_text(std::string_view text, const ChunkerConfig& cfg = {});

}  // namespace vecdb

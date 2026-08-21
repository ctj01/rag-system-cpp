/**
 * @file chunker.cpp
 * @brief Word-window chunking implementation.
 */
#include "rag/chunker.hpp"

#include <cctype>
#include <stdexcept>

namespace vecdb {

namespace {
struct WordSpan {
    std::size_t begin;
    std::size_t end;  // one past the last character
};

bool is_space(unsigned char c) { return std::isspace(c) != 0; }
}  // namespace

std::vector<std::string> chunk_text(std::string_view text, const ChunkerConfig& cfg) {
    if (cfg.max_words == 0 || cfg.overlap_words >= cfg.max_words) {
        throw std::invalid_argument("chunk_text: overlap_words must be < max_words");
    }

    // Locate word boundaries once; chunks are then substrings of the
    // original text spanning whole words — no reassembly, no lost spacing.
    std::vector<WordSpan> words;
    std::size_t i = 0;
    const std::size_t n = text.size();
    while (i < n) {
        while (i < n && is_space(static_cast<unsigned char>(text[i]))) ++i;
        const std::size_t begin = i;
        while (i < n && !is_space(static_cast<unsigned char>(text[i]))) ++i;
        if (i > begin) words.push_back({begin, i});
    }
    if (words.empty()) return {};

    const std::size_t stride = cfg.max_words - cfg.overlap_words;
    std::vector<std::string> chunks;
    for (std::size_t start = 0;; start += stride) {
        const std::size_t end = std::min(start + cfg.max_words, words.size());
        const std::size_t from = words[start].begin;
        const std::size_t to = words[end - 1].end;
        // [C#→C++] substr on string_view is O(1) (adjusts pointer+length);
        // constructing std::string from it copies — the copy is deliberate
        // here, chunks must outlive the input text.
        chunks.emplace_back(text.substr(from, to - from));
        if (end == words.size()) break;
    }
    return chunks;
}

}  // namespace vecdb

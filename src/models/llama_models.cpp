/**
 * @file llama_models.cpp
 * @brief llama.cpp backends: GGUF embedding + generation behind the vecdb
 *        model interfaces.
 */
#include "models/llama_models.hpp"

#include <llama.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace vecdb {
namespace {

// ---- RAII over the C API ----------------------------------------------------
// [C#→C++] llama.cpp is a C library: raw pointers + explicit free functions
// (llama_model_free etc.) — the C++ idiom is to never hold those raw: wrap
// them in unique_ptr with a custom deleter type, and leaks become impossible
// by construction (destructor calls the right free, on every path including
// exceptions). This is what SafeHandle does in .NET, minus the GC.
struct ModelDeleter {
    void operator()(llama_model* m) const { llama_model_free(m); }
};
struct ContextDeleter {
    void operator()(llama_context* c) const { llama_free(c); }
};
struct SamplerDeleter {
    void operator()(llama_sampler* s) const { llama_sampler_free(s); }
};
using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

void quiet_log(ggml_log_level level, const char* text, void* /*user*/) {
    if (level >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

void ensure_backend() {
    // [C#→C++] "magic statics": initialization of a function-local static is
    // guaranteed thread-safe and runs exactly once — a built-in Lazy<T>.
    static const bool initialized = [] {
        llama_log_set(quiet_log, nullptr);  // model loading is very chatty
        llama_backend_init();
        return true;
    }();
    (void)initialized;
}

int default_threads(int requested) {
    if (requested > 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 1 ? static_cast<int>(hw / 2) : 1;
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, std::string_view text,
                                  bool add_special) {
    // First call with a null buffer returns -(needed token count).
    const int needed = -llama_tokenize(vocab, text.data(),
                                       static_cast<int32_t>(text.size()), nullptr, 0,
                                       add_special, true);
    if (needed <= 0) return {};
    std::vector<llama_token> tokens(static_cast<std::size_t>(needed));
    const int written = llama_tokenize(vocab, text.data(),
                                       static_cast<int32_t>(text.size()),
                                       tokens.data(), needed, add_special, true);
    tokens.resize(written > 0 ? static_cast<std::size_t>(written) : 0);
    return tokens;
}

ModelPtr load_model(const std::string& path) {
    ensure_backend();
    ModelPtr model(llama_model_load_from_file(path.c_str(), llama_model_default_params()));
    if (!model) {
        throw std::runtime_error("llama: failed to load GGUF model: " + path);
    }
    return model;
}

// ---- embedder ----------------------------------------------------------------

class LlamaEmbedder final : public EmbeddingModel {
public:
    LlamaEmbedder(const std::string& path, const LlamaModelOptions& options)
        : model_(load_model(path)) {
        auto params = llama_context_default_params();
        params.n_ctx = options.n_ctx;  // 0 = model default (512 for BERT-likes)
        params.n_threads = default_threads(options.n_threads);
        params.n_threads_batch = params.n_threads;
        params.embeddings = true;
        // Mean pooling over token embeddings -> one vector per sequence.
        params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        ctx_.reset(llama_init_from_model(model_.get(), params));
        if (!ctx_) throw std::runtime_error("llama: failed to create embedding context");

        // A pooled non-causal batch must fit in one physical ubatch, so the
        // token budget is min(context, batch sizes configured above).
        max_tokens_ = std::min(llama_n_ctx(ctx_.get()),
                               llama_context_default_params().n_batch);
        vocab_ = llama_model_get_vocab(model_.get());
        dim_ = static_cast<std::size_t>(llama_model_n_embd(model_.get()));
        encoder_only_ = llama_model_has_encoder(model_.get()) &&
                        !llama_model_has_decoder(model_.get());
    }

    std::size_t dim() const override { return dim_; }

    std::vector<float> embed(std::string_view text) const override {
        std::lock_guard<std::mutex> lock(mu_);  // one llama context = one caller

        auto tokens = tokenize(vocab_, text, /*add_special=*/true);
        if (tokens.empty()) return std::vector<float>(dim_, 0.0f);
        if (tokens.size() > max_tokens_) tokens.resize(max_tokens_);  // truncate

        // Fresh sequence per call: drop whatever the previous call left.
        llama_memory_clear(llama_get_memory(ctx_.get()), true);

        llama_batch batch =
            llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        const int rc = encoder_only_ ? llama_encode(ctx_.get(), batch)
                                     : llama_decode(ctx_.get(), batch);
        if (rc != 0) throw std::runtime_error("llama: embedding inference failed");

        const float* raw = llama_get_embeddings_seq(ctx_.get(), 0);
        if (!raw) throw std::runtime_error("llama: no pooled embedding produced");
        return std::vector<float>(raw, raw + dim_);
    }

private:
    ModelPtr model_;
    ContextPtr ctx_;
    const llama_vocab* vocab_ = nullptr;
    std::size_t dim_ = 0;
    std::size_t max_tokens_ = 0;
    bool encoder_only_ = false;
    mutable std::mutex mu_;
};

// ---- generator ---------------------------------------------------------------

class LlamaGenerator final : public Generator {
public:
    LlamaGenerator(const std::string& path, const LlamaModelOptions& options)
        : model_(load_model(path)), max_new_tokens_(options.max_new_tokens) {
        auto params = llama_context_default_params();
        // Cap the context: "model default" on modern LLMs can mean 128k,
        // whose KV cache would eat gigabytes for a demo.
        params.n_ctx = options.n_ctx != 0 ? options.n_ctx : 4096;
        params.n_threads = default_threads(options.n_threads);
        params.n_threads_batch = params.n_threads;
        ctx_.reset(llama_init_from_model(model_.get(), params));
        if (!ctx_) throw std::runtime_error("llama: failed to create generation context");
        vocab_ = llama_model_get_vocab(model_.get());

        sampler_.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        // Repeat penalty first in the chain: greedy decoding on small models
        // degenerates into loops without it (observed with Qwen2.5-0.5B).
        if (options.repeat_penalty > 1.0f && options.repeat_last_n > 0) {
            llama_sampler_chain_add(
                sampler_.get(),
                llama_sampler_init_penalties(llama_vocab_n_tokens(vocab_),
                                             options.repeat_last_n,
                                             options.repeat_penalty, 0.0f, 0.0f));
        }
        if (options.temperature > 0.0f) {
            llama_sampler_chain_add(sampler_.get(),
                                    llama_sampler_init_temp(options.temperature));
            llama_sampler_chain_add(sampler_.get(),
                                    llama_sampler_init_dist(options.seed));
        } else {
            // Greedy: deterministic completions, the right default for RAG
            // (we want the evidence to drive the answer, not the sampler).
            llama_sampler_chain_add(sampler_.get(), llama_sampler_init_greedy());
        }
        use_chat_template_ = options.use_chat_template &&
                             llama_model_chat_template(model_.get(), nullptr) != nullptr;
    }

    std::string generate(std::string_view prompt) override {
        std::lock_guard<std::mutex> lock(mu_);

        // Instruct models are trained INSIDE their chat template; a raw
        // completion prompt puts them out of distribution. Wrap the RAG
        // prompt as a single user turn when the model ships a template.
        std::string templated;
        bool add_special = true;
        if (use_chat_template_) {
            const std::string prompt_z(prompt);  // template API needs a C string
            const llama_chat_message msg{"user", prompt_z.c_str()};
            const char* tmpl = llama_model_chat_template(model_.get(), nullptr);
            std::vector<char> buf(prompt_z.size() * 2 + 512);
            int32_t n = llama_chat_apply_template(tmpl, &msg, 1, /*add_ass=*/true,
                                                  buf.data(),
                                                  static_cast<int32_t>(buf.size()));
            if (n > static_cast<int32_t>(buf.size())) {  // rare: template expands a lot
                buf.resize(static_cast<std::size_t>(n));
                n = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(),
                                              static_cast<int32_t>(buf.size()));
            }
            if (n > 0) {
                templated.assign(buf.data(), static_cast<std::size_t>(n));
                prompt = templated;
                // The template text already contains BOS/special markers:
                // adding another BOS would double it.
                add_special = false;
            }
        }

        auto tokens = tokenize(vocab_, prompt, add_special);
        if (tokens.empty()) return {};
        // Leave room in the context window for the completion.
        const std::size_t budget = llama_n_ctx(ctx_.get());
        const std::size_t reserve = static_cast<std::size_t>(max_new_tokens_);
        if (tokens.size() + reserve > budget && budget > reserve) {
            // Keep the TAIL: the question and the instructions closest to it
            // matter more than the start of the context when truncating.
            tokens.erase(tokens.begin(),
                         tokens.begin() +
                             static_cast<std::ptrdiff_t>(tokens.size() -
                                                         (budget - reserve)));
        }

        llama_memory_clear(llama_get_memory(ctx_.get()), true);

        std::string out;
        llama_batch batch =
            llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        for (int produced = 0; produced < max_new_tokens_; ++produced) {
            if (llama_decode(ctx_.get(), batch) != 0) {
                throw std::runtime_error("llama: decode failed");
            }
            llama_token token = llama_sampler_sample(sampler_.get(), ctx_.get(), -1);
            if (llama_vocab_is_eog(vocab_, token)) break;

            char piece[256];
            const int n = llama_token_to_piece(vocab_, token, piece,
                                               static_cast<int32_t>(sizeof(piece)), 0,
                                               /*special=*/false);
            if (n > 0) out.append(piece, static_cast<std::size_t>(n));

            last_token_ = token;
            batch = llama_batch_get_one(&last_token_, 1);
        }
        return out;
    }

private:
    ModelPtr model_;
    ContextPtr ctx_;
    SamplerPtr sampler_;
    const llama_vocab* vocab_ = nullptr;
    int max_new_tokens_;
    bool use_chat_template_ = false;
    llama_token last_token_ = 0;
    std::mutex mu_;
};

}  // namespace

std::unique_ptr<EmbeddingModel> make_llama_embedder(const std::string& gguf_path,
                                                    LlamaModelOptions options) {
    return std::make_unique<LlamaEmbedder>(gguf_path, options);
}

std::unique_ptr<Generator> make_llama_generator(const std::string& gguf_path,
                                                LlamaModelOptions options) {
    return std::make_unique<LlamaGenerator>(gguf_path, options);
}

}  // namespace vecdb

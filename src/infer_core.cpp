#include "infer.h"

#include <llama.h>
#include <ggml-backend.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t STATE_MAGIC = 0x494E4653; // "INFS"
constexpr uint32_t DEFAULT_BATCH_CAP = 512;
constexpr uint32_t DEFAULT_FALLBACK_CTX = 4096;
constexpr size_t   MAX_STATE_FILE_SIZE = 4ULL * 1024ULL * 1024ULL * 1024ULL; // 4GB sanity bound

/* Backend reference counter */
std::mutex g_backend_mutex;
int32_t    g_backend_ref_count = 0;

void ref_llama_backend() {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    if (g_backend_ref_count == 0) {
        ggml_backend_load_all();
        llama_backend_init();
    }
    g_backend_ref_count++;
}

void unref_llama_backend() {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    g_backend_ref_count--;
    if (g_backend_ref_count <= 0) {
        llama_backend_free();
        g_backend_ref_count = 0;
    }
}

/* Validates the largest prefix of a string that forms valid, unclipped UTF-8 */
size_t find_valid_utf8_end(const std::string& s) {
    if (s.empty()) return 0;
    for (size_t back = 1; back <= std::min<size_t>(4, s.size()); ++back) {
        uint8_t b = static_cast<uint8_t>(s[s.size() - back]);
        if ((b & 0x80) == 0x00) {
            return s.size(); // Suffix ends on ASCII
        } else if ((b & 0xC0) == 0xC0) {
            size_t expected = ((b & 0xE0) == 0xC0) ? 2 : (((b & 0xF0) == 0xE0) ? 3 : (((b & 0xF8) == 0xF0) ? 4 : 1));
            if (back == expected) {
                return s.size(); // Complete multibyte character
            } else if (back < expected) {
                return s.size() - back; // Partial multibyte sequence
            } else {
                return s.size();
            }
        }
    }
    return s.size();
}

/* Decides if a prompt needs an explicit BOS token prepended */
bool want_bos(const struct llama_vocab* v, const char* text) {
    if (!v || !text || !llama_vocab_get_add_bos(v)) return false;
    llama_token bos_id = llama_vocab_bos(v);
    if (bos_id == LLAMA_TOKEN_NULL) return false;
    const char* bos = llama_vocab_get_text(v, bos_id);
    return !(bos && std::strncmp(text, bos, std::strlen(bos)) == 0);
}

/* Streaming buffer: manages UTF-8 alignment, hold-back window, and stop sequences */
class StreamOutputBuffer {
public:
    StreamOutputBuffer(infer_stream_cb cb, void* udata, const std::vector<std::string>& stops)
        : callback_(cb), user_data_(udata), stop_words_(stops) {
        for (const auto& w : stop_words_) {
            if (w.size() > max_stop_len_) {
                max_stop_len_ = w.size();
            }
        }
    }

    bool append_and_flush(const char* piece, size_t len) {
        buffer_.append(piece, len);

        // 1. Check for earliest stop word occurrence
        size_t earliest_pos = std::string::npos;
        for (const auto& stop : stop_words_) {
            if (stop.empty()) continue;
            size_t pos = buffer_.find(stop);
            if (pos != std::string::npos) {
                if (earliest_pos == std::string::npos || pos < earliest_pos) {
                    earliest_pos = pos;
                }
            }
        }

        if (earliest_pos != std::string::npos) {
            hit_stop_ = true;
            if (earliest_pos > 0) {
                std::string pre = buffer_.substr(0, earliest_pos);
                size_t valid = find_valid_utf8_end(pre);
                if (valid > 0) {
                    callback_(pre.substr(0, valid).c_str(), user_data_);
                }
            }
            buffer_.clear();
            return false;
        }

        // 2. Hold back candidate trailing characters for incomplete stop sequences
        size_t keep_len = (max_stop_len_ > 1) ? (max_stop_len_ - 1) : 0;
        if (buffer_.size() <= keep_len) {
            return true;
        }

        size_t candidate = buffer_.size() - keep_len;
        // Never slice across a UTF-8 continuation byte
        while (candidate > 0 && candidate < buffer_.size() &&
               (static_cast<uint8_t>(buffer_[candidate]) & 0xC0) == 0x80) {
            candidate--;
        }

        std::string candidate_slice = buffer_.substr(0, candidate);
        size_t flushable = find_valid_utf8_end(candidate_slice);

        if (flushable > 0) {
            std::string to_emit = buffer_.substr(0, flushable);
            buffer_.erase(0, flushable);
            if (!callback_(to_emit.c_str(), user_data_)) {
                return false;
            }
        }

        return true;
    }

    bool flush_remaining() {
        if (hit_stop_) return true;

        size_t earliest_pos = std::string::npos;
        for (const auto& stop : stop_words_) {
            if (stop.empty()) continue;
            size_t pos = buffer_.find(stop);
            if (pos != std::string::npos) {
                if (earliest_pos == std::string::npos || pos < earliest_pos) {
                    earliest_pos = pos;
                }
            }
        }

        if (earliest_pos != std::string::npos) {
            hit_stop_ = true;
            if (earliest_pos > 0) {
                std::string pre = buffer_.substr(0, earliest_pos);
                size_t valid = find_valid_utf8_end(pre);
                if (valid > 0) {
                    callback_(pre.substr(0, valid).c_str(), user_data_);
                }
            }
            buffer_.clear();
            return true;
        }

        if (!buffer_.empty()) {
            size_t valid = find_valid_utf8_end(buffer_);
            if (valid > 0) {
                callback_(buffer_.substr(0, valid).c_str(), user_data_);
            }
            buffer_.clear();
        }
        return true;
    }

    bool hit_stop() const { return hit_stop_; }

private:
    infer_stream_cb callback_;
    void* user_data_;
    const std::vector<std::string>& stop_words_;
    size_t max_stop_len_ = 0;
    std::string buffer_;
    bool hit_stop_ = false;
};

#pragma pack(push, 1)
struct StateFileHeader {
    uint32_t magic;
    uint32_t n_ctx;
    uint32_t n_past;
    uint32_t vocab_size;
    uint64_t data_size;
};
#pragma pack(pop)

} // anonymous namespace

/* Internal context definition */
struct infer_ctx {
    struct llama_model*   model   = nullptr;
    struct llama_context* lctx    = nullptr;
    struct llama_sampler* sampler = nullptr;

    infer_sampler_params     sampler_params{};
    std::string              grammar_gbnf;
    std::vector<std::string> stop_words;
    uint32_t                 n_ctx = 0;

    std::atomic<int> n_past{0};

    std::atomic<bool> abort_flag{false};
    std::atomic<bool> is_generating{false};
    std::thread       async_worker;
    mutable std::mutex worker_mutex;
    mutable std::mutex ctx_mutex;
    std::string        last_error;

    void set_error_locked(const std::string& err) {
        last_error = err;
    }

    void set_error(const std::string& err) {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        last_error = err;
    }

    void clear_error() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        last_error.clear();
    }
};

struct OwnedMsg {
    std::string role;
    std::string content;
};

/* RAII Guard to claim exclusive operation status */
struct GenAcquireGuard {
    infer_ctx* ctx = nullptr;
    bool acquired = false;

    explicit GenAcquireGuard(infer_ctx* c) : ctx(c) {
        if (!ctx) return;
        bool expected = false;
        if (ctx->is_generating.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            acquired = true;
        }
    }

    ~GenAcquireGuard() {
        if (acquired && ctx) {
            ctx->is_generating.store(false, std::memory_order_release);
        }
    }
};

/* RAII Guard for Prompt Prefill Rollback */
struct PrefillRollbackGuard {
    struct llama_context* lctx;
    int start_past;
    bool completed = false;

    ~PrefillRollbackGuard() {
        if (!completed && lctx) {
            llama_memory_seq_rm(llama_get_memory(lctx), 0, start_past, -1);
        }
    }
};

static ggml_backend_dev_t infer_get_optimal_device() {
    size_t dev_count = ggml_backend_dev_count();
    ggml_backend_dev_t fallback_dev = nullptr;

    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return dev;
        }
        if (!fallback_dev) {
            fallback_dev = dev;
        }
    }
    return fallback_dev;
}

static uint32_t calculate_safe_n_ctx(struct llama_model* model, uint32_t requested_ctx) {
    uint32_t max_train = static_cast<uint32_t>(llama_model_n_ctx_train(model));
    uint32_t target_ctx = (requested_ctx > 0) ? requested_ctx : DEFAULT_FALLBACK_CTX;
    if (max_train > 0) {
        target_ctx = std::min<uint32_t>(target_ctx, max_train);
    }

    int32_t n_layer   = llama_model_n_layer(model);
    int32_t n_head_kv = llama_model_n_head_kv(model);
    int32_t n_head    = llama_model_n_head(model);
    int32_t n_embd    = llama_model_n_embd(model);

    if (n_head <= 0 || n_layer <= 0 || n_head_kv <= 0 || n_embd <= 0) {
        return target_ctx;
    }

    int32_t head_dim = n_embd / n_head;
    constexpr size_t bytes_per_element = 2; // GGML_TYPE_F16
    size_t bytes_per_token = static_cast<size_t>(2) * n_layer * n_head_kv * head_dim * bytes_per_element;

    ggml_backend_dev_t dev = infer_get_optimal_device();
    size_t free_vram = 0, total_vram = 0; // Handled correctly in compilation

    if (dev) {
        ggml_backend_dev_memory(dev, &free_vram, &total_vram);
    }

    if (free_vram == 0 || bytes_per_token == 0) {
        return target_ctx;
    }

    constexpr size_t headroom = 128ULL * 1024ULL * 1024ULL;
    if (free_vram <= headroom) {
        return std::min<uint32_t>(target_ctx, 512);
    }

    size_t available_vram = free_vram - headroom;
    uint32_t safe_tokens = static_cast<uint32_t>(available_vram / bytes_per_token);

    // Keep within user request ceiling without arbitrary inflation
    return std::max<uint32_t>(1, std::min<uint32_t>(target_ctx, safe_tokens));
}

static struct llama_sampler* create_sampler(
    const infer_sampler_params& params,
    struct llama_model* model,
    const std::string& grammar_gbnf,
    std::string& err_out
) {
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* chain = llama_sampler_chain_init(sparams);
    if (!chain) return nullptr;

    const struct llama_vocab* vocab = model ? llama_model_get_vocab(model) : nullptr;

    // 1. Repetition penalties
    if (params.repeat_penalty > 0.0f && params.repeat_penalty != 1.0f && vocab) {
        int32_t last_n = (params.repeat_last_n > 0) ? params.repeat_last_n : 64;
        int32_t n_vocab = llama_vocab_n_tokens(vocab);
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, last_n, params.repeat_penalty, 0.0f, 0.0f));
    }

    // 2. Grammar sampler: must precede distribution truncation
    if (!grammar_gbnf.empty() && vocab) {
        struct llama_sampler* gsampler = llama_sampler_init_grammar(vocab, grammar_gbnf.c_str(), "root");
        if (gsampler) {
            llama_sampler_chain_add(chain, gsampler);
        } else {
            err_out = "Failed to parse grammar rules.";
            llama_sampler_free(chain);
            return nullptr;
        }
    }

    // 3. Distribution filtering
    if (params.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.min_p > 0.0f && params.min_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(params.min_p, 1));
    }

    // 4. Softmax / Greedy temperature
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed ? params.seed : LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }

    return chain;
}

/* ========================================================================== */
/* Unified Generation Engine                                                  */
/* ========================================================================== */

static bool execute_decode(
    infer_ctx* ctx,
    const std::vector<llama_token>& prompt_tokens,
    bool append_to_existing_past,
    infer_stream_cb callback,
    void* user_data
) {
    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);

    // Multi-turn context shift if turn prompt exceeds remaining space
    if (append_to_existing_past) {
        if (static_cast<uint32_t>(prompt_tokens.size()) >= ctx->n_ctx) {
            ctx->set_error("Prompt tokens exceed context capacity.");
            return false;
        }
        int current_past = ctx->n_past.load(std::memory_order_relaxed);
        while (static_cast<uint32_t>(current_past + prompt_tokens.size()) >= ctx->n_ctx) {
            if (!llama_memory_can_shift(llama_get_memory(ctx->lctx))) {
                ctx->set_error("Context full and KV cache does not support shifting.");
                return false;
            }
            int n_keep = (current_past > 64) ? 64 : 1;
            int needed = static_cast<int>(current_past + prompt_tokens.size() - ctx->n_ctx + 16);
            int n_discard = std::max(needed, (static_cast<int>(ctx->n_ctx) - n_keep) / 2);
            if (current_past - n_discard <= n_keep) {
                n_discard = current_past - n_keep;
            }
            if (n_discard <= 0) break;
            llama_memory_seq_rm(llama_get_memory(ctx->lctx), 0, n_keep, n_keep + n_discard);
            llama_memory_seq_add(llama_get_memory(ctx->lctx), 0, n_keep + n_discard, current_past, -n_discard);
            current_past -= n_discard;
            ctx->n_past.store(current_past, std::memory_order_relaxed);
        }
    } else {
        llama_memory_clear(llama_get_memory(ctx->lctx), true);
        ctx->n_past.store(0, std::memory_order_relaxed);
    }

    if (ctx->sampler) {
        llama_sampler_reset(ctx->sampler);
    }

    int start_past = append_to_existing_past ? ctx->n_past.load(std::memory_order_relaxed) : 0;

    if (static_cast<uint32_t>(start_past + prompt_tokens.size()) >= ctx->n_ctx) {
        ctx->set_error("Prompt tokens exceed context capacity.");
        return false;
    }

    struct llama_batch batch = llama_batch_init(DEFAULT_BATCH_CAP, 0, 1);
    struct BatchGuard {
        struct llama_batch& b;
        ~BatchGuard() { llama_batch_free(b); }
    } batch_guard{batch};

    // RAII guard to clean up sequence 0 on prefill abort or failure
    PrefillRollbackGuard prefill_guard{ctx->lctx, start_past};

    // Prefill prompt in chunks
    for (size_t i = 0; i < prompt_tokens.size(); i += batch.n_tokens) {
        if (ctx->abort_flag.load(std::memory_order_relaxed)) {
            return false;
        }

        batch.n_tokens = 0;
        int chunk = std::min<int>(DEFAULT_BATCH_CAP, static_cast<int>(prompt_tokens.size() - i));

        for (int j = 0; j < chunk; ++j) {
            size_t idx = i + j;
            batch.token[j]     = prompt_tokens[idx];
            batch.pos[j]       = static_cast<llama_pos>(start_past + idx);
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = (idx == prompt_tokens.size() - 1);
            batch.n_tokens++;
        }

        if (llama_decode(ctx->lctx, batch) != 0) {
            ctx->set_error("llama_decode failed during prompt prefill.");
            return false;
        }
    }

    // Prefill succeeded cleanly
    prefill_guard.completed = true;

    int n_cur = start_past + static_cast<int>(prompt_tokens.size());
    ctx->n_past.store(n_cur, std::memory_order_relaxed);

    std::vector<std::string> active_stops;
    {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        active_stops = ctx->stop_words;
    }

    StreamOutputBuffer out_buf(callback, user_data, active_stops);
    std::vector<char> piece_buf(256);

    // Generation loop
    while (true) {
        if (ctx->abort_flag.load(std::memory_order_relaxed)) break;

        // Context shift if near window boundary
        if (n_cur >= static_cast<int>(ctx->n_ctx) - 4) {
            if (!llama_memory_can_shift(llama_get_memory(ctx->lctx))) {
                ctx->set_error("Context full and KV cache does not support shifting.");
                return false;
            }

            int n_keep = (n_cur > 64) ? 64 : 1;
            int n_discard = (ctx->n_ctx - n_keep) / 2;
            if (n_discard <= 0) n_discard = 1;

            llama_memory_seq_rm(llama_get_memory(ctx->lctx), 0, n_keep, n_keep + n_discard);
            llama_memory_seq_add(llama_get_memory(ctx->lctx), 0, n_keep + n_discard, n_cur, -n_discard);
            n_cur -= n_discard;
            ctx->n_past.store(n_cur, std::memory_order_relaxed);
        }

        llama_token new_token_id = llama_sampler_sample(ctx->sampler, ctx->lctx, -1);
        llama_sampler_accept(ctx->sampler, new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            // Append terminal EOG into KV cache
            batch.n_tokens = 0;
            batch.token[0]     = new_token_id;
            batch.pos[0]       = static_cast<llama_pos>(n_cur);
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = false;
            batch.n_tokens     = 1;
            if (llama_decode(ctx->lctx, batch) == 0) {
                n_cur++;
            }
            break;
        }

        int n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf.data(), piece_buf.size(), 0, false);
        if (n_chars < 0) {
            piece_buf.resize(-n_chars + 1);
            n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf.data(), piece_buf.size(), 0, false);
        }

        if (n_chars > 0) {
            if (!out_buf.append_and_flush(piece_buf.data(), static_cast<size_t>(n_chars))) {
                if (!out_buf.hit_stop()) {
                    ctx->abort_flag.store(true, std::memory_order_release);
                }
                // Append EOS token so subsequent turns retain valid boundaries
                llama_token eos_token = llama_vocab_eos(vocab);
                if (eos_token != LLAMA_TOKEN_NULL) {
                    batch.n_tokens = 0;
                    batch.token[0]     = eos_token;
                    batch.pos[0]       = static_cast<llama_pos>(n_cur);
                    batch.n_seq_id[0]  = 1;
                    batch.seq_id[0][0] = 0;
                    batch.logits[0]    = false;
                    batch.n_tokens     = 1;
                    if (llama_decode(ctx->lctx, batch) == 0) {
                        n_cur++;
                    }
                }
                break;
            }
        }

        batch.n_tokens = 0;
        batch.token[0]     = new_token_id;
        batch.pos[0]       = static_cast<llama_pos>(n_cur);
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = true;
        batch.n_tokens     = 1;

        if (llama_decode(ctx->lctx, batch) != 0) {
            ctx->set_error("llama_decode failed during token generation.");
            ctx->n_past.store(n_cur, std::memory_order_relaxed);
            return false;
        }

        n_cur++;
        ctx->n_past.store(n_cur, std::memory_order_relaxed);
    }

    out_buf.flush_remaining();
    ctx->n_past.store(n_cur, std::memory_order_relaxed);
    return true;
}

static bool infer_generate_chat_internal(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    infer_stream_cb callback,
    void* user_data
) {
    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    const char* chat_template = llama_model_chat_template(ctx->model, nullptr);

    std::vector<struct llama_chat_message> llama_msgs;
    llama_msgs.reserve(num_msgs);
    for (size_t i = 0; i < num_msgs; ++i) {
        llama_msgs.push_back({ msgs[i].role, msgs[i].content });
    }

    int alloc_size = llama_chat_apply_template(chat_template, llama_msgs.data(), llama_msgs.size(), true, nullptr, 0);
    if (alloc_size <= 0) {
        ctx->set_error("Chat template formatting failed.");
        return false;
    }

    std::vector<char> formatted(alloc_size + 1, '\0');
    int written = llama_chat_apply_template(chat_template, llama_msgs.data(), llama_msgs.size(), true, formatted.data(), formatted.size());
    if (written < 0) {
        ctx->set_error("Chat template failed to render prompt.");
        return false;
    }

    bool add_bos = want_bos(vocab, formatted.data());
    int n_prompt_tokens = -llama_tokenize(vocab, formatted.data(), written, nullptr, 0, add_bos, true);
    if (n_prompt_tokens <= 0) {
        ctx->set_error("Tokenization calculation failed.");
        return false;
    }

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    int real_tokens = llama_tokenize(vocab, formatted.data(), written, prompt_tokens.data(), prompt_tokens.size(), add_bos, true);
    if (real_tokens < 0) {
        ctx->set_error("Failed to tokenize rendered chat prompt.");
        return false;
    }
    prompt_tokens.resize(real_tokens);

    return execute_decode(ctx, prompt_tokens, false, callback, user_data);
}

/* ========================================================================== */
/* Public API Implementation                                                  */
/* ========================================================================== */

extern "C" {

infer_sampler_params infer_sampler_default_params(void) {
    return { 0.70f, 0.90f, 40, 0.05f, 1.10f, 64, 0 };
}

infer_ctx* infer_init(void) {
    infer_ctx* ctx = nullptr;
    try {
        ref_llama_backend();
        ctx = new (std::nothrow) infer_ctx();
        if (!ctx) {
            unref_llama_backend();
            return nullptr;
        }
        ctx->sampler_params = infer_sampler_default_params();
        return ctx;
    } catch (...) {
        if (ctx) delete ctx;
        unref_llama_backend();
        return nullptr;
    }
}

bool infer_load_model(infer_ctx* ctx, const char* path, uint32_t requested_ctx) {
    if (!ctx || !path) return false;

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot load model while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        ctx->last_error.clear();

        if (ctx->sampler) { llama_sampler_free(ctx->sampler); ctx->sampler = nullptr; }
        if (ctx->lctx)    { llama_free(ctx->lctx); ctx->lctx = nullptr; }
        if (ctx->model)   { llama_model_free(ctx->model); ctx->model = nullptr; }

        struct llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 999;

        ctx->model = llama_model_load_from_file(path, model_params);
        if (!ctx->model) {
            ctx->set_error_locked("Failed to load model from path: " + std::string(path));
            return false;
        }

        ctx->n_ctx = calculate_safe_n_ctx(ctx->model, requested_ctx);

        struct llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx   = ctx->n_ctx;
        ctx_params.n_batch = std::min<uint32_t>(ctx->n_ctx, DEFAULT_BATCH_CAP);

        ctx->lctx = llama_init_from_model(ctx->model, ctx_params);
        if (!ctx->lctx) {
            llama_model_free(ctx->model);
            ctx->model = nullptr;
            ctx->set_error_locked("Failed to initialize llama_context.");
            return false;
        }

        std::string err;
        ctx->sampler = create_sampler(ctx->sampler_params, ctx->model, ctx->grammar_gbnf, err);
        if (!ctx->sampler) {
            ctx->set_error_locked(err);
            llama_free(ctx->lctx); ctx->lctx = nullptr;
            llama_model_free(ctx->model); ctx->model = nullptr;
            return false;
        }

        ctx->n_past.store(0, std::memory_order_relaxed);
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(std::string("Exception in infer_load_model: ") + e.what());
        return false;
    }
}

bool infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params) {
    if (!ctx || !params) return false;

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot change sampler while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

        // Transactional build: verify before replacing active state
        std::string err;
        struct llama_sampler* new_sampler = create_sampler(*params, ctx->model, ctx->grammar_gbnf, err);
        if (!new_sampler) {
            ctx->set_error_locked(err);
            return false;
        }

        if (ctx->sampler) {
            llama_sampler_free(ctx->sampler);
        }
        ctx->sampler = new_sampler;
        ctx->sampler_params = *params;
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    }
}

bool infer_get_model_info(infer_ctx* ctx, infer_model_info* out_info) {
    if (!ctx || !out_info) return false;
    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->model) return false;

        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        out_info->n_ctx_train = static_cast<uint32_t>(llama_model_n_ctx_train(ctx->model));
        out_info->n_embd      = static_cast<uint32_t>(llama_model_n_embd(ctx->model));
        out_info->n_layer     = static_cast<uint32_t>(llama_model_n_layer(ctx->model));
        out_info->n_head      = static_cast<uint32_t>(llama_model_n_head(ctx->model));
        out_info->vocab_size  = vocab ? static_cast<uint32_t>(llama_vocab_n_tokens(vocab)) : 0;

        std::memset(out_info->description, 0, sizeof(out_info->description));
        llama_model_desc(ctx->model, out_info->description, sizeof(out_info->description));
        return true;
    } catch (...) {
        return false;
    }
}

size_t infer_get_last_error(infer_ctx* ctx, char* out_buf, size_t buf_size) {
    if (!ctx) return 0;
    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!out_buf || buf_size == 0) {
            return ctx->last_error.size();
        }
        size_t copy_len = std::min(buf_size - 1, ctx->last_error.size());
        std::memcpy(out_buf, ctx->last_error.c_str(), copy_len);
        out_buf[copy_len] = '\0';
        return copy_len;
    } catch (...) {
        return 0;
    }
}

int32_t infer_tokenize_count(infer_ctx* ctx, const char* text, bool add_special) {
    if (!ctx || !text) return 0;
    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->model) return 0;
        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        if (!vocab) return 0;

        int n = llama_tokenize(vocab, text, static_cast<int32_t>(std::strlen(text)), nullptr, 0, add_special, false);
        return (n < 0) ? -n : n;
    } catch (...) {
        return 0;
    }
}

uint32_t infer_get_kv_cache_token_count(infer_ctx* ctx) {
    if (!ctx) return 0;
    try {
        return static_cast<uint32_t>(ctx->n_past.load(std::memory_order_relaxed));
    } catch (...) {
        return 0;
    }
}

bool infer_set_stop_sequences(infer_ctx* ctx, const char* const* stop_words, size_t num_words) {
    if (!ctx) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot change stop sequences while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        ctx->stop_words.clear();
        if (stop_words && num_words > 0) {
            for (size_t i = 0; i < num_words; ++i) {
                if (stop_words[i] && std::strlen(stop_words[i]) > 0) {
                    ctx->stop_words.emplace_back(stop_words[i]);
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    }
}

bool infer_set_grammar(infer_ctx* ctx, const char* grammar_gbnf) {
    if (!ctx) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot change grammar while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        std::string new_grammar = grammar_gbnf ? grammar_gbnf : "";

        // Transactional build: test before applying
        std::string err;
        struct llama_sampler* new_sampler = create_sampler(ctx->sampler_params, ctx->model, new_grammar, err);
        if (!new_sampler) {
            ctx->set_error_locked(err);
            return false;
        }

        if (ctx->sampler) {
            llama_sampler_free(ctx->sampler);
        }
        ctx->sampler = new_sampler;
        ctx->grammar_gbnf = std::move(new_grammar);
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    }
}

void infer_memory_clear(infer_ctx* ctx) {
    if (!ctx) return;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot clear memory while context is busy.");
        return;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (ctx->lctx) {
            llama_memory_clear(llama_get_memory(ctx->lctx), true);
        }
        if (ctx->sampler) {
            llama_sampler_reset(ctx->sampler);
        }
        ctx->n_past.store(0, std::memory_order_relaxed);
    } catch (...) {
        // Suppress exceptions in C API
    }
}

bool infer_state_save(infer_ctx* ctx, const char* filepath) {
    if (!ctx || !ctx->lctx || !ctx->model || !filepath) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot save state while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        size_t state_size = llama_state_get_size(ctx->lctx);
        if (state_size == 0 || state_size > MAX_STATE_FILE_SIZE) {
            ctx->set_error_locked("Invalid context state size.");
            return false;
        }

        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        uint32_t vocab_size = vocab ? static_cast<uint32_t>(llama_vocab_n_tokens(vocab)) : 0;

        std::vector<uint8_t> buffer(state_size);
        size_t written = llama_state_get_data(ctx->lctx, buffer.data(), buffer.size());
        if (written != state_size) {
            ctx->set_error_locked("Failed to extract context state data.");
            return false;
        }

        FILE* f = std::fopen(filepath, "wb");
        if (!f) {
            ctx->set_error_locked("Failed to open file for state saving: " + std::string(filepath));
            return false;
        }

        StateFileHeader header{};
        header.magic      = STATE_MAGIC;
        header.n_ctx      = ctx->n_ctx;
        header.n_past     = static_cast<uint32_t>(ctx->n_past.load(std::memory_order_relaxed));
        header.vocab_size = vocab_size;
        header.data_size  = static_cast<uint64_t>(state_size);

        bool success = (std::fwrite(&header, sizeof(header), 1, f) == 1) &&
                       (std::fwrite(buffer.data(), 1, buffer.size(), f) == buffer.size());

        if (std::fclose(f) != 0 || !success) {
            ctx->set_error_locked("Failed to write state file cleanly.");
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    }
}

bool infer_state_load(infer_ctx* ctx, const char* filepath) {
    if (!ctx || !ctx->lctx || !ctx->model || !filepath) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot load state while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        FILE* f = std::fopen(filepath, "rb");
        if (!f) {
            ctx->set_error_locked("Failed to open file for state loading: " + std::string(filepath));
            return false;
        }

        StateFileHeader header{};
        if (std::fread(&header, sizeof(header), 1, f) != 1) {
            std::fclose(f);
            ctx->set_error_locked("Corrupt or incomplete state file header.");
            return false;
        }

        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        uint32_t current_vocab_size = vocab ? static_cast<uint32_t>(llama_vocab_n_tokens(vocab)) : 0;
        size_t expected_state_size = llama_state_get_size(ctx->lctx);

        if (header.magic != STATE_MAGIC ||
            header.n_ctx != ctx->n_ctx ||
            header.n_past > ctx->n_ctx ||
            header.vocab_size != current_vocab_size ||
            header.data_size != expected_state_size ||
            header.data_size > MAX_STATE_FILE_SIZE) {
            std::fclose(f);
            ctx->set_error_locked("Incompatible or corrupted state file header.");
            return false;
        }

        std::vector<uint8_t> buffer(header.data_size);
        if (std::fread(buffer.data(), 1, header.data_size, f) != header.data_size) {
            std::fclose(f);
            ctx->set_error_locked("Failed to read complete state data from file.");
            return false;
        }
        std::fclose(f);

        size_t loaded = llama_state_set_data(ctx->lctx, buffer.data(), buffer.size());
        if (loaded != header.data_size) {
            llama_memory_clear(llama_get_memory(ctx->lctx), true);
            if (ctx->sampler) llama_sampler_reset(ctx->sampler);
            ctx->n_past.store(0, std::memory_order_relaxed);
            ctx->set_error_locked("llama_state_set_data failed to restore context state.");
            return false;
        }

        ctx->n_past.store(static_cast<int>(header.n_past), std::memory_order_relaxed);
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    }
}

int32_t infer_get_embedding_dim(infer_ctx* ctx) {
    if (!ctx) return 0;
    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->model) return 0;
        return llama_model_n_embd(ctx->model);
    } catch (...) {
        return 0;
    }
}

bool infer_get_embeddings(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len) {
    if (!ctx || !text || !out_vec) return false;

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Context is busy.");
        return false;
    }

    ctx->abort_flag.store(false, std::memory_order_release);

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->model || !ctx->lctx) return false;
        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        if (!vocab) return false;

        int n_embd = llama_model_n_embd(ctx->model);
        if (vec_len < static_cast<size_t>(n_embd)) {
            ctx->set_error_locked("Output buffer smaller than model embedding dimensions.");
            return false;
        }

        bool add_bos = want_bos(vocab, text);
        int n_tokens = -llama_tokenize(vocab, text, static_cast<int32_t>(std::strlen(text)), nullptr, 0, add_bos, false);
        if (n_tokens <= 0) return false;

        std::vector<llama_token> tokens(n_tokens);
        int real_tokens = llama_tokenize(vocab, text, static_cast<int32_t>(std::strlen(text)), tokens.data(), tokens.size(), add_bos, false);
        if (real_tokens < 0) return false;
        tokens.resize(real_tokens);

        if (static_cast<uint32_t>(tokens.size()) > ctx->n_ctx) {
            ctx->set_error_locked("Input text exceeds context capacity for embedding generation.");
            return false;
        }

        // Switch to embedding decode mode with RAII restore
        llama_set_embeddings(ctx->lctx, true);
        struct EmbeddingModeGuard {
            struct llama_context* l;
            ~EmbeddingModeGuard() {
                llama_set_embeddings(l, false);
                // Remove sequence 1 from KV cache without disturbing sequence 0 (chat KV cache)
                llama_memory_seq_rm(llama_get_memory(l), 1, -1, -1);
            }
        } embd_guard{ctx->lctx};

        int batch_cap = std::min<int>(DEFAULT_BATCH_CAP, static_cast<int>(ctx->n_ctx));
        struct llama_batch batch = llama_batch_init(batch_cap, 0, 1);
        struct BatchGuard {
            struct llama_batch& b;
            ~BatchGuard() { llama_batch_free(b); }
        } guard{batch};

        for (size_t i = 0; i < tokens.size(); i += batch.n_tokens) {
            if (ctx->abort_flag.load(std::memory_order_relaxed)) {
                ctx->set_error_locked("Embedding generation aborted.");
                return false;
            }

            batch.n_tokens = 0;
            int chunk = std::min<int>(batch_cap, static_cast<int>(tokens.size() - i));

            for (int j = 0; j < chunk; ++j) {
                size_t idx = i + j;
                batch.token[j]     = tokens[idx];
                batch.pos[j]       = static_cast<llama_pos>(idx);
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 1; // Sequence 1 to preserve Sequence 0 chat cache
                batch.logits[j]    = (idx == tokens.size() - 1);
                batch.n_tokens++;
            }

            if (llama_decode(ctx->lctx, batch) != 0) {
                ctx->set_error_locked("Failed to decode prompt for embedding generation.");
                return false;
            }
        }

        const float* embd = llama_get_embeddings_seq(ctx->lctx, 1);
        if (!embd) embd = llama_get_embeddings_ith(ctx->lctx, -1);
        if (!embd) embd = llama_get_embeddings(ctx->lctx);

        if (!embd) {
            ctx->set_error_locked("Failed to extract embeddings from context.");
            return false;
        }

        std::memcpy(out_vec, embd, n_embd * sizeof(float));
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    }
}

void infer_abort(infer_ctx* ctx) {
    if (ctx) {
        ctx->abort_flag.store(true, std::memory_order_release);
    }
}

bool infer_is_generating(infer_ctx* ctx) {
    return ctx ? ctx->is_generating.load(std::memory_order_acquire) : false;
}

void infer_wait(infer_ctx* ctx) {
    if (!ctx) return;
    try {
        std::lock_guard<std::mutex> lock(ctx->worker_mutex);
        if (ctx->async_worker.joinable()) {
            ctx->async_worker.join();
        }
    } catch (...) {
        // Suppress exceptions in C API
    }
}

bool infer_generate_chat(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !ctx->model || !ctx->lctx || !ctx->sampler || !msgs || num_msgs == 0 || !callback) {
        if (ctx) ctx->set_error("Invalid parameters passed to infer_generate_chat.");
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Inference context is currently busy.");
        return false;
    }

    ctx->abort_flag.store(false, std::memory_order_release);
    ctx->clear_error();

    for (size_t i = 0; i < num_msgs; ++i) {
        if (!msgs[i].role || !msgs[i].content) {
            ctx->set_error("Message role or content cannot be null.");
            return false;
        }
    }

    try {
        return infer_generate_chat_internal(ctx, msgs, num_msgs, callback, user_data);
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    }
}

bool infer_generate_chat_async(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !msgs || num_msgs == 0 || !callback) return false;

    if (!ctx->model || !ctx->lctx || !ctx->sampler) {
        ctx->set_error("Model/context not initialized before async chat execution.");
        return false;
    }

    // Join prior worker under mutex
    {
        std::lock_guard<std::mutex> wlock(ctx->worker_mutex);
        if (ctx->async_worker.joinable()) {
            if (ctx->is_generating.load(std::memory_order_acquire)) {
                ctx->set_error("An inference worker is already active.");
                return false;
            }
            ctx->async_worker.join();
        }
    }

    // Acquire generation state on calling thread before dispatching
    bool expected = false;
    if (!ctx->is_generating.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        ctx->set_error("Inference context is currently busy.");
        return false;
    }

    ctx->abort_flag.store(false, std::memory_order_release);

    try {
        std::vector<OwnedMsg> owned;
        owned.reserve(num_msgs);
        for (size_t i = 0; i < num_msgs; ++i) {
            if (!msgs[i].role || !msgs[i].content) {
                ctx->is_generating.store(false, std::memory_order_release);
                ctx->set_error("Message role or content cannot be null.");
                return false;
            }
            owned.push_back({ msgs[i].role, msgs[i].content });
        }

        std::lock_guard<std::mutex> wlock(ctx->worker_mutex);
        ctx->async_worker = std::thread([ctx, owned_msgs = std::move(owned), callback, user_data]() {
            struct WorkerFlagGuard {
                infer_ctx* c;
                ~WorkerFlagGuard() {
                    c->is_generating.store(false, std::memory_order_release);
                }
            } flag_guard{ctx};

            std::vector<infer_chat_msg> view(owned_msgs.size());
            for (size_t i = 0; i < owned_msgs.size(); ++i) {
                view[i].role    = owned_msgs[i].role.c_str();
                view[i].content = owned_msgs[i].content.c_str();
            }

            try {
                ctx->clear_error();
                infer_generate_chat_internal(ctx, view.data(), view.size(), callback, user_data);
            } catch (const std::exception& e) {
                ctx->set_error(e.what());
            } catch (...) {
                ctx->set_error("Unknown exception during async chat execution.");
            }
        });

        return true;
    } catch (const std::exception& e) {
        ctx->is_generating.store(false, std::memory_order_release);
        ctx->set_error(e.what());
        return false;
    }
}

bool infer_generate_chat_turn(
    infer_ctx* ctx,
    const char* role,
    const char* content,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !ctx->model || !ctx->lctx || !ctx->sampler || !role || !content || !callback) {
        if (ctx) ctx->set_error("Invalid arguments for infer_generate_chat_turn.");
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Inference context is currently busy.");
        return false;
    }

    ctx->abort_flag.store(false, std::memory_order_release);
    ctx->clear_error();

    try {
        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        const char* chat_template = llama_model_chat_template(ctx->model, nullptr);

        struct llama_chat_message msg = { role, content };
        int alloc_size = llama_chat_apply_template(chat_template, &msg, 1, true, nullptr, 0);
        if (alloc_size <= 0) {
            ctx->set_error("Chat template formatting failed.");
            return false;
        }

        std::vector<char> formatted(alloc_size + 1, '\0');
        int written = llama_chat_apply_template(chat_template, &msg, 1, true, formatted.data(), formatted.size());
        if (written < 0) {
            ctx->set_error("Chat template formatting failed.");
            return false;
        }

        bool is_first = (ctx->n_past.load(std::memory_order_relaxed) == 0);
        bool add_bos = is_first ? want_bos(vocab, formatted.data()) : false;

        int n_tokens = -llama_tokenize(vocab, formatted.data(), written, nullptr, 0, add_bos, true);
        if (n_tokens <= 0) {
            ctx->set_error("Tokenization calculation failed.");
            return false;
        }

        std::vector<llama_token> prompt_tokens(n_tokens);
        int real_tokens = llama_tokenize(vocab, formatted.data(), written, prompt_tokens.data(), prompt_tokens.size(), add_bos, true);
        if (real_tokens < 0) {
            ctx->set_error("Failed to tokenize chat turn.");
            return false;
        }
        prompt_tokens.resize(real_tokens);

        return execute_decode(ctx, prompt_tokens, true, callback, user_data);
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    }
}

bool infer_generate_completion(
    infer_ctx* ctx,
    const char* raw_prompt,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !ctx->model || !ctx->lctx || !ctx->sampler || !raw_prompt || !callback) {
        if (ctx) ctx->set_error("Invalid arguments for infer_generate_completion.");
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Inference context is currently busy.");
        return false;
    }

    ctx->abort_flag.store(false, std::memory_order_release);
    ctx->clear_error();

    try {
        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        int written = static_cast<int>(std::strlen(raw_prompt));

        bool add_bos = want_bos(vocab, raw_prompt);
        int n_tokens = -llama_tokenize(vocab, raw_prompt, written, nullptr, 0, add_bos, false);
        if (n_tokens <= 0) {
            ctx->set_error("Tokenization calculation failed.");
            return false;
        }

        std::vector<llama_token> prompt_tokens(n_tokens);
        int real_tokens = llama_tokenize(vocab, raw_prompt, written, prompt_tokens.data(), prompt_tokens.size(), add_bos, false);
        if (real_tokens < 0) {
            ctx->set_error("Prompt tokenization failed.");
            return false;
        }
        prompt_tokens.resize(real_tokens);

        return execute_decode(ctx, prompt_tokens, false, callback, user_data);
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    }
}

void infer_free(infer_ctx* ctx) {
    if (!ctx) return;

    try {
        ctx->abort_flag.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> wlock(ctx->worker_mutex);
            if (ctx->async_worker.joinable()) {
                ctx->async_worker.join();
            }
        }

        {
            std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
            if (ctx->sampler) {
                llama_sampler_free(ctx->sampler);
                ctx->sampler = nullptr;
            }
            if (ctx->lctx) {
                llama_free(ctx->lctx);
                ctx->lctx = nullptr;
            }
            if (ctx->model) {
                llama_model_free(ctx->model);
                ctx->model = nullptr;
            }
        } // Mutex cleanly unlocked prior to delete

        delete ctx;
        unref_llama_backend();
    } catch (...) {
        // Suppress exceptions in final teardown
    }
}

} // extern "C"
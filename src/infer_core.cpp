#include "infer.h"

#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/* Internal context definition */
struct infer_ctx {
    struct llama_model*   model   = nullptr;
    struct llama_context* lctx    = nullptr;
    struct llama_sampler* sampler = nullptr;

    infer_sampler_params sampler_params{};
    uint32_t             n_ctx = 0;

    /* Async & thread synchronization state */
    std::atomic<bool> abort_flag{false};
    std::atomic<bool> is_generating{false};
    std::thread       async_worker;
    mutable std::mutex ctx_mutex;
    std::string        last_error;

    void set_error(const std::string& err) {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        last_error = err;
    }

    void clear_error() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        last_error.clear();
    }
};

/* Internal struct used to safely deep-copy messages for async dispatch */
struct OwnedMsg {
    std::string role;
    std::string content;
};

/* Probes available devices to find the primary hardware accelerator */
static ggml_backend_dev_t infer_get_optimal_device() {
    size_t dev_count = ggml_backend_dev_count();
    ggml_backend_dev_t fallback_dev = nullptr;

    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        // FIX: API renamed DEV_TYPE to DEVICE_TYPE
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return dev;
        }
        if (!fallback_dev) {
            fallback_dev = dev;
        }
    }
    return fallback_dev;
}

/* Auto-clamps n_ctx dynamically using hardware VRAM queries */
static uint32_t calculate_safe_n_ctx(struct llama_model* model, uint32_t requested_ctx) {
    int32_t n_layer   = llama_model_n_layer(model);
    int32_t n_head_kv = llama_model_n_head_kv(model);
    int32_t n_head    = llama_model_n_head(model);
    int32_t n_embd    = llama_model_n_embd(model);
    uint32_t max_train = static_cast<uint32_t>(llama_model_n_ctx_train(model));

    if (n_head <= 0 || n_layer <= 0 || n_head_kv <= 0 || n_embd <= 0) {
        return (requested_ctx > 0) ? requested_ctx : 2048;
    }

    int32_t head_dim = n_embd / n_head;
    constexpr size_t bytes_per_element = 2; // GGML_TYPE_F16
    size_t bytes_per_token = static_cast<size_t>(2) * n_layer * n_head_kv * head_dim * bytes_per_element;

    ggml_backend_dev_t dev = infer_get_optimal_device();
    size_t free_vram = 0, total_vram = 0;

    if (dev) {
        ggml_backend_dev_memory(dev, &free_vram, &total_vram);
    }

    if (free_vram == 0) {
        if (requested_ctx == 0) return std::min<uint32_t>(max_train, 4096);
        return std::min<uint32_t>(requested_ctx, max_train);
    }

    constexpr size_t headroom = 512ull * 1024ull * 1024ull;
    size_t available_vram = (free_vram > headroom) ? (free_vram - headroom) : 0;
    uint32_t safe_tokens = static_cast<uint32_t>(available_vram / bytes_per_token);

    constexpr uint32_t min_tokens = 512;
    if (safe_tokens < min_tokens) safe_tokens = min_tokens;
    safe_tokens = std::min<uint32_t>(safe_tokens, max_train);

    if (requested_ctx == 0) return safe_tokens;
    return std::min<uint32_t>(requested_ctx, safe_tokens);
}

static struct llama_sampler* create_sampler(const infer_sampler_params& params) {
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* chain = llama_sampler_chain_init(sparams);

    if (params.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed ? params.seed : LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }

    return chain;
}

/* ========================================================================== */
/* Public API                                                                 */
/* ========================================================================== */

infer_sampler_params infer_sampler_default_params(void) {
    return { 0.70f, 0.90f, 40, 0 };
}

infer_ctx* infer_init(void) {
    llama_backend_init();

    infer_ctx* ctx = new (std::nothrow) infer_ctx();
    if (ctx) {
        ctx->sampler_params = infer_sampler_default_params();
    }
    return ctx;
}

bool infer_load_model(infer_ctx* ctx, const char* path, uint32_t requested_ctx) {
    if (!ctx || !path) return false;
    ctx->clear_error();

    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

    if (ctx->sampler) { llama_sampler_free(ctx->sampler); ctx->sampler = nullptr; }
    if (ctx->lctx)    { llama_free(ctx->lctx); ctx->lctx = nullptr; }
    if (ctx->model)   { llama_model_free(ctx->model); ctx->model = nullptr; }

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;

    // FIX: Updated from llama_load_model_from_file
    ctx->model = llama_model_load_from_file(path, model_params);
    if (!ctx->model) {
        ctx->set_error("Failed to load model weights: " + std::string(path));
        return false;
    }

    ctx->n_ctx = calculate_safe_n_ctx(ctx->model, requested_ctx);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = ctx->n_ctx;
    ctx_params.n_batch = std::min<uint32_t>(ctx->n_ctx, 512);

    // FIX: Updated from llama_new_context_with_model
    ctx->lctx = llama_init_from_model(ctx->model, ctx_params);
    if (!ctx->lctx) {
        llama_model_free(ctx->model);
        ctx->model = nullptr;
        ctx->set_error("Failed to instantiate llama context.");
        return false;
    }

    ctx->sampler = create_sampler(ctx->sampler_params);
    return true;
}

bool infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params) {
    if (!ctx || !params) return false;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

    ctx->sampler_params = *params;
    if (ctx->sampler) {
        llama_sampler_free(ctx->sampler);
    }
    ctx->sampler = create_sampler(ctx->sampler_params);
    return true;
}

void infer_abort(infer_ctx* ctx) {
    if (ctx) {
        ctx->abort_flag.store(true);
    }
}

bool infer_is_generating(infer_ctx* ctx) {
    return ctx ? ctx->is_generating.load(std::memory_order_acquire) : false;
}

void infer_wait(infer_ctx* ctx) {
    if (ctx && ctx->async_worker.joinable()) {
        ctx->async_worker.join();
    }
}

const char* infer_get_last_error(infer_ctx* ctx) {
    if (!ctx) return "Null context pointer.";
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
    return ctx->last_error.c_str();
}

bool infer_generate_chat(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !ctx->model || !ctx->lctx || !ctx->sampler || !msgs || num_msgs == 0 || !callback) {
        if (ctx) ctx->set_error("Invalid arguments provided to infer_generate_chat.");
        return false;
    }
    ctx->clear_error();

    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    const char* chat_template = llama_model_chat_template(ctx->model, nullptr);

    // 1. Convert messages for Jinja processing
    std::vector<struct llama_chat_message> llama_msgs;
    llama_msgs.reserve(num_msgs);
    for (size_t i = 0; i < num_msgs; ++i) {
        llama_msgs.push_back({ msgs[i].role, msgs[i].content });
    }

    // FIX: Reverted to 6 arguments (ctx->model parameter was removed in latest master)
    int alloc_size = llama_chat_apply_template(
        chat_template,
        llama_msgs.data(),
        llama_msgs.size(),
        true,
        nullptr,
        0
    );

    if (alloc_size <= 0) {
        ctx->set_error("Jinja template formatting failed during length calculation.");
        return false;
    }

    // 3. Pass 2: Format prompt
    std::vector<char> formatted(alloc_size + 1, '\0');
    int written = llama_chat_apply_template(
        chat_template,
        llama_msgs.data(),
        llama_msgs.size(),
        true,
        formatted.data(),
        formatted.size()
    );

    if (written < 0) {
        ctx->set_error("Failed to render prompt with chat template.");
        return false;
    }

    // 4. Tokenization
    int n_prompt_tokens = -llama_tokenize(vocab, formatted.data(), written, nullptr, 0, true, true);
    if (n_prompt_tokens <= 0) {
        ctx->set_error("Tokenization failed to determine token count.");
        return false;
    }

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    int real_tokens = llama_tokenize(
        vocab,
        formatted.data(),
        written,
        prompt_tokens.data(),
        prompt_tokens.size(),
        true,
        true
    );

    if (real_tokens < 0) {
        ctx->set_error("Prompt tokenization failed.");
        return false;
    }
    prompt_tokens.resize(real_tokens);

    if (static_cast<uint32_t>(prompt_tokens.size()) >= ctx->n_ctx) {
        ctx->set_error("Prompt tokens exceed context capacity.");
        return false;
    }

    // FIX: llama_kv_cache_* namespace was entirely renamed to llama_memory_*
    llama_memory_clear(ctx->lctx);
    llama_sampler_reset(ctx->sampler);

    const int32_t batch_cap = 512;
    struct llama_batch batch = llama_batch_init(batch_cap, 0, 1);

    struct BatchGuard {
        struct llama_batch& b;
        ~BatchGuard() { llama_batch_free(b); }
    } guard{batch};

    // 5. Ingest Prompt in Chunks
    for (size_t i = 0; i < prompt_tokens.size(); i += batch.n_tokens) {
        if (ctx->abort_flag.load()) {
            return false;
        }

        batch.n_tokens = 0;
        int chunk = std::min<int>(batch_cap, static_cast<int>(prompt_tokens.size() - i));

        for (int j = 0; j < chunk; ++j) {
            size_t idx = i + j;
            batch.token[j]    = prompt_tokens[idx];
            batch.pos[j]      = static_cast<llama_pos>(idx);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0]= 0;
            batch.logits[j]   = (idx == prompt_tokens.size() - 1);
            batch.n_tokens++;
        }

        if (llama_decode(ctx->lctx, batch) != 0) {
            ctx->set_error("Prompt decoding failed in llama_decode.");
            return false;
        }
    }

    // 6. Infinite Chat Generation with Context Shifting
    int n_cur = static_cast<int>(prompt_tokens.size());
    int n_keep = (prompt_tokens.size() > 64) ? 64 : std::min<int>(static_cast<int>(prompt_tokens.size()), 1);

    if (n_keep >= static_cast<int>(ctx->n_ctx) - 8) {
        n_keep = std::max<int>(0, static_cast<int>(ctx->n_ctx) / 4);
    }

    while (true) {
        if (ctx->abort_flag.load()) {
            break;
        }

        if (n_cur >= static_cast<int>(ctx->n_ctx) - 4) {
            int n_discard = (ctx->n_ctx - n_keep) / 2;
            if (n_discard <= 0) n_discard = 1;

            // FIX: llama_kv_cache_seq_* was completely replaced by llama_memory_seq_*
            llama_memory_seq_rm(ctx->lctx, 0, n_keep, n_keep + n_discard);
            llama_memory_seq_add(ctx->lctx, 0, n_keep + n_discard, n_cur, -n_discard);

            n_cur -= n_discard;
        }

        llama_token new_token_id = llama_sampler_sample(ctx->sampler, ctx->lctx, -1);
        llama_sampler_accept(ctx->sampler, new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char piece_buf[128];
        int n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (n_chars > 0) {
            piece_buf[std::min<int>(n_chars, static_cast<int>(sizeof(piece_buf) - 1))] = '\0';
            if (!callback(piece_buf, user_data)) {
                ctx->abort_flag.store(true);
                break;
            }
        }

        batch.n_tokens = 0;
        batch.token[0]    = new_token_id;
        batch.pos[0]      = static_cast<llama_pos>(n_cur);
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0]= 0;
        batch.logits[0]   = true;
        batch.n_tokens    = 1;

        if (llama_decode(ctx->lctx, batch) != 0) {
            ctx->set_error("Failed to evaluate token in KV cache.");
            return false;
        }

        n_cur++;
    }

    return true;
}

bool infer_generate_chat_async(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !msgs || num_msgs == 0 || !callback) return false;

    if (ctx->async_worker.joinable()) {
        if (ctx->is_generating.load()) {
            ctx->set_error("An inference worker is already active.");
            return false;
        }
        ctx->async_worker.join();
    }

    ctx->abort_flag.store(false);
    ctx->is_generating.store(true);

    std::vector<OwnedMsg> owned;
    owned.reserve(num_msgs);
    for (size_t i = 0; i < num_msgs; ++i) {
        owned.push_back({
            msgs[i].role ? msgs[i].role : "",
            msgs[i].content ? msgs[i].content : ""
        });
    }

    ctx->async_worker = std::thread([ctx, owned_msgs = std::move(owned), callback, user_data]() {
        std::vector<infer_chat_msg> view(owned_msgs.size());
        for (size_t i = 0; i < owned_msgs.size(); ++i) {
            view[i].role    = owned_msgs[i].role.c_str();
            view[i].content = owned_msgs[i].content.c_str();
        }

        infer_generate_chat(ctx, view.data(), view.size(), callback, user_data);
        ctx->is_generating.store(false);
    });

    return true;
}

void infer_free(infer_ctx* ctx) {
    if (!ctx) return;

    ctx->abort_flag.store(true);
    if (ctx->async_worker.joinable()) {
        ctx->async_worker.join();
    }

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

    delete ctx;
    llama_backend_free();
}

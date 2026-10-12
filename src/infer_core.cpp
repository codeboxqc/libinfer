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

    infer_sampler_params     sampler_params{};
    std::string              grammar_gbnf;
    std::vector<std::string> stop_words;
    uint32_t                 n_ctx = 0;

    /* KV Cache tracking */
    int n_past = 0;

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

static struct llama_sampler* create_sampler(
    const infer_sampler_params& params,
    struct llama_model* model,
    const std::string& grammar_gbnf
) {
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* chain = llama_sampler_chain_init(sparams);
 
if (params.repeat_penalty != 1.0f && params.repeat_penalty > 0.0f) {
        int32_t last_n = (params.repeat_last_n > 0) ? params.repeat_last_n : 64;
        
        int32_t n_vocab = 0;
        if (model) {
            const struct llama_vocab* vocab = llama_model_get_vocab(model);
            if (vocab) {
                n_vocab = llama_vocab_n_tokens(vocab);
            }
        }
        
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, last_n, params.repeat_penalty, 0.0f, 0.0f));
    }
    if (params.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.min_p > 0.0f && params.min_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(params.min_p, 1));
    }

    if (!grammar_gbnf.empty() && model) {
        const struct llama_vocab* vocab = llama_model_get_vocab(model);
        if (vocab) {
            struct llama_sampler* gsampler = llama_sampler_init_grammar(vocab, grammar_gbnf.c_str(), "root");
            if (gsampler) {
                llama_sampler_chain_add(chain, gsampler);
            }
        }
    }

    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed ? params.seed : LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }

    return chain;
}

static bool check_stop_words(const std::string& buffer, const std::vector<std::string>& stop_words) {
    for (const auto& stop : stop_words) {
        if (!stop.empty() && buffer.size() >= stop.size()) {
            if (buffer.compare(buffer.size() - stop.size(), stop.size(), stop) == 0) {
                return true;
            }
        }
    }
    return false;
}

/* ========================================================================== */
/* Public API                                                                 */
/* ========================================================================== */

infer_sampler_params infer_sampler_default_params(void) {
    return { 0.70f, 0.90f, 40, 0.05f, 1.10f, 64, 0 };
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

    ctx->model = llama_model_load_from_file(path, model_params);
    if (!ctx->model) {
        ctx->set_error("Failed to load model weights: " + std::string(path));
        return false;
    }

    ctx->n_ctx = calculate_safe_n_ctx(ctx->model, requested_ctx);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = ctx->n_ctx;
    ctx_params.n_batch = std::min<uint32_t>(ctx->n_ctx, 512);

    ctx->lctx = llama_init_from_model(ctx->model, ctx_params);
    if (!ctx->lctx) {
        llama_model_free(ctx->model);
        ctx->model = nullptr;
        ctx->set_error("Failed to instantiate llama context.");
        return false;
    }

    ctx->sampler = create_sampler(ctx->sampler_params, ctx->model, ctx->grammar_gbnf);
    ctx->n_past = 0;
    return true;
}

bool infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params) {
    if (!ctx || !params) return false;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

    ctx->sampler_params = *params;
    if (ctx->sampler) {
        llama_sampler_free(ctx->sampler);
    }
    ctx->sampler = create_sampler(ctx->sampler_params, ctx->model, ctx->grammar_gbnf);
    return true;
}

bool infer_get_model_info(infer_ctx* ctx, infer_model_info* out_info) {
    if (!ctx || !ctx->model || !out_info) return false;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    out_info->n_ctx_train = static_cast<uint32_t>(llama_model_n_ctx_train(ctx->model));
    out_info->n_embd      = static_cast<uint32_t>(llama_model_n_embd(ctx->model));
    out_info->n_layer     = static_cast<uint32_t>(llama_model_n_layer(ctx->model));
    out_info->n_head      = static_cast<uint32_t>(llama_model_n_head(ctx->model));
    out_info->vocab_size  = vocab ? static_cast<uint32_t>(llama_vocab_n_tokens(vocab)) : 0;

    std::memset(out_info->description, 0, sizeof(out_info->description));
    llama_model_desc(ctx->model, out_info->description, sizeof(out_info->description));
    return true;
}

int32_t infer_tokenize_count(infer_ctx* ctx, const char* text, bool add_special) {
    if (!ctx || !ctx->model || !text) return 0;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    if (!vocab) return 0;

    int n = llama_tokenize(vocab, text, static_cast<int32_t>(std::strlen(text)), nullptr, 0, add_special, true);
    return (n < 0) ? -n : n;
}

uint32_t infer_get_kv_cache_token_count(infer_ctx* ctx) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
    return static_cast<uint32_t>(ctx->n_past);
}

bool infer_set_stop_sequences(infer_ctx* ctx, const char** stop_words, size_t num_words) {
    if (!ctx) return false;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

    ctx->stop_words.clear();
    if (stop_words && num_words > 0) {
        for (size_t i = 0; i < num_words; ++i) {
            if (stop_words[i]) {
                ctx->stop_words.push_back(stop_words[i]);
            }
        }
    }
    return true;
}

bool infer_set_grammar(infer_ctx* ctx, const char* grammar_gbnf) {
    if (!ctx) return false;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

    ctx->grammar_gbnf = grammar_gbnf ? grammar_gbnf : "";
    if (ctx->sampler) {
        llama_sampler_free(ctx->sampler);
        ctx->sampler = create_sampler(ctx->sampler_params, ctx->model, ctx->grammar_gbnf);
    }
    return true;
}

bool infer_state_save(infer_ctx* ctx, const char* filepath) {
    if (!ctx || !ctx->lctx || !filepath) return false;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

    size_t state_size = llama_state_get_size(ctx->lctx);
    if (state_size == 0) {
        ctx->set_error("llama_state_get_size returned 0.");
        return false;
    }

    std::vector<uint8_t> buffer(state_size);
    size_t written = llama_state_get_data(ctx->lctx, buffer.data(), buffer.size());
    if (written != state_size) {
        ctx->set_error("Failed to extract context state data.");
        return false;
    }

    FILE* f = std::fopen(filepath, "wb");
    if (!f) {
        ctx->set_error("Failed to open file for state saving: " + std::string(filepath));
        return false;
    }

    uint32_t header_magic = 0x494E4653; // "INFS"
    uint32_t n_past_val   = static_cast<uint32_t>(ctx->n_past);
    uint64_t data_size    = static_cast<uint64_t>(state_size);

    std::fwrite(&header_magic, sizeof(header_magic), 1, f);
    std::fwrite(&n_past_val, sizeof(n_past_val), 1, f);
    std::fwrite(&data_size, sizeof(data_size), 1, f);
    std::fwrite(buffer.data(), 1, buffer.size(), f);
    std::fclose(f);

    return true;
}

bool infer_state_load(infer_ctx* ctx, const char* filepath) {
    if (!ctx || !ctx->lctx || !filepath) return false;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

    FILE* f = std::fopen(filepath, "rb");
    if (!f) {
        ctx->set_error("Failed to open file for state loading: " + std::string(filepath));
        return false;
    }

    uint32_t header_magic = 0;
    uint32_t n_past_val   = 0;
    uint64_t data_size    = 0;

    if (std::fread(&header_magic, sizeof(header_magic), 1, f) != 1 ||
        std::fread(&n_past_val, sizeof(n_past_val), 1, f) != 1 ||
        std::fread(&data_size, sizeof(data_size), 1, f) != 1 ||
        header_magic != 0x494E4653) {
        std::fclose(f);
        ctx->set_error("Invalid state file header or magic bytes.");
        return false;
    }

    std::vector<uint8_t> buffer(data_size);
    if (std::fread(buffer.data(), 1, data_size, f) != data_size) {
        std::fclose(f);
        ctx->set_error("Failed to read complete state data from file.");
        return false;
    }
    std::fclose(f);

    size_t loaded_size = llama_state_set_data(ctx->lctx, buffer.data(), buffer.size());
    if (loaded_size != data_size) {
        ctx->set_error("llama_state_set_data failed to load context state.");
        return false;
    }

    ctx->n_past = static_cast<int>(n_past_val);
    return true;
}

int32_t infer_get_embedding_dim(infer_ctx* ctx) {
    if (!ctx || !ctx->model) return 0;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
    return llama_model_n_embd(ctx->model);
}

bool infer_get_embeddings(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len) {
    if (!ctx || !ctx->model || !ctx->lctx || !text || !out_vec) return false;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);

    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    if (!vocab) return false;

    int n_embd = llama_model_n_embd(ctx->model);
    if (vec_len < static_cast<size_t>(n_embd)) {
        ctx->set_error("Provided output buffer is smaller than model embedding dimension.");
        return false;
    }

    int n_tokens = -llama_tokenize(vocab, text, static_cast<int32_t>(std::strlen(text)), nullptr, 0, true, true);
    if (n_tokens <= 0) return false;

    std::vector<llama_token> tokens(n_tokens);
    int real_tokens = llama_tokenize(vocab, text, static_cast<int32_t>(std::strlen(text)), tokens.data(), tokens.size(), true, true);
    if (real_tokens < 0) return false;
    tokens.resize(real_tokens);

    llama_memory_clear(llama_get_memory(ctx->lctx), true);

    struct llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = static_cast<llama_pos>(i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == tokens.size() - 1);
        batch.n_tokens++;
    }

    if (llama_decode(ctx->lctx, batch) != 0) {
        llama_batch_free(batch);
        ctx->set_error("Failed to decode prompt for embedding generation.");
        return false;
    }
    llama_batch_free(batch);

    const float* embd = llama_get_embeddings_seq(ctx->lctx, 0);
    if (!embd) embd = llama_get_embeddings_ith(ctx->lctx, -1);
    if (!embd) embd = llama_get_embeddings(ctx->lctx);

    if (!embd) {
        ctx->set_error("Failed to retrieve embeddings from llama context.");
        return false;
    }

    std::memcpy(out_vec, embd, n_embd * sizeof(float));
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

void infer_memory_clear(infer_ctx* ctx) {
    if (!ctx || !ctx->lctx) return;
    std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
    
    llama_memory_clear(llama_get_memory(ctx->lctx), true);
    
    if (ctx->sampler) {
        llama_sampler_reset(ctx->sampler);
    }
    
    ctx->n_past = 0;
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

    std::vector<struct llama_chat_message> llama_msgs;
    llama_msgs.reserve(num_msgs);
    for (size_t i = 0; i < num_msgs; ++i) {
        llama_msgs.push_back({ msgs[i].role, msgs[i].content });
    }

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

    llama_memory_clear(llama_get_memory(ctx->lctx), true);
    llama_sampler_reset(ctx->sampler);
    ctx->n_past = 0;

    const int32_t batch_cap = 512;
    struct llama_batch batch = llama_batch_init(batch_cap, 0, 1);

    struct BatchGuard {
        struct llama_batch& b;
        ~BatchGuard() { llama_batch_free(b); }
    } guard{batch};

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

    int n_cur = static_cast<int>(prompt_tokens.size());
    std::string generated_acc;

    while (true) {
        if (ctx->abort_flag.load()) break;

        if (n_cur >= static_cast<int>(ctx->n_ctx) - 4) {
            int n_keep = (n_cur > 64) ? 64 : std::min<int>(n_cur, 1);
            int n_discard = (ctx->n_ctx - n_keep) / 2;
            if (n_discard <= 0) n_discard = 1;

            llama_memory_seq_rm(llama_get_memory(ctx->lctx), 0, n_keep, n_keep + n_discard);
            llama_memory_seq_add(llama_get_memory(ctx->lctx), 0, n_keep + n_discard, n_cur, -n_discard);
            n_cur -= n_discard;
        }

        llama_token new_token_id = llama_sampler_sample(ctx->sampler, ctx->lctx, -1);
        llama_sampler_accept(ctx->sampler, new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char piece_buf[128];
        int n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (n_chars > 0) {
            piece_buf[std::min<int>(n_chars, static_cast<int>(sizeof(piece_buf) - 1))] = '\0';
            generated_acc += piece_buf;

            if (check_stop_words(generated_acc, ctx->stop_words)) {
                break;
            }

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

    ctx->n_past = n_cur;
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
    ctx->clear_error();

    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    const char* chat_template = llama_model_chat_template(ctx->model, nullptr);

    struct llama_chat_message msg = { role, content };

    int alloc_size = llama_chat_apply_template(chat_template, &msg, 1, true, nullptr, 0);
    if (alloc_size <= 0) {
        ctx->set_error("Jinja template formatting failed during length calculation.");
        return false;
    }

    std::vector<char> formatted(alloc_size + 1, '\0');
    int written = llama_chat_apply_template(chat_template, &msg, 1, true, formatted.data(), formatted.size());
    if (written < 0) {
        ctx->set_error("Failed to render prompt with chat template.");
        return false;
    }

    bool is_first = (ctx->n_past == 0);
    int n_tokens = -llama_tokenize(vocab, formatted.data(), written, nullptr, 0, is_first, true);
    
    if (n_tokens <= 0) {
        ctx->set_error("Tokenization failed to determine token count.");
        return false;
    }

    std::vector<llama_token> prompt_tokens(n_tokens);
    int real_tokens = llama_tokenize(vocab, formatted.data(), written, prompt_tokens.data(), prompt_tokens.size(), is_first, true);
    if (real_tokens < 0) {
        ctx->set_error("Prompt tokenization failed.");
        return false;
    }
    prompt_tokens.resize(real_tokens);

    if (static_cast<uint32_t>(ctx->n_past + prompt_tokens.size()) >= ctx->n_ctx) {
        ctx->set_error("Prompt tokens exceed context capacity.");
        return false;
    }

    const int32_t batch_cap = 512;
    struct llama_batch batch = llama_batch_init(batch_cap, 0, 1);
    
    struct BatchGuard {
        struct llama_batch& b;
        ~BatchGuard() { llama_batch_free(b); }
    } guard{batch};

    for (size_t i = 0; i < prompt_tokens.size(); i += batch.n_tokens) {
        if (ctx->abort_flag.load()) return false;

        batch.n_tokens = 0;
        int chunk = std::min<int>(batch_cap, static_cast<int>(prompt_tokens.size() - i));

        for (int j = 0; j < chunk; ++j) {
            size_t idx = i + j;
            batch.token[j]    = prompt_tokens[idx];
            batch.pos[j]      = static_cast<llama_pos>(ctx->n_past + idx); 
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

    ctx->n_past += prompt_tokens.size();
    std::string generated_acc;

    while (true) {
        if (ctx->abort_flag.load()) break;

        if (ctx->n_past >= static_cast<int>(ctx->n_ctx) - 4) {
            int n_keep = (ctx->n_past > 64) ? 64 : std::min<int>(ctx->n_past, 1);
            int n_discard = (ctx->n_ctx - n_keep) / 2;
            if (n_discard <= 0) n_discard = 1;

            llama_memory_seq_rm(llama_get_memory(ctx->lctx), 0, n_keep, n_keep + n_discard);
            llama_memory_seq_add(llama_get_memory(ctx->lctx), 0, n_keep + n_discard, ctx->n_past, -n_discard);
            ctx->n_past -= n_discard;
        }

        llama_token new_token = llama_sampler_sample(ctx->sampler, ctx->lctx, -1);
        llama_sampler_accept(ctx->sampler, new_token);

        if (llama_vocab_is_eog(vocab, new_token)) break;

        char piece_buf[128];
        int n_chars = llama_token_to_piece(vocab, new_token, piece_buf, sizeof(piece_buf), 0, true);
        if (n_chars > 0) {
            piece_buf[std::min<int>(n_chars, static_cast<int>(sizeof(piece_buf) - 1))] = '\0';
            generated_acc += piece_buf;

            if (check_stop_words(generated_acc, ctx->stop_words)) {
                break;
            }

            if (!callback(piece_buf, user_data)) {
                ctx->abort_flag.store(true);
                break;
            }
        }

        batch.n_tokens = 0;
        batch.token[0]    = new_token;
        batch.pos[0]      = static_cast<llama_pos>(ctx->n_past);
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0]= 0;
        batch.logits[0]   = true;
        batch.n_tokens    = 1;

        if (llama_decode(ctx->lctx, batch) != 0) {
            ctx->set_error("Failed to evaluate token in KV cache.");
            return false;
        }

        ctx->n_past++;
    }

    return true;
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
    ctx->clear_error();

    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    int written = static_cast<int>(std::strlen(raw_prompt));

    int n_tokens = -llama_tokenize(vocab, raw_prompt, written, nullptr, 0, true, true);
    if (n_tokens <= 0) {
        ctx->set_error("Tokenization failed to determine token count.");
        return false;
    }

    std::vector<llama_token> prompt_tokens(n_tokens);
    int real_tokens = llama_tokenize(vocab, raw_prompt, written, prompt_tokens.data(), prompt_tokens.size(), true, true);
    if (real_tokens < 0) {
        ctx->set_error("Prompt tokenization failed.");
        return false;
    }
    prompt_tokens.resize(real_tokens);

    if (static_cast<uint32_t>(prompt_tokens.size()) >= ctx->n_ctx) {
        ctx->set_error("Prompt tokens exceed context capacity.");
        return false;
    }

    llama_memory_clear(llama_get_memory(ctx->lctx), true);
    llama_sampler_reset(ctx->sampler);
    ctx->n_past = 0;

    const int32_t batch_cap = 512;
    struct llama_batch batch = llama_batch_init(batch_cap, 0, 1);

    struct BatchGuard {
        struct llama_batch& b;
        ~BatchGuard() { llama_batch_free(b); }
    } guard{batch};

    for (size_t i = 0; i < prompt_tokens.size(); i += batch.n_tokens) {
        if (ctx->abort_flag.load()) return false;

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

    int n_cur = static_cast<int>(prompt_tokens.size());
    std::string generated_acc;

    while (true) {
        if (ctx->abort_flag.load()) break;

        if (n_cur >= static_cast<int>(ctx->n_ctx) - 4) {
            int n_keep = (n_cur > 64) ? 64 : std::min<int>(n_cur, 1);
            int n_discard = (ctx->n_ctx - n_keep) / 2;
            if (n_discard <= 0) n_discard = 1;

            llama_memory_seq_rm(llama_get_memory(ctx->lctx), 0, n_keep, n_keep + n_discard);
            llama_memory_seq_add(llama_get_memory(ctx->lctx), 0, n_keep + n_discard, n_cur, -n_discard);
            n_cur -= n_discard;
        }

        llama_token new_token_id = llama_sampler_sample(ctx->sampler, ctx->lctx, -1);
        llama_sampler_accept(ctx->sampler, new_token_id);

        if (llama_vocab_is_eog(vocab, new_token_id)) break;

        char piece_buf[128];
        int n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (n_chars > 0) {
            piece_buf[std::min<int>(n_chars, static_cast<int>(sizeof(piece_buf) - 1))] = '\0';
            generated_acc += piece_buf;

            if (check_stop_words(generated_acc, ctx->stop_words)) {
                break;
            }

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

    ctx->n_past = n_cur;
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

#ifndef LIBINFER_H
#define LIBINFER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(INFER_BUILD_SHARED)
        #define INFER_API __declspec(dllexport)
    #else
        #define INFER_API __declspec(dllimport)
    #endif
#elif defined(__GNUC__) && __GNUC__ >= 4
    #define INFER_API __attribute__((visibility("default")))
#else
    #define INFER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle representing the inference context.
 * Thread-safety contract:
 * - infer_abort, infer_is_generating, and infer_wait are safe to invoke concurrently from any thread.
 * - Mutating calls (infer_load_model, infer_set_sampler, infer_set_grammar, infer_memory_clear)
 *   will fail safely with an error message if an operation or generation is in progress.
 * - Async streaming callbacks execute on the internal worker thread.
 */
typedef struct infer_ctx infer_ctx;

/**
 * @brief Streaming token callback. Return false to abort generation early.
 */
typedef bool (*infer_stream_cb)(const char* piece, void* user_data);

/**
 * @brief Sampling parameters for token selection.
 */
typedef struct infer_sampler_params {
    float    temperature;   /**< <= 0.0f enables greedy decoding */
    float    top_p;         /**< 1.0f disables */
    int32_t  top_k;         /**< <= 0 disables */
    float    min_p;         /**< <= 0.0f disables (e.g. 0.05f) */
    float    repeat_penalty;/**< 1.0f disables (e.g. 1.1f) */
    int32_t  repeat_last_n; /**< <= 0 for default 64 */
    uint32_t seed;          /**< 0 for random seed */
} infer_sampler_params;

/**
 * @brief Represents a single chat message.
 */
typedef struct infer_chat_msg {
    const char* role;
    const char* content;
} infer_chat_msg;

/**
 * @brief Model metadata information.
 */
typedef struct infer_model_info {
    uint32_t n_ctx_train;     /**< Model's trained max context length */
    uint32_t n_embd;          /**< Embedding dimensions */
    uint32_t n_layer;         /**< Number of layers */
    uint32_t n_head;          /**< Number of attention heads */
    uint32_t vocab_size;      /**< Vocabulary size */
    char     description[256];/**< Model architecture / description */
} infer_model_info;

/* Core Lifecycle */
INFER_API infer_sampler_params infer_sampler_default_params(void);
INFER_API infer_ctx* infer_init(void);
INFER_API bool infer_load_model(infer_ctx* ctx, const char* path, uint32_t requested_ctx);
INFER_API bool infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params);
INFER_API void infer_free(infer_ctx* ctx);

/* Diagnostics */
INFER_API bool infer_get_model_info(infer_ctx* ctx, infer_model_info* out_info);
INFER_API size_t infer_get_last_error(infer_ctx* ctx, char* out_buf, size_t buf_size);

/* Tokenization & Counting */
INFER_API int32_t infer_tokenize_count(infer_ctx* ctx, const char* text, bool add_special);
INFER_API uint32_t infer_get_kv_cache_token_count(infer_ctx* ctx);

/* Generation Modes & Control */
INFER_API bool infer_generate_chat(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    infer_stream_cb callback,
    void* user_data
);

INFER_API bool infer_generate_chat_async(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    infer_stream_cb callback,
    void* user_data
);

INFER_API bool infer_generate_chat_turn(
    infer_ctx* ctx,
    const char* role,
    const char* content,
    infer_stream_cb callback,
    void* user_data
);

INFER_API bool infer_generate_completion(
    infer_ctx* ctx,
    const char* raw_prompt,
    infer_stream_cb callback,
    void* user_data
);

/* Stop Sequences & Grammar */
INFER_API bool infer_set_stop_sequences(infer_ctx* ctx, const char* const* stop_words, size_t num_words);
INFER_API bool infer_set_grammar(infer_ctx* ctx, const char* grammar_gbnf);

/* KV Cache & Session State Save/Load */
INFER_API void infer_memory_clear(infer_ctx* ctx);
INFER_API bool infer_state_save(infer_ctx* ctx, const char* filepath);
INFER_API bool infer_state_load(infer_ctx* ctx, const char* filepath);

/* Embeddings */
INFER_API int32_t infer_get_embedding_dim(infer_ctx* ctx);
INFER_API bool infer_get_embeddings(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len);

/* Async & Thread Synchronization */
INFER_API void infer_abort(infer_ctx* ctx);
INFER_API bool infer_is_generating(infer_ctx* ctx);
INFER_API void infer_wait(infer_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* LIBINFER_H */
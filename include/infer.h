/* ========================================================================== */
/* infer.h (v2.2.0 - Production Standard)                                    */
/* ========================================================================== */
#ifndef LIBINFER_H
#define LIBINFER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(INFER_STATIC)
    #define INFER_API
#elif defined(_WIN32) || defined(__CYGWIN__)
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

#define INFER_VERSION_MAJOR 2
#define INFER_VERSION_MINOR 2
#define INFER_VERSION_PATCH 0
#define INFER_VERSION_NUM   ((INFER_VERSION_MAJOR << 16) | (INFER_VERSION_MINOR << 8) | (INFER_VERSION_PATCH))

typedef struct infer_ctx infer_ctx;

typedef bool (*infer_stream_cb)(const char* piece, void* user_data);

typedef enum infer_log_level {
    INFER_LOG_NONE  = 0,
    INFER_LOG_DEBUG = 1,
    INFER_LOG_INFO  = 2,
    INFER_LOG_WARN  = 3,
    INFER_LOG_ERROR = 4,
    INFER_LOG_CONT  = 5
} infer_log_level;

typedef void (*infer_log_cb)(infer_log_level level, const char* text, void* user_data);

typedef struct infer_sampler_params {
    uint32_t struct_size;    /**< Set to sizeof(infer_sampler_params) */
    float    temperature;    /**< <= 0.0f enables greedy decoding */
    float    top_p;          /**< 1.0f disables */
    int32_t  top_k;          /**< <= 0 disables */
    float    min_p;          /**< <= 0.0f disables */
    float    repeat_penalty; /**< 1.0f disables */
    int32_t  repeat_last_n;  /**< <= 0 for default 64 */
    uint32_t seed;           /**< 0 for random seed */
} infer_sampler_params;

typedef struct infer_chat_msg {
    const char* role;        /**< Whitelisted: "system", "user", "assistant", "tool", "developer" */
    const char* content;
} infer_chat_msg;

typedef struct infer_model_info {
    uint32_t struct_size;    /**< Set to sizeof(infer_model_info) */
    uint32_t n_ctx_train;
    uint32_t n_embd;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t vocab_size;
    char     description[256];
    uint32_t n_ctx_active;
} infer_model_info;

INFER_API uint32_t infer_version(void);
INFER_API void infer_set_log_callback(infer_log_cb callback, void* user_data);
INFER_API bool infer_get_model_info(infer_ctx* ctx, infer_model_info* out_info);
INFER_API size_t infer_get_last_error(infer_ctx* ctx, char* out_buf, size_t buf_size);
INFER_API bool infer_last_success(infer_ctx* ctx);

INFER_API infer_sampler_params infer_sampler_default_params(void);
INFER_API infer_ctx* infer_init(void);
INFER_API bool infer_load_model(infer_ctx* ctx, const char* path, uint32_t requested_ctx);
INFER_API bool infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params);
INFER_API void infer_free(infer_ctx* ctx);

INFER_API int32_t infer_tokenize_count(infer_ctx* ctx, const char* text, bool add_special);
INFER_API uint32_t infer_get_kv_cache_token_count(infer_ctx* ctx);

INFER_API bool infer_generate_chat(infer_ctx* ctx, const infer_chat_msg* msgs, size_t num_msgs, int32_t max_tokens, infer_stream_cb callback, void* user_data);
INFER_API bool infer_generate_chat_async(infer_ctx* ctx, const infer_chat_msg* msgs, size_t num_msgs, int32_t max_tokens, infer_stream_cb callback, void* user_data);
INFER_API bool infer_generate_chat_turn(infer_ctx* ctx, const char* role, const char* content, int32_t max_tokens, infer_stream_cb callback, void* user_data);
INFER_API bool infer_generate_completion(infer_ctx* ctx, const char* raw_prompt, int32_t max_tokens, infer_stream_cb callback, void* user_data);

INFER_API bool infer_set_stop_sequences(infer_ctx* ctx, const char* const* stop_words, size_t num_words);
INFER_API bool infer_set_grammar(infer_ctx* ctx, const char* grammar_gbnf);

INFER_API void infer_memory_clear(infer_ctx* ctx);
INFER_API bool infer_state_save(infer_ctx* ctx, const char* filepath);
INFER_API bool infer_state_load(infer_ctx* ctx, const char* filepath);

INFER_API int32_t infer_get_embedding_dim(infer_ctx* ctx);
INFER_API bool infer_get_embeddings(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len);

INFER_API void infer_abort(infer_ctx* ctx);
INFER_API bool infer_is_generating(infer_ctx* ctx);
INFER_API void infer_wait(infer_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* LIBINFER_H */
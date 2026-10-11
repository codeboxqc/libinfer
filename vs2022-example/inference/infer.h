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
    float    temperature; /**< <= 0.0f enables greedy decoding */
    float    top_p;       /**< 1.0f disables */
    int32_t  top_k;       /**< <= 0 disables */
    uint32_t seed;        /**< 0 for random */
} infer_sampler_params;

/**
 * @brief Represents a single chat message.
 */
typedef struct infer_chat_msg {
    const char* role;
    const char* content;
} infer_chat_msg;

/* Core Lifecycle */
INFER_API infer_sampler_params infer_sampler_default_params(void);
INFER_API infer_ctx* infer_init(void);
INFER_API bool infer_load_model(infer_ctx* ctx, const char* path, uint32_t requested_ctx);
INFER_API bool infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params);
INFER_API void infer_free(infer_ctx* ctx);

/* Generation & Control */
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

INFER_API void infer_abort(infer_ctx* ctx);
INFER_API bool infer_is_generating(infer_ctx* ctx);
INFER_API void infer_wait(infer_ctx* ctx);

/* Diagnostics */
INFER_API const char* infer_get_last_error(infer_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* LIBINFER_H */
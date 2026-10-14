/* ========================================================================== */
/* infer.h                                                                    */
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

/* Bump MAJOR on ABI break; bump MINOR on non-breaking additions/behavior shifts. */
#define INFER_VERSION_MAJOR 2
#define INFER_VERSION_MINOR 1
#define INFER_VERSION_PATCH 0
#define INFER_VERSION_NUM   ((INFER_VERSION_MAJOR << 16) | (INFER_VERSION_MINOR << 8) | (INFER_VERSION_PATCH))

/**
 * @brief Opaque handle representing one inference context (one model, one KV cache).
 *
 * Thread-safety contract
 * ----------------------
 * Always safe from any thread, at any time, and never blocked by engine work:
 *   infer_abort, infer_is_generating, infer_get_last_error, infer_last_success,
 *   infer_get_kv_cache_token_count.
 *
 * Exclusive operations: infer_load_model, infer_set_sampler, infer_set_grammar,
 *   infer_set_stop_sequences, infer_memory_clear, infer_state_save, infer_state_load,
 *   infer_get_embeddings and every infer_generate_* call.
 *   Only one runs at a time. If another is active the call fails immediately without
 *   blocking (returns false; void calls are no-ops) and sets the last error.
 *
 * Query calls: infer_get_model_info, infer_tokenize_count, infer_get_embedding_dim.
 *   These briefly lock engine state and therefore MAY BLOCK while infer_load_model,
 *   infer_get_embeddings, infer_state_save or infer_state_load is running.
 *   They do not wait for text generation.
 *
 * infer_wait joins the async worker started by infer_generate_chat_async; it blocks
 *   until that generation finishes. It does not wait for synchronous generate calls.
 *
 * infer_free aborts any generation, waits for it to finish and for a running
 *   infer_load_model to complete (infer_abort does not interrupt model loading),
 *   then destroys the context. It MUST be the final call on the context: the caller
 *   guarantees no other thread uses the handle during or after infer_free, and that
 *   it is called exactly once.
 *
 * Callbacks run on the calling thread for synchronous calls and on the internal worker
 * thread for infer_generate_chat_async. A callback MUST NOT call infer_free or infer_wait,
 * and MUST NOT call any exclusive operation (the context is busy by definition).
 * infer_abort, infer_get_last_error and the other always-safe calls are allowed.
 */
typedef struct infer_ctx infer_ctx;

/**
 * @brief Streaming token callback.
 * @param piece     NUL-terminated UTF-8 text. Valid only for the duration of the call.
 *                  It may contain several tokens' worth of text (stop-sequence hold-back
 *                  can delay and merge output). A piece is never split inside a UTF-8
 *                  character. Text containing an embedded NUL is truncated at the NUL.
 * @param user_data The pointer passed to the generate call.
 * @return true to continue; false to halt generation immediately. After returning false
 *         the callback is not invoked again for that generation.
 */
typedef bool (*infer_stream_cb)(const char* piece, void* user_data);

/**
 * @brief Log severity. Values match ggml_log_level.
 */
typedef enum infer_log_level {
    INFER_LOG_NONE  = 0,
    INFER_LOG_DEBUG = 1,
    INFER_LOG_INFO  = 2,
    INFER_LOG_WARN  = 3,
    INFER_LOG_ERROR = 4,
    INFER_LOG_CONT  = 5   /**< Continuation of the previous message (same line, no new prefix). */
} infer_log_level;

/**
 * @brief Diagnostic log callback. @p text is NUL-terminated and valid only during the call.
 */
typedef void (*infer_log_cb)(infer_log_level level, const char* text, void* user_data);

/**
 * @brief Sampling parameters for token selection.
 */
typedef struct infer_sampler_params {
    float    temperature;    /**< <= 0.0f enables greedy decoding */
    float    top_p;          /**< 1.0f disables */
    int32_t  top_k;          /**< <= 0 disables */
    float    min_p;          /**< <= 0.0f disables (e.g. 0.05f) */
    float    repeat_penalty; /**< 1.0f disables (e.g. 1.1f) */
    int32_t  repeat_last_n;  /**< <= 0 for default 64 */
    uint32_t seed;           /**< 0 for random seed */
} infer_sampler_params;

/**
 * @brief Represents a single chat message. Both fields must be non-NULL.
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
    char     description[256];/**< Model architecture description */
    uint32_t n_ctx_active;    /**< Context size actually allocated at runtime */
} infer_model_info;

/* ------------------------------------------------------------------------- */
/* Diagnostics & introspection                                               */
/* ------------------------------------------------------------------------- */

/** @brief Returns INFER_VERSION_NUM of the loaded library (compare against the header at startup). */
INFER_API uint32_t infer_version(void);

/**
 * @brief Install (or, with NULL, remove) a log callback for llama/ggml output.
 * Process-global, not per-context. May be called before infer_init. The callback can be
 * invoked from any thread (including the async worker) and may still be invoked briefly
 * after this function returns. It may re-enter the library, but should be fast.
 */
INFER_API void infer_set_log_callback(infer_log_cb callback, void* user_data);

/**
 * @brief Fill @p out_info. Returns false if no model is loaded. May block during
 * model load / embeddings / state save or load (see thread-safety contract).
 */
INFER_API bool infer_get_model_info(infer_ctx* ctx, infer_model_info* out_info);

/**
 * @brief Copy the last error message (NUL-terminated, truncated to fit).
 * @return With @p out_buf == NULL (or @p buf_size == 0): the full message length in bytes.
 *         Otherwise: the number of bytes copied, excluding the NUL. This can be smaller
 *         than the full length, so query the length first to detect truncation.
 * The message is cleared at the start of each generation. An empty message does not
 * prove success; use infer_last_success.
 */
INFER_API size_t infer_get_last_error(infer_ctx* ctx, char* out_buf, size_t buf_size);

/**
 * @brief Result of the most recently STARTED generation (any infer_generate_*).
 * true  = finished normally (end-of-generation token, stop sequence, or max_tokens).
 * false = failed, aborted via infer_abort, halted by the callback, or still running
 *         (for async, read it after infer_wait). Before any generation it returns true.
 * Rejected calls (busy, invalid arguments) do not change it.
 */
INFER_API bool infer_last_success(infer_ctx* ctx);

/* ------------------------------------------------------------------------- */
/* Core lifecycle                                                            */
/* ------------------------------------------------------------------------- */

INFER_API infer_sampler_params infer_sampler_default_params(void);
INFER_API infer_ctx* infer_init(void);

/**
 * @brief Load a model, replacing any previously loaded one. Transactional: on failure the
 * previous model stays active. Blocks for the duration of the load.
 * @param requested_ctx Desired context length. 0 selects 4096. The value is clamped to the
 *        model's trained length and to what fits in free device memory; the granted size
 *        is reported in infer_model_info::n_ctx_active. Minimum context size is 256.
 */
INFER_API bool infer_load_model(infer_ctx* ctx, const char* path, uint32_t requested_ctx);

/** @brief Replace the sampler. May be called before a model is loaded (applied at load). */
INFER_API bool infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params);

/** @brief Destroy the context. See the thread-safety contract: must be the final call. */
INFER_API void infer_free(infer_ctx* ctx);

/* ------------------------------------------------------------------------- */
/* Tokenization & counting                                                   */
/* ------------------------------------------------------------------------- */

/**
 * @brief Count tokens in @p text, parsing special tokens (matches the chat paths).
 * Note: infer_generate_completion and infer_get_embeddings tokenize WITHOUT parsing
 * special tokens, so their counts can differ for text containing special-token strings.
 * @return Token count (>= 0), or -1 on error (no model loaded, NULL arguments).
 */
INFER_API int32_t infer_tokenize_count(infer_ctx* ctx, const char* text, bool add_special);

/** @brief Tokens currently held in the chat sequence of the KV cache. Lock-free. */
INFER_API uint32_t infer_get_kv_cache_token_count(infer_ctx* ctx);

/* ------------------------------------------------------------------------- */
/* Generation                                                                */
/*                                                                           */
/* Common to all infer_generate_* functions:                                 */
/*  - max_tokens: maximum number of tokens to generate. <= 0 selects a       */
/*    safety cap of 4x the active context window.                             */
/*  - Return value: true = finished normally (see infer_last_success).       */
/*    false = invalid arguments, busy, error, infer_abort, or callback       */
/*    returned false; infer_get_last_error describes the reason.             */
/*  - The callback is required (non-NULL).                                   */
/* ------------------------------------------------------------------------- */

/**
 * @brief Stateless chat: renders @p msgs with the model's chat template, CLEARS the KV cache,
 * and generates a reply synchronously. Pass the whole conversation each call.
 */
INFER_API bool infer_generate_chat(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
);

/**
 * @brief Same as infer_generate_chat but runs on an internal worker thread.
 * Returns true only if the work was DISPATCHED (messages are copied; @p msgs need not
 * outlive the call, but @p user_data must outlive the generation). The outcome is
 * available after infer_wait() via infer_last_success() / infer_get_last_error().
 */
INFER_API bool infer_generate_chat_async(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
);

/**
 * @brief Stateful chat: APPENDS one message to the existing KV cache (does not clear it) and
 * generates the reply. Use infer_memory_clear to start a new conversation. When the context
 * fills, older tokens are discarded after a protected opening prefix (the first turn's
 * prompt, up to a quarter of the window).
 */
INFER_API bool infer_generate_chat_turn(
    infer_ctx* ctx,
    const char* role,
    const char* content,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
);

/**
 * @brief Raw text completion with no chat template. CLEARS the KV cache.
 */
INFER_API bool infer_generate_completion(
    infer_ctx* ctx,
    const char* raw_prompt,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
);

/* ------------------------------------------------------------------------- */
/* Stop sequences & grammar                                                  */
/* ------------------------------------------------------------------------- */

/** @brief Replace the stop sequences (NULL/0 clears). Empty strings are ignored. Text is
 * held back until it can no longer be the start of a stop sequence, and the stop text
 * itself is never delivered to the callback. */
INFER_API bool infer_set_stop_sequences(infer_ctx* ctx, const char* const* stop_words, size_t num_words);

/** @brief Set a GBNF grammar (root rule "root"); NULL or "" clears it. Transactional:
 * an invalid grammar is rejected and the previous one stays active. */
INFER_API bool infer_set_grammar(infer_ctx* ctx, const char* grammar_gbnf);

/* ------------------------------------------------------------------------- */
/* KV cache & session state                                                  */
/* ------------------------------------------------------------------------- */

/** @brief Clear the KV cache and reset the sampler (starts a new conversation). */
INFER_API void infer_memory_clear(infer_ctx* ctx);

/**
 * @brief Save the KV cache to @p filepath (UTF-8 path). Written to a unique temp file and then
 * renamed over the destination, so a failed save never destroys an existing file.
 * State files are compatible only with the same model dimensions and vocabulary.
 */
INFER_API bool infer_state_save(infer_ctx* ctx, const char* filepath);

/** @brief Restore a state saved by infer_state_save. The current context must be at
 * least as large as the number of saved tokens. */
INFER_API bool infer_state_load(infer_ctx* ctx, const char* filepath);

/* ------------------------------------------------------------------------- */
/* Embeddings                                                                */
/* ------------------------------------------------------------------------- */

/** @brief Embedding dimension of the loaded model, or 0 if none. */
INFER_API int32_t infer_get_embedding_dim(infer_ctx* ctx);

/**
 * @brief Compute an embedding for @p text into @p out_vec (needs @p vec_len >= the embedding
 * dimension). The vector is NOT normalized.
 */
INFER_API bool infer_get_embeddings(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len);

/* ------------------------------------------------------------------------- */
/* Async & thread synchronization                                            */
/* ------------------------------------------------------------------------- */

/** @brief Request the active generation (sync or async) to stop. Safe from any thread. */
INFER_API void infer_abort(infer_ctx* ctx);

/** @brief True while an exclusive operation or generation is in progress. */
INFER_API bool infer_is_generating(infer_ctx* ctx);

/** @brief Block until the async worker (if any) has finished. See thread-safety contract. */
INFER_API void infer_wait(infer_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* LIBINFER_H */
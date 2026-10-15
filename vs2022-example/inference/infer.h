/* ========================================================================== */
/* infer.h - Production C/C++ Inference API                                   */
/* ========================================================================== */
/**
 * @file infer.h
 * @brief Thread-safe C API wrapper for llama.cpp model lifecycle, chat,
 *        completion, state persistence, and embedding extraction.
 *
 * @section build_sec Build & Symbol Visibility
 * - **Shared Library (Windows)**: Define `INFER_BUILD_SHARED` when compiling the
 *   DLL to export symbols (`__declspec(dllexport)`). Consumers linking against
 *   the DLL require no special macro (`__declspec(dllimport)` is the default).
 * - **Static Library (All Platforms)**: Define `INFER_STATIC` in both the
 *   library build and consumer targets to disable DLL import/export attributes.
 * - **ELF/Mach-O Visibility (Linux/macOS)**: Compile with `-fvisibility=hidden`
 *   and `-fvisibility-inlines-hidden` so only `INFER_API` entry points are
 *   exported from the shared object.
 *
 * @section thread_sec Threading & Concurrency Model
 * - **Context Serialization**: A single `infer_ctx` instance serializes all
 *   mutating and generation operations internally. Concurrent generation requests
 *   on the same `infer_ctx` will immediately return `false` with a busy error.
 * - **Cross-Thread Control**: `infer_abort()`, `infer_is_generating()`,
 *   `infer_wait()`, `infer_get_last_error()`, `infer_last_success()`, and
 *   `infer_get_kv_cache_token_count()` are safe to call concurrently from any
 *   non-callback thread.
 * - **Stream Callback Contract**:
 *   1. The callback is invoked synchronously on the thread executing decode
 *      (the caller's thread for synchronous APIs, or the internal worker thread
 *      for `infer_generate_chat_async*`).
 *   2. The `piece` pointer is valid ONLY for the duration of the callback call.
 *   3. Returning `false` from the callback immediately halts generation and
 *      marks the generation as aborted.
 *   4. **NEVER** call `infer_free()` or `infer_wait()` from inside a stream
 *      callback. Both detect self-invocation from the generation thread and
 *      refuse to block to prevent self-deadlock.
 * - **Async Lifecycle**: After starting `infer_generate_chat_async()`, call
 *   `infer_wait()` to synchronize completion, then check `infer_last_success()`
 *   and `infer_get_last_error()` for the outcome.
 * - **Ownership**: `infer_init()` / `infer_init_ex()` transfer ownership of the
 *   returned `infer_ctx*` to the caller, which must eventually release it via
 *   `infer_free()` from a non-callback thread.
 */

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

#if defined(__cplusplus) && __cplusplus >= 201703L
    #define INFER_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
    #define INFER_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER) && _MSC_VER >= 1700
    #define INFER_NODISCARD _Check_return_
#else
    #define INFER_NODISCARD
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define INFER_STR_HELPER(x) #x
#define INFER_STR(x)        INFER_STR_HELPER(x)

#define INFER_VERSION_MAJOR 2
#define INFER_VERSION_MINOR 3
#define INFER_VERSION_PATCH 0
#define INFER_VERSION_NUM   ((INFER_VERSION_MAJOR << 16) | (INFER_VERSION_MINOR << 8) | (INFER_VERSION_PATCH))
#define INFER_VERSION_STR   INFER_STR(INFER_VERSION_MAJOR) "." INFER_STR(INFER_VERSION_MINOR) "." INFER_STR(INFER_VERSION_PATCH)

/** Opaque inference context handle. */
typedef struct infer_ctx infer_ctx;

/**
 * @brief Legacy NUL-terminated streaming token callback.
 * @param piece     NUL-terminated valid UTF-8 slice. Valid only during the call.
 * @param user_data Opaque pointer supplied by the caller.
 * @return `true` to continue generation, `false` to abort immediately.
 */
typedef bool (*infer_stream_cb)(const char* piece, void* user_data);

/**
 * @brief Binary-safe streaming token callback with explicit byte length.
 * @param piece     Pointer to valid UTF-8 bytes (also NUL-terminated for convenience).
 *                  Valid only during the call.
 * @param len       Byte length of @p piece (excluding the trailing NUL).
 * @param user_data Opaque pointer supplied by the caller.
 * @return `true` to continue generation, `false` to abort immediately.
 */
typedef bool (*infer_stream_cb_v2)(const char* piece, size_t len, void* user_data);

/** Log severity levels forwarded from the backend. */
typedef enum infer_log_level {
    INFER_LOG_NONE  = 0,
    INFER_LOG_DEBUG = 1,
    INFER_LOG_INFO  = 2,
    INFER_LOG_WARN  = 3,
    INFER_LOG_ERROR = 4,
    INFER_LOG_CONT  = 5
} infer_log_level;

/**
 * @brief Global log callback type.
 * @param level     Log severity level.
 * @param text      NUL-terminated log message string.
 * @param user_data Opaque user pointer registered with `infer_set_log_callback()`.
 */
typedef void (*infer_log_cb)(infer_log_level level, const char* text, void* user_data);

/** KV cache element data types for `infer_model_params`. */
typedef enum infer_kv_type {
    INFER_KV_TYPE_DEFAULT = 0, /**< Default (F16) */
    INFER_KV_TYPE_F16     = 1, /**< 16-bit floating point */
    INFER_KV_TYPE_BF16    = 2, /**< 16-bit brain floating point */
    INFER_KV_TYPE_Q8_0    = 3, /**< 8-bit quantized KV */
    INFER_KV_TYPE_Q4_0    = 4  /**< 4-bit quantized KV */
} infer_kv_type;

/** Sampler configuration parameters. Initialize via `infer_sampler_default_params()`. */
typedef struct infer_sampler_params {
    uint32_t struct_size;    /**< Must be set to `sizeof(infer_sampler_params)` */
    float    temperature;    /**< `<= 0.0f` enables greedy decoding */
    float    top_p;          /**< `1.0f` or `<= 0.0f` disables nucleus sampling */
    int32_t  top_k;          /**< `<= 0` disables top-k sampling */
    float    min_p;          /**< `<= 0.0f` or `>= 1.0f` disables min-p sampling */
    float    repeat_penalty; /**< `1.0f` or `<= 0.0f` disables repetition penalty (default: `1.0f`) */
    int32_t  repeat_last_n;  /**< `<= 0` defaults to 64; clamped to active `n_ctx` */
    uint32_t seed;           /**< `0` selects a default/random seed */
} infer_sampler_params;

/** Extended model and context loading parameters. Initialize via `infer_model_default_params()`. */
typedef struct infer_model_params {
    uint32_t    struct_size;         /**< Must be set to `sizeof(infer_model_params)` */
    uint32_t    requested_ctx;       /**< Desired context size (`0` = auto-scale to VRAM/RAM budget) */
    int32_t     n_gpu_layers;        /**< Number of layers to offload to GPU (`< 0` = all, `0` = CPU only) */
    bool        use_mmap;            /**< Memory-map model weights where supported (default: `true`) */
    bool        use_mlock;           /**< Lock model weights in RAM to prevent paging (default: `false`) */
    bool        flash_attn;          /**< Enable Flash Attention when supported (default: `false`) */
    int32_t     kv_type_k;           /**< Key cache quantization (`infer_kv_type`, default: `INFER_KV_TYPE_F16`) */
    int32_t     kv_type_v;           /**< Value cache quantization (`infer_kv_type`, default: `INFER_KV_TYPE_F16`) */
    bool        allow_ctx_reduction; /**< If `true`, allows shrinking an explicit `requested_ctx` on OOM */
    const char* chat_template;       /**< Optional custom Jinja/named chat template override (`NULL` = model default) */
} infer_model_params;

/** Single chat message entry. */
typedef struct infer_chat_msg {
    const char* role;    /**< Whitelisted: `"system"`, `"user"`, `"assistant"`, `"tool"`, `"developer"` */
    const char* content; /**< UTF-8 message body */
} infer_chat_msg;

/** Model metadata and active context dimensions. Initialize `struct_size` before calling `infer_get_model_info()`. */
typedef struct infer_model_info {
    uint32_t struct_size;      /**< Set to `sizeof(infer_model_info)` before querying */
    uint32_t n_ctx_train;      /**< Model native training context length */
    uint32_t n_embd;           /**< Embedding dimension size */
    uint32_t n_layer;          /**< Number of transformer layers */
    uint32_t n_head;           /**< Number of attention query heads */
    uint32_t vocab_size;       /**< Number of tokens in vocabulary */
    char     description[256]; /**< NUL-terminated human-readable model description */
    uint32_t n_ctx_active;     /**< Actual allocated context length (`n_ctx`) */
} infer_model_info;

/** @return Packed version integer `((MAJOR << 16) | (MINOR << 8) | PATCH)`. */
INFER_API uint32_t infer_version(void);

/**
 * @brief Registers or clears a global logging callback. Thread-safe.
 * @param callback  Function pointer, or `NULL` to silence backend logs.
 * @param user_data Opaque pointer passed back to @p callback.
 */
INFER_API void infer_set_log_callback(infer_log_cb callback, void* user_data);

/**
 * @brief Populates @p out_info with active model metadata.
 * @param ctx      Initialized context with a loaded model.
 * @param out_info Output struct; caller MUST set `out_info->struct_size = sizeof(infer_model_info)` prior to calling.
 * @return `true` on success, `false` if invalid or no model is loaded.
 */
INFER_API INFER_NODISCARD bool infer_get_model_info(infer_ctx* ctx, infer_model_info* out_info);

/**
 * @brief Retrieves the last error string recorded on @p ctx.
 * @param ctx      Context handle.
 * @param out_buf  Destination buffer, or `NULL` to query required length.
 * @param buf_size Size of @p out_buf in bytes.
 * @return Full byte length of the error message (excluding the NUL terminator), matching `snprintf` semantics.
 */
INFER_API size_t infer_get_last_error(infer_ctx* ctx, char* out_buf, size_t buf_size);

/**
 * @brief Returns whether the most recent generation or embedding call on @p ctx succeeded.
 *        Especially useful after `infer_generate_chat_async()` + `infer_wait()`.
 */
INFER_API bool infer_last_success(infer_ctx* ctx);

/** @return Default sampler parameters (`repeat_penalty = 1.0f`, `temperature = 0.7f`). */
INFER_API infer_sampler_params infer_sampler_default_params(void);

/** @return Default model loading parameters (`n_gpu_layers = -1`, `use_mmap = true`, F16 KV cache). */
INFER_API infer_model_params infer_model_default_params(void);

/**
 * @brief Initializes the backend and allocates a new `infer_ctx`.
 * @return Newly allocated context handle, or `NULL` on failure.
 */
INFER_API INFER_NODISCARD infer_ctx* infer_init(void);

/**
 * @brief Initializes the backend and allocates a new `infer_ctx`, writing diagnostic text on failure.
 * @param err_buf      Optional buffer to receive a NUL-terminated error message on failure.
 * @param err_buf_size Capacity of @p err_buf in bytes.
 * @return Newly allocated context handle, or `NULL` on failure.
 */
INFER_API INFER_NODISCARD infer_ctx* infer_init_ex(char* err_buf, size_t err_buf_size);

/**
 * @brief Loads a GGUF model from @p path using default model parameters.
 *        Frees any previously loaded model on @p ctx before allocating the new one.
 * @param ctx           Context handle.
 * @param path          UTF-8 file path to the `.gguf` file.
 * @param requested_ctx Target context size (`0` to auto-size based on free memory; if non-zero,
 *                      returns `false` if the requested size cannot fit).
 * @return `true` on success, `false` on error.
 */
INFER_API INFER_NODISCARD bool infer_load_model(infer_ctx* ctx, const char* path, uint32_t requested_ctx);

/**
 * @brief Loads a GGUF model from @p path with explicit GPU offload, mmap, mlock, KV quantization, and Flash Attention controls.
 * @param ctx    Context handle.
 * @param path   UTF-8 file path to the `.gguf` file.
 * @param params Pointer to initialized `infer_model_params` (`struct_size` must be non-zero).
 * @return `true` on success, `false` on error.
 */
INFER_API INFER_NODISCARD bool infer_load_model_ex(infer_ctx* ctx, const char* path, const infer_model_params* params);

/**
 * @brief Unloads the active model and releases all associated VRAM/RAM buffers on @p ctx.
 * @return `true` on success, `false` if @p ctx is null or currently busy generating.
 */
INFER_API INFER_NODISCARD bool infer_unload_model(infer_ctx* ctx);

/**
 * @brief Updates the active sampler chain on @p ctx.
 * @param ctx    Context handle.
 * @param params Pointer to `infer_sampler_params` with `struct_size > 0`.
 * @return `true` on success, `false` on validation failure or busy context.
 */
INFER_API INFER_NODISCARD bool infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params);

/**
 * @brief Overrides the chat template used by `infer_generate_chat*` and `infer_generate_chat_turn*`.
 * @param ctx             Context handle.
 * @param custom_template Jinja/named chat template string, or `NULL` / `""` to revert to the model's built-in template.
 * @return `true` on success, `false` if busy.
 */
INFER_API INFER_NODISCARD bool infer_set_chat_template(infer_ctx* ctx, const char* custom_template);

/**
 * @brief Aborts any in-flight generation, waits for completion, frees all model/context resources,
 *        and destroys @p ctx. Must NOT be called from inside a stream callback.
 */
INFER_API void infer_free(infer_ctx* ctx);

/**
 * @brief Counts the number of tokens produced by tokenizing @p text after special-token neutralization.
 * @return Token count (`>= 0`), or `-1` on error.
 */
INFER_API int32_t infer_tokenize_count(infer_ctx* ctx, const char* text, bool add_special);

/** @return Number of tokens currently stored in sequence 0 of the KV cache. */
INFER_API uint32_t infer_get_kv_cache_token_count(infer_ctx* ctx);

/**
 * @brief Runs synchronous chat generation over @p msgs.
 *        Populates internal chat history upon completion so subsequent `infer_generate_chat_turn()`
 *        calls can seamlessly continue the conversation.
 * @param max_tokens Maximum tokens to generate (`<= 0` caps at active `n_ctx`).
 */
INFER_API INFER_NODISCARD bool infer_generate_chat(infer_ctx* ctx, const infer_chat_msg* msgs, size_t num_msgs, int32_t max_tokens, infer_stream_cb callback, void* user_data);

/** Binary-safe v2 callback variant of `infer_generate_chat()`. */
INFER_API INFER_NODISCARD bool infer_generate_chat_v2(infer_ctx* ctx, const infer_chat_msg* msgs, size_t num_msgs, int32_t max_tokens, infer_stream_cb_v2 callback, void* user_data);

/**
 * @brief Launches asynchronous chat generation on a background worker thread.
 *        Use `infer_wait()` followed by `infer_last_success()` to check completion status.
 */
INFER_API INFER_NODISCARD bool infer_generate_chat_async(infer_ctx* ctx, const infer_chat_msg* msgs, size_t num_msgs, int32_t max_tokens, infer_stream_cb callback, void* user_data);

/** Binary-safe v2 callback variant of `infer_generate_chat_async()`. */
INFER_API INFER_NODISCARD bool infer_generate_chat_async_v2(infer_ctx* ctx, const infer_chat_msg* msgs, size_t num_msgs, int32_t max_tokens, infer_stream_cb_v2 callback, void* user_data);

/**
 * @brief Appends a single turn (`role`, `content`) to the active conversation, pruning whole
 *        historical turns if needed, and streams the assistant response while reusing the KV prefix.
 *        Note: Once the prompt is prefilled into the KV cache, the user turn remains in history
 *        even if generation is subsequently aborted by the caller.
 */
INFER_API INFER_NODISCARD bool infer_generate_chat_turn(infer_ctx* ctx, const char* role, const char* content, int32_t max_tokens, infer_stream_cb callback, void* user_data);

/** Binary-safe v2 callback variant of `infer_generate_chat_turn()`. */
INFER_API INFER_NODISCARD bool infer_generate_chat_turn_v2(infer_ctx* ctx, const char* role, const char* content, int32_t max_tokens, infer_stream_cb_v2 callback, void* user_data);

/**
 * @brief Runs raw text completion without chat template formatting or special-token neutralization.
 * @param max_tokens Maximum tokens to generate (`<= 0` caps at active `n_ctx`).
 */
INFER_API INFER_NODISCARD bool infer_generate_completion(infer_ctx* ctx, const char* raw_prompt, int32_t max_tokens, infer_stream_cb callback, void* user_data);

/** Binary-safe v2 callback variant of `infer_generate_completion()`. */
INFER_API INFER_NODISCARD bool infer_generate_completion_v2(infer_ctx* ctx, const char* raw_prompt, int32_t max_tokens, infer_stream_cb_v2 callback, void* user_data);

/**
 * @brief Configures stop sequences that halt generation and roll back trailing stop tokens.
 *        Every stop sequence must be non-null and between 1 and 256 bytes in length.
 */
INFER_API INFER_NODISCARD bool infer_set_stop_sequences(infer_ctx* ctx, const char* const* stop_words, size_t num_words);

/**
 * @brief Sets or clears (`NULL` / `""`) a GBNF grammar constraint on the sampler chain.
 */
INFER_API INFER_NODISCARD bool infer_set_grammar(infer_ctx* ctx, const char* grammar_gbnf);

/**
 * @brief Clears the KV cache, resets sampler state, and empties stored chat history.
 * @return `true` on success, `false` if @p ctx is null, busy, or encounters an internal error.
 */
INFER_API INFER_NODISCARD bool infer_memory_clear(infer_ctx* ctx);

/**
 * @brief Atomically saves KV cache state and chat history to @p filepath with permissions `0600`
 *        and full-file CRC32 integrity verification.
 */
INFER_API INFER_NODISCARD bool infer_state_save(infer_ctx* ctx, const char* filepath);

/**
 * @brief Restores KV cache state and validated/neutralized chat history from @p filepath.
 */
INFER_API INFER_NODISCARD bool infer_state_load(infer_ctx* ctx, const char* filepath);

/** @return Embedding dimension (`n_embd`) of the loaded model, or `0` if none. */
INFER_API int32_t infer_get_embedding_dim(infer_ctx* ctx);

/**
 * @brief Computes embeddings for @p text into @p out_vec.
 *        Note: Output vectors are unnormalized raw model embeddings (for `LLAMA_POOLING_TYPE_NONE`,
 *        the last token's unnormalized hidden state is returned). Use `infer_get_embeddings_ex()`
 *        to optionally L2-normalize the output vector.
 */
INFER_API INFER_NODISCARD bool infer_get_embeddings(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len);

/**
 * @brief Computes embeddings for @p text into @p out_vec, with optional L2 normalization.
 */
INFER_API INFER_NODISCARD bool infer_get_embeddings_ex(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len, bool normalize);

/** Signals an asynchronous abort to any in-flight generation or embedding call on @p ctx. */
INFER_API void infer_abort(infer_ctx* ctx);

/** @return `true` if a generation or mutating operation is currently active on @p ctx. */
INFER_API bool infer_is_generating(infer_ctx* ctx);

/**
 * @brief Blocks until any active synchronous or asynchronous generation on @p ctx completes.
 *        Must NOT be called from inside a stream callback.
 */
INFER_API void infer_wait(infer_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* LIBINFER_H */
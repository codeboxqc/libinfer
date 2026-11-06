# libinfer

[![Build libinfer](https://github.com/codeboxqc/libinfer/actions/workflows/build.yml/badge.svg)](https://github.com/codeboxqc/libinfer/actions)
[![Latest Release](https://img.shields.io/github/v/release/codeboxqc/libinfer?include_prereleases&color=blue)](https://github.com/codeboxqc/libinfer/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


libinfer-windows-x64.zip 
 https://github.com/codeboxqc/libinfer/releases/download/latest/libinfer-windows-x64.zip

 
libinfer-macos-arm64.zip
https://github.com/codeboxqc/libinfer/releases/download/latest/libinfer-macos-arm64.zip


ibinfer-linux-x64.zip
https://github.com/codeboxqc/libinfer/releases/download/latest/libinfer-linux-x64.zip


**`libinfer`** is a high-performance, lightweight, cross-platform dynamic inference library for Large Language Models (LLMs) built on top of [`llama.cpp`](https://github.com/ggerganov/llama.cpp). 

It exposes a simple C ABI interface (`infer.h`), making it easy to embed real-time local LLM inference, streaming token callbacks, and context shifting into C, C++, Rust, Python, Go, or GUI applications across **Windows**, **Linux**, and **macOS**.

---

 # libinfer

**libinfer** is a production-ready, thread-safe C/C++ inference API wrapper around `llama.cpp`. It is designed to provide a robust, easy-to-use interface for managing model lifecycles, chat generation, text completion, state persistence, and embedding extraction with automatic GPU/CPU scaling.

## Features
* **Thread-Safe**: Safely call abort, wait, and state checks from any thread.
* **Auto GPU/CPU Scaling**: Automatically detects system RAM/VRAM and distributes layers dynamically.
* **State Persistence**: Save and load KV cache and chat history to disk with CRC32 verification.
* **Async Generation**: Built-in background worker thread for non-blocking UI integration.
* **Streaming Callbacks**: Real-time token streaming with safe abortion hooks.
* **Advanced Sampling**: Full support for Top-K, Top-P, Min-P, GBNF grammars, and repetition penalties.

## Build and Install

This project uses CMake and optionally vcpkg.

```bash
# Clone with submodules (required for llama.cpp)
git clone --recursive https://github.com/codeboxqc/libinfer.git
cd libinfer

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

---

## API Reference (Wiki)

Here is a detailed breakdown of all functions available in `infer.h`. 

### Initialization & Lifecycle
* `infer_init()` / `infer_init_ex()`: Initializes the backend and allocates a new inference context (`infer_ctx`).
* `infer_free(infer_ctx* ctx)`: Safely aborts any running generation, clears memory, and destroys the context.
* `infer_version()`: Returns the compiled version integer.

### Model Management
* `infer_load_model(infer_ctx* ctx, const char* path, uint32_t requested_ctx)`: Loads a `.gguf` model using default auto-scaling parameters.
* `infer_load_model_ex(infer_ctx* ctx, const char* path, const infer_model_params* params)`: Loads a model with explicit GPU offload, mmap, mlock, and Flash Attention settings.
* `infer_unload_model(infer_ctx* ctx)`: Unloads the active model and releases VRAM/RAM without destroying the context.
* `infer_get_model_info(infer_ctx* ctx, infer_model_info* out_info)`: Retrieves model metadata (layers, dimensions, active context).

### Configuration
* `infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params)`: Updates the sampler chain (temperature, top-p, min-p, etc.).
* `infer_set_chat_template(infer_ctx* ctx, const char* custom_template)`: Overrides the built-in Jinja chat template.
* `infer_set_stop_sequences(infer_ctx* ctx, const char* const* stop_words, size_t num_words)`: Configures custom stop words.
* `infer_set_grammar(infer_ctx* ctx, const char* grammar_gbnf)`: Constrains output using a GBNF grammar string.
* `infer_set_log_callback(infer_log_cb callback, void* user_data)`: Hooks into the backend logging system.

### Generation & Chat
* `infer_generate_chat(...)` / `infer_generate_chat_v2(...)`: Synchronous chat generation over an array of messages.
* `infer_generate_chat_async(...)` / `infer_generate_chat_async_v2(...)`: Non-blocking chat generation on a background thread.
* `infer_generate_chat_turn(...)` / `infer_generate_chat_turn_v2(...)`: Appends a single turn to the ongoing history, reusing the KV cache prefix automatically.
* `infer_generate_completion(...)` / `infer_generate_completion_v2(...)`: Raw text completion without chat templates.

### State & Memory Management
* `infer_memory_clear(infer_ctx* ctx)`: Clears the KV cache and empties the stored chat history.
* `infer_state_save(infer_ctx* ctx, const char* filepath)`: Atomically saves KV cache and chat history to disk.
* `infer_state_load(infer_ctx* ctx, const char* filepath)`: Restores KV cache and chat history from disk.
* `infer_get_kv_cache_token_count(infer_ctx* ctx)`: Returns the number of active tokens in the KV cache.

### Embeddings
* `infer_get_embeddings(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len)`: Computes raw unnormalized embeddings.
* `infer_get_embeddings_ex(...)`: Computes embeddings with optional L2 normalization.
* `infer_get_embedding_dim(infer_ctx* ctx)`: Returns the embedding dimension (`n_embd`) of the loaded model.

### Concurrency & Synchronization
* `infer_abort(infer_ctx* ctx)`: Asynchronously halts any in-flight generation.
* `infer_wait(infer_ctx* ctx)`: Blocks until active async generation completes.
* `infer_is_generating(infer_ctx* ctx)`: Returns `true` if the context is currently busy.
* `infer_last_success(infer_ctx* ctx)`: Returns `true` if the last generation/embedding task completed without errors.
* `infer_get_last_error(infer_ctx* ctx, char* out_buf, size_t buf_size)`: Retrieves the last error message string.

---

## Minimal Examples

### 1. Initializing and Loading a Model

This example demonstrates how to spin up a context and load a model with an automatic context size (passing `0` allows it to auto-size based on free RAM/VRAM).

 c
#include <stdio.h>
#include "infer.h"

int main() {
    // 1. Initialize the inference context
    infer_ctx* ctx = infer_init();
    if (!ctx) {
        printf("Failed to initialize libinfer.\n");
        return 1;
    }

    // 2. Load a model (requesting auto-sized context by passing 0)
    printf("Loading model...\n");
    if (!infer_load_model(ctx, "models/llama3-8b.gguf", 0)) {
        char err[256];
        infer_get_last_error(ctx, err, sizeof(err));
        printf("Error loading model: %s\n", err);
        infer_free(ctx);
        return 1;
    }

    printf("Model loaded successfully!\n");
    
    // 3. Clean up
    infer_free(ctx);
    return 0;
}
 

### 2. Synchronous Chat Generation with Streaming

This example shows how to set up a stream callback to print tokens as they are generated, and how to execute a synchronous chat completion.

 c
#include <stdio.h>
#include <string.h>
#include "infer.h"

// Stream callback to print generated pieces in real-time
bool stream_callback(const char* piece, void* user_data) {
    printf("%s", piece);
    fflush(stdout);
    return true; // Return false to abort generation
}

int main() {
    infer_ctx* ctx = infer_init();
    infer_load_model(ctx, "models/llama3-8b.gguf", 4096);

    // Setup chat messages
    infer_chat_msg msgs[] = {
        {"system", "You are a helpful AI assistant."},
        {"user", "Write a haiku about C++ programming."}
    };

    printf("Assistant: ");
    
    // Generate chat (max 100 tokens)
    bool success = infer_generate_chat(
        ctx, 
        msgs, 
        2,             // Number of messages
        100,           // Max tokens to generate
        stream_callback, 
        NULL           // User data
    );

    printf("\n");

    if (!success) {
        char err[256];
        infer_get_last_error(ctx, err, sizeof(err));
        printf("Generation failed: %s\n", err);
    }

    infer_free(ctx);
    return 0;
}
 

### 3. Extracting Embeddings

This example demonstrates how to extract L2-normalized embeddings for a string of text.

c
#include <stdio.h>
#include <stdlib.h>
#include "infer.h"

int main() {
    infer_ctx* ctx = infer_init();
    
    // Note: The loaded model must support embeddings
    infer_load_model(ctx, "models/embedding-model.gguf", 1024);

    // Get the required vector dimension
    int32_t dim = infer_get_embedding_dim(ctx);
    if (dim <= 0) {
        printf("Model does not support embeddings.\n");
        infer_free(ctx);
        return 1;
    }

    // Allocate memory for the output vector
    float* embd = (float*)malloc(dim * sizeof(float));
    
    // Extract normalized embeddings (passing 'true' for normalize)
    const char* text_to_embed = "This is a test document.";
    if (infer_get_embeddings_ex(ctx, text_to_embed, embd, dim, true)) {
        printf("Successfully generated embedding of size %d.\n", dim);
        printf("First 3 values: [%f, %f, %f]\n", embd[0], embd[1], embd[2]);
    } else {
        char err[256];
        infer_get_last_error(ctx, err, sizeof(err));
        printf("Failed to get embeddings: %s\n", err);
    }

    free(embd);
    infer_free(ctx);
    return 0;
}
 

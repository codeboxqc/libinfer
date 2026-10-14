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

## 🌟 Key Features

* 🔌 **Clean C ABI Boundary (`infer.h`):** Simple C-compatible API for easy integration into any language.
* ⚡ **Streaming Token Callbacks:** Real-time token output streaming with early abort control.
* 🔄 **Async & Non-Blocking Generation:** Background thread worker dispatch (`infer_generate_chat_async`).
* 🧠 **Smart Memory & VRAM Clamping:** Automatically queries hardware VRAM to safe-clamp token context size (`n_ctx`).
* ⚙️ **Dynamic Backend Loading:** Dynamic load support for `ggml-cuda` dynamic modules on hardware accelerators.
* 📦 **vcpkg & Submodule Ready:** Native support for both `vcpkg.json` manifest mode and CMake submodules.

---

## 🏛️ Shared Libraries & DLL Structure

When you build or download `libinfer`, the compilation produces standard platform dynamic libraries:

| Platform | Runtime Dynamic Library | Import Library (Compile-time) | Backend Modules (Optional) |
| :--- | :--- | :--- | :--- |
| **Windows** | `infer.dll` | `infer.lib` *(Required for MSVC)* | `ggml-cuda.dll` |
| **Linux** | `libinfer.so` | *(Linked directly to `libinfer.so`)* | `ggml-cuda.so` |
| **macOS** | `libinfer.dylib` | *(Linked directly to `libinfer.dylib`)* | Metal backend built-in |

> **Important for Windows Users:**  
> When compiling applications against `infer.dll` using MSVC, you **must** link against `infer.lib` at compile time and keep `infer.dll` alongside your compiled executable at runtime.

---

## 📦 Compatible GGUF Models

`libinfer` supports all GGUF model architectures supported by `llama.cpp` (e.g. Llama 3, Qwen 2.5, DeepSeek R1, Phi-3, Mistral).

### Recommended Download Links

| Model | Model File | Size / VRAM | Download Link |
| :--- | :--- | :--- | :--- |
| **Qwen 2.5 Coder 7B** *(Coding)* | `qwen2.5-coder-7b-instruct-q4_k_m.gguf` | ~4.7 GB | [HuggingFace Link](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/blob/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf) |
| **Llama 3.2 3B** *(Lightweight)* | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | ~2.0 GB | [HuggingFace Link](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| **Llama 3.1 8B** *(General)* | `Llama-3.1-8B-Instruct-Q4_K_M.gguf` | ~4.9 GB | [HuggingFace Link](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) |
| **DeepSeek R1 Distill Qwen 7B** *(Reasoning)* | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | ~4.7 GB | [HuggingFace Link](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF) |
| **Phi 3.5 Mini** *(Mobile/Fast)* | `Phi-3.5-mini-instruct-Q4_K_M.gguf` | ~2.4 GB | [HuggingFace Link](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) |

* **Recommended Quantization Types:** `Q4_K_M` (Best balance of speed and memory), `Q5_K_M`, `Q8_0` (Highest accuracy).
* 🔍 **Browse all GGUF models on Hugging Face:** [HuggingFace GGUF Models](https://huggingface.co/models?library=gguf)

---


<img width="709" height="301" alt="2" src="https://github.com/user-attachments/assets/37afa1bb-3d0f-451d-a063-f79921a6e72b" />

you 9 dll next to your exe!
<img width="537" height="476" alt="Untitled" src="https://github.com/user-attachments/assets/efe17356-e195-4512-8da3-ddc4e4d34908" />



## 🚀 Quick Start Guide

### Step 1: Get the Pre-Built Binaries
Download pre-compiled binaries for your operating system directly from **[GitHub Releases](https://github.com/codeboxqc/libinfer/releases)**:
* `libinfer-windows-x64.zip`
* `libinfer-linux-x64.zip`
* `libinfer-macos-arm64.zip`

---

### Step 2: Write Your Application (`main.c`)

Create a file named `main.c`:


IF it slow you need bigger NVEDIA
you canuse any ai model .gguf   
This example is using qwen2.5-coder-7b-instruct-q4_k_m.gguf
https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/blob/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "infer.h"

// Automatically link infer.lib when compiling with MSVC on Windows
#if defined(_WIN32) && defined(_MSC_VER)
#pragma comment(lib, "infer.lib")
#endif

// 1. Streaming Callback: Prints generated code/text tokens as they arrive
static bool on_token_stream(const char* piece, void* user_data) {
    (void)user_data;
    printf("%s", piece);
    fflush(stdout); // Real-time terminal output
    return true;    // Return true to continue, false to interrupt
}

int main(int argc, char** argv) {
    // Default model filename and coding prompt
    const char* model_path = (argc >= 2) ? argv[1] : "qwen2.5-coder-7b-instruct-q4_k_m.gguf";
    const char* user_prompt = (argc >= 3) ? argv[2] :
        "Write a fast C function that performs binary search on a sorted integer array. Include short usage comments.";

    printf("=======================================================\n");
    printf(" Running libinfer with Qwen 2.5 Coder\n");
    printf(" Model:  %s\n", model_path);
    printf(" Prompt: %s\n", user_prompt);
    printf("=======================================================\n\n");

    // 2. Initialize inference context
    infer_ctx* ctx = infer_init();
    if (!ctx) {
        fprintf(stderr, "Error: Failed to initialize infer context.\n");
        return 1;
    }

    // 3. Load Qwen2.5-Coder GGUF Model (2048 token context window)
    printf("Loading model weights into memory/GPU...\n");
    if (!infer_load_model(ctx, model_path, 2048)) {
        fprintf(stderr, "\nFailed to load model '%s':\n%s\n",
            model_path, infer_get_last_error(ctx));
        infer_free(ctx);
        return 1;
    }

    // 4. Set optimal sampling parameters for code generation (Low temperature for precision)
    infer_sampler_params sparams = infer_sampler_default_params();
    sparams.temperature = 0.20f;  // Lower temperature produces more precise code
    sparams.top_p = 0.95f;
    sparams.top_k = 40;
    sparams.seed = 42;
    infer_set_sampler(ctx, &sparams);

    // 5. Structure conversation with system prompt optimized for coding
    infer_chat_msg chat[] = {
        { "system", "You are an expert C/C++ coding assistant. Provide concise, clean, and bug-free code solutions." },
        { "user",   user_prompt }
    };
    size_t num_msgs = sizeof(chat) / sizeof(chat[0]);

    printf("\n--- Qwen2.5 Coder Response ---\n\n");

    // 6. Run synchronous chat generation with streaming output
    bool success = infer_generate_chat(ctx, chat, num_msgs, on_token_stream, NULL);

    printf("\n\n-------------------------------\n");

    if (!success) {
        fprintf(stderr, "Inference error: %s\n", infer_get_last_error(ctx));
    }

    // 7. Cleanup resources
    infer_free(ctx);
    printf("Inference context freed successfully.\n");

    return 0;
}












# libinfer API Documentation

**`libinfer`** is a streamlined C/C++ inference engine wrapping `llama.cpp`. It provides hardware-accelerated LLM execution (via CUDA or Metal) with advanced features like persistent KV caching for multi-turn chat, asynchronous generation, grammar constraints, and embedding extraction.

## 1. Core Lifecycle & Initialization

* **`infer_init()`**
  Initializes the GGML backend and allocates an empty `infer_ctx` (inference context) handle.

* **`infer_load_model(ctx, path, requested_ctx)`**
  Loads a `.gguf` model file from disk into RAM/VRAM. It automatically probes your hardware and shrinks `requested_ctx` if you do not have enough VRAM available.

* **`infer_free(ctx)`**
  Safely aborts any running generation, clears VRAM, and destroys the context.

---

## 2. Text Generation & Chat

* **`infer_generate_chat(ctx, msgs, num_msgs, callback, user_data)`**
  **Single-shot mode:** Ingests an entire array of chat messages, generates a response, and triggers your `callback` for every generated token. *Note: This function wipes the KV cache before running, giving the AI "amnesia" from previous calls.*

* **`infer_generate_chat_async(...)`**
  Spawns a background thread (`std::thread`) to run generation without freezing your application's main thread or UI render loop.

* **`infer_generate_chat_turn(ctx, role, content, callback, user_data)`**
  **Multi-turn REPL mode:** Instead of clearing memory, this function tokenizes only your *newest* message and appends it to the existing KV cache (tracked by `n_past`). This makes follow-up questions evaluate instantly without recalculating the whole conversation history.

* **`infer_generate_completion(ctx, raw_prompt, callback, user_data)`**
  Bypasses Jinja chat templates entirely. It feeds raw text directly into the model, useful for code autocompletion or legacy base models.

---

## 3. Memory & Session Management (KV Cache)

* **`infer_memory_clear(ctx)`**
  Forces the KV cache `n_past` tracker back to `0` and clears the hardware memory. Use this when the user clicks "New Chat".

* **`infer_state_save(ctx, filepath)`**
  Dumps the active KV cache memory to a file on your hard drive.

* **`infer_state_load(ctx, filepath)`**
  Reloads a previously saved KV cache file. This allows you to close your program, reopen it later, and instantly resume a conversation exactly where you left off.

---

## 4. Advanced Control (Samplers, Grammar & Stops)

* **`infer_set_sampler(ctx, params)`**
  Updates parameters like `temperature`, `top_p`, `min_p`, and `repeat_penalty`. A temperature of `0.0f` enables greedy decoding (deterministic), while higher values increase creativity.

* **`infer_set_grammar(ctx, grammar_gbnf)`**
  Forces the AI to generate text that strictly matches a GBNF grammar schema. This is critical for forcing the AI to output valid JSON or specific tool-calling formats without syntax errors.

* **`infer_set_stop_sequences(ctx, stop_words, num_words)`**
  Provides a list of strings (e.g., `["User:", "<|endoftext|>"]`). The generation loop will immediately halt if the AI generates any of these exact strings.

---

## 5. Embeddings (RAG)

* **`infer_get_embedding_dim(ctx)`**
  Returns the vector size for the loaded model (e.g., `4096`).

* **`infer_get_embeddings(ctx, text, out_vec, vec_len)`**
  Converts a string of text into a high-dimensional float array (`out_vec`) instead of generating text. Used for semantic search and Retrieval-Augmented Generation (RAG).

---

## 6. Thread Control & Diagnostics

* **`infer_abort(ctx)`**
  Flips an atomic boolean flag (`abort_flag`) that instantly halts the active generation loop.

* **`infer_is_generating(ctx)`**
  Returns `true` if the background async worker is currently busy.

* **`infer_get_model_info(ctx, out_info)`**
  Extracts metadata directly from the `.gguf` file, including layer count, attention heads, context limit (`n_ctx_train`), and architecture description.

* **`infer_get_last_error(ctx)`**
  Returns a thread-safe string explaining the most recent failure (e.g., *"Prompt tokens exceed context capacity"*).

 

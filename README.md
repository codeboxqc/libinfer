# libinfer

[![Build libinfer](https://github.com/codeboxqc/libinfer/actions/workflows/build.yml/badge.svg)](https://github.com/codeboxqc/libinfer/actions)
[![Latest Release](https://img.shields.io/github/v/release/codeboxqc/libinfer?include_prereleases&color=blue)](https://github.com/codeboxqc/libinfer/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## 🚀 Quick Start Guide

### Step 1: Get the Pre-Built Binaries
Download pre-compiled binaries for your operating system directly from **[GitHub Releases](https://github.com/codeboxqc/libinfer/releases)**:
* `libinfer-windows-x64.zip`
* `libinfer-linux-x64.zip`
* `libinfer-macos-arm64.zip`

---

### Step 2: Write Your Application (`main.c`)

Create a file named `main.c`:

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "infer.h"

// Automatically link infer.lib on Windows MSVC
#if defined(_WIN32) && defined(_MSC_VER)
#pragma comment(lib, "infer.lib")
#endif

// Streaming Callback: Triggered whenever a new text piece is generated
static bool on_token_stream(const char* piece, void* user_data) {
    (void)user_data;
    printf("%s", piece);
    fflush(stdout);
    return true; // Return true to continue, or false to interrupt generation
}

int main(int argc, char** argv) {
    const char* model_path = (argc >= 2) ? argv[1] : "qwen2.5-coder-7b-instruct-q4_k_m.gguf";
    const char* prompt     = (argc >= 3) ? argv[2] : "Write a fast C function for binary search.";

    printf("Initializing libinfer context...\n");
    infer_ctx* ctx = infer_init();
    if (!ctx) {
        fprintf(stderr, "Failed to create context.\n");
        return 1;
    }

    printf("Loading model: %s...\n", model_path);
    if (!infer_load_model(ctx, model_path, 2048)) {
        fprintf(stderr, "Error loading model: %s\n", infer_get_last_error(ctx));
        infer_free(ctx);
        return 1;
    }

    // Set sampler parameters
    infer_sampler_params sparams = infer_sampler_default_params();
    sparams.temperature = 0.20f; // Low temperature for code precision
    sparams.top_p       = 0.95f;
    sparams.top_k       = 40;
    infer_set_sampler(ctx, &sparams);

    // Prepare chat conversation
    infer_chat_msg messages[] = {
        { "system", "You are an expert C/C++ coding assistant." },
        { "user",   prompt }
    };
    size_t num_msgs = sizeof(messages) / sizeof(messages[0]);

    printf("\n--- Assistant Output ---\n");
    bool ok = infer_generate_chat(ctx, messages, num_msgs, on_token_stream, NULL);
    printf("\n------------------------\n");

    if (!ok) {
        fprintf(stderr, "Generation error: %s\n", infer_get_last_error(ctx));
    }

    infer_free(ctx);
    return 0;
}


Step 3: Compile Your Application
On Windows (MSVC)
code
Cmd
On Linux (GCC)
code
Bash
gcc main.c -Iinclude -Lstaging -linfer -Wl,-rpath,staging -o infer_cli
On macOS (Clang)
code
Bash
clang main.c -Iinclude -Lstaging -linfer -Wl,-rpath,staging -o infer_cli
Step 4: Run Inference
Download a GGUF model (e.g. qwen2.5-coder-7b-instruct-q4_k_m.gguf) and run:
code
Bash
./infer_cli qwen2.5-coder-7b-instruct-q4_k_m.gguf "Write a thread-safe stack in C."
🛠️ Building libinfer From Source
Method A: Building via CMake & Git Submodules
code
Bash
# 1. Clone the repository with submodules
git clone --recursive https://github.com/codeboxqc/libinfer.git
cd libinfer

# 2. Configure CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 3. Build libinfer
cmake --build build --config Release --parallel 4
Method B: Building via vcpkg (Manifest Mode)
If you use vcpkg for dependency management:
code
Bash
# 1. Set VCPKG_ROOT environment variable to your vcpkg installation
export VCPKG_ROOT=/path/to/vcpkg

# 2. Configure using vcpkg toolchain integration
cmake -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# 3. Build
cmake --build build --config Release
❓ Troubleshooting & FAQ
1. unresolved external symbol __imp_infer_* on Windows
This linker error happens when compiling with MSVC without passing the import library infer.lib.
Fix: Pass /link /LIBPATH:<dir> infer.lib to your compiler, or add #pragma comment(lib, "infer.lib") at the top of your main.c.
2. error while loading shared libraries: libinfer.so on Linux
Linux cannot locate libinfer.so at runtime.
Fix: Set LD_LIBRARY_PATH:
code
Bash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
or compile with rpath: -Wl,-rpath,.
3. Enabling CUDA / GPU Acceleration
Ensure CUDA Toolkit 12.x is installed on your system. CMake will automatically detect nvcc during the configure step and build ggml-cuda.dll / ggml-cuda.so. Place this backend module in the same directory as infer.dll / libinfer.so for automatic hardware acceleration discovery.
 

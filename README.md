# libinfer

`libinfer` is a lightweight C-API abstraction layer built on top of `llama.cpp` and `ggml`. It provides a clean, stable ABI boundary for native desktop developers (using SDL, OpenGL, DirectX, Qt, etc.) without requiring tensor math, manual context management, or static CUDA coupling.

## Features
- **Dynamic Backend Loading**: Leverages `GGML_BACKEND_DL` so CUDA, Vulkan, and CPU modules are loaded at runtime via `LoadLibrary`/`dlopen`.
- **Pure C-ABI**: Safe for FFI across C, C++, Rust, C#, Go, and Swift.
- **Auto-VRAM Context Calculation**: Analyzes Transformer KV cache dimensions (`n_layer`, `n_head_kv`, `n_embd`) against real-time free VRAM to prevent OOM errors.
- **Automated Jinja Chat Templates**: Uses embedded GGUF Jinja templates via `llama_chat_apply_template()`.

## Directory Structure
```
libinfer/
├── CMakeLists.txt
├── README.md
├── include/
│   └── infer.h
├── src/
│   └── infer_core.cpp
├── examples/
│   └── main.c
└── third_party/
    └── llama.cpp/ (submodule)
```

## Quick Start Build

```bash
# Clone with submodules
git clone --recursive <repo-url> libinfer
cd libinfer

# Or if cloned without submodules:
git submodule update --init --recursive third_party/llama.cpp

# Configure and compile
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

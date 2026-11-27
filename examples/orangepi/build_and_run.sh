#!/usr/bin/env bash
# Quick-start helper for Orange Pi 5 (RK3588) & Orange Pi 6 (CIX P1) - Linux ARM64
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build-arm64"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-model.gguf> [extra infer_orange_pi args...]"
    echo "Example (Orange Pi 5 CPU):    $0 ~/models/qwen2.5-3b-q4_k_m.gguf --cpu --kv-q8"
    echo "Example (Orange Pi 6 Vulkan): $0 ~/models/qwen2.5-7b-q4_k_m.gguf --vulkan --ctx 4096"
    exit 1
fi

MODEL_PATH="$1"
shift

# Option A: If prebuilt libinfer.so is already in the current directory (from libinfer-linux-arm64.zip),
# compile orange_pi_chat.c directly against it with zero CMake overhead.
if [ -f "./libinfer.so" ] && [ -f "./infer.h" ]; then
    echo "[libinfer] Linking against prebuilt ./libinfer.so..."
    gcc -O3 "$ROOT_DIR/examples/orange_pi_chat.c" \
        -I. -L. -linfer -Wl,-rpath,'$ORIGIN' \
        -o ./infer_orange_pi
    exec ./infer_orange_pi "$MODEL_PATH" "$@"
fi

# Option B: Build libinfer + infer_orange_pi natively from source
echo "[libinfer] Building natively for $(uname -m)..."
EXTRA_CMAKE=()
if command -v glslc >/dev/null 2>&1 && pkg-config --exists vulkan 2>/dev/null; then
    echo "[libinfer] Vulkan SDK detected -> enabling GGML_VULKAN=ON"
    EXTRA_CMAKE+=(-DGGML_VULKAN=ON)
else
    echo "[libinfer] Vulkan dev headers not found -> building CPU-only (install libvulkan-dev glslc spirv-headers for Vulkan)"
fi

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    ${EXTRA_CMAKE[@]+"${EXTRA_CMAKE[@]}"}

cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"

# Place Vulkan plugin in vulkan/ subdirectory next to the binary so libinfer auto-discovers it
if [ -f "$BUILD_DIR/third_party/llama.cpp/bin/libggml-vulkan.so" ]; then
    mkdir -p "$BUILD_DIR/vulkan"
    cp -a "$BUILD_DIR/third_party/llama.cpp/bin"/libggml-vulkan*.so* "$BUILD_DIR/vulkan/" 2>/dev/null || true
fi

exec "$BUILD_DIR/infer_orange_pi" "$MODEL_PATH" "$@"

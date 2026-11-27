# Running `libinfer` on Orange Pi 5 & Orange Pi 6 (Linux ARM64)

`libinfer` includes native **Linux ARM64 (`aarch64`)** support with big.LITTLE core detection, accurate Linux `MemAvailable` RAM budgeting, multi-variant ARM CPU dynamic dispatch (`NEON`, `dotprod`, `i8mm`, `SVE2`), and **Vulkan** GPU offloading.

## Hardware Profiles

| Board | SoC | CPU Topology | Auto Threads (`libinfer`) | GPU (Vulkan) | Recommended Models (Q4_K_M) |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Orange Pi 5 / 5B / 5 Plus / 5 Max / 5 Ultra** | Rockchip RK3588 / RK3588S | 4x Cortex-A76 + 4x Cortex-A55 | **4** (A76 big cores) | Arm Mali-G610 | 1.5B–3B (4GB/8GB), 7B–8B (16GB) |
| **Orange Pi 6 / 6 Plus** | CIX P1 (CD8180) | 8x Cortex-A720 + 4x Cortex-A520 | **8** (A720 big cores) | Arm Immortalis-G720 MC10 | 7B–14B (16GB/24GB+) |

> **Why big-core auto-detection matters:** On asymmetric big.LITTLE ARM64 SoCs (RK3588 and CIX P1), scheduling `ggml` threads across both big and LITTLE cores causes the slower A55/A520 cores to stall synchronization barriers. `libinfer` inspects `/sys/devices/system/cpu/cpu*/cpu_capacity` to automatically pick the performance cores (4 on Orange Pi 5, 8 on Orange Pi 6). You can always override this with `INFER_N_THREADS=<N>`.

---

## Option 1: Quick Start with Prebuilt Release Binaries

Download the latest `libinfer-linux-arm64.zip` from [GitHub Releases](https://github.com/codeboxqc/libinfer/releases):

```bash
mkdir -p ~/libinfer-arm64 && cd ~/libinfer-arm64
curl -LO https://github.com/codeboxqc/libinfer/releases/download/latest/libinfer-linux-arm64.zip
unzip libinfer-linux-arm64.zip
chmod +x infer_cli infer_orange_pi
```

Run directly on **Orange Pi 5** (CPU big cores + Q8_0 KV cache to conserve RAM):

```bash
./infer_orange_pi ~/models/qwen2.5-3b-instruct-q4_k_m.gguf --cpu --ctx 2048 --kv-q8
```

Run on **Orange Pi 6** (Vulkan GPU offload on Immortalis-G720, or `--cpu` for 8x Cortex-A720):

```bash
./infer_orange_pi ~/models/qwen2.5-7b-instruct-q4_k_m.gguf --vulkan --ctx 4096 --flash-attn
```

Compile your own C/C++ app against the extracted `libinfer.so`:

```bash
gcc -O3 my_app.c -I. -L. -linfer -Wl,-rpath,'$ORIGIN' -o my_app
./my_app
```

---

## Option 2: Building Natively on Orange Pi 5 / 6

### 1. Install Build & Vulkan Dependencies (Debian / Ubuntu / Armbian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake ninja-build git pkg-config \
    libvulkan-dev glslc spirv-headers vulkan-tools
```

Verify Vulkan driver visibility (Panvk/Mesa or vendor Mali driver):

```bash
vulkaninfo --summary
```

### 2. Clone & Build

```bash
git clone --recursive https://github.com/codeboxqc/libinfer.git
cd libinfer

cmake -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DGGML_VULKAN=ON

cmake --build build --config Release -j"$(nproc)"
```

Or use the included helper script:

```bash
chmod +x examples/orangepi/build_and_run.sh
./examples/orangepi/build_and_run.sh ~/models/qwen2.5-3b-instruct-q4_k_m.gguf --cpu --kv-q8
```

---

## Tips for Best Performance on Orange Pi 5 / 6

* **Thread override:** Test `INFER_N_THREADS=4` vs `INFER_N_THREADS=6` on Orange Pi 5, or `INFER_N_THREADS=8` on Orange Pi 6.
* **4GB / 8GB boards:** Pass `--kv-q8` or `--kv-q4` (automatically enables Flash Attention) to shrink KV cache RAM usage by 50–75%.
* **CPU Governor:** Set performance governor during inference benchmarks:
  ```bash
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```
* **Vulkan fallback:** If your Orange Pi OS image lacks a Vulkan 1.2+ driver (e.g., headless server image without Mali/Panvk ICD), pass `--cpu` or delete the `vulkan/` subdirectory next to `libinfer.so` to use pure ARM64 NEON/dotprod/i8mm CPU inference.

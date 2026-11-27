/* ========================================================================== */
/* examples/orange_pi_chat.c                                                  */
/* Interactive Chat & Benchmark Example for ARM64 Devices                     */
/* Targets: Orange Pi 5 (RK3588), Orange Pi 6 (CIX P1), Linux & Windows ARM64 */
/* ========================================================================== */
#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "infer.h"

typedef struct stream_metrics {
    size_t pieces;
    size_t bytes;
} stream_metrics;

static double monotonic_seconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static void set_env_var(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void detect_device_banner(char* out, size_t out_size) {
    if (!out || out_size == 0) return;

#if defined(__linux__)
    FILE* f = fopen("/proc/device-tree/model", "rb");
    if (f) {
        size_t n = fread(out, 1, out_size - 1, f);
        fclose(f);
        if (n > 0) {
            out[n] = '\0';
            while (n > 0 && (out[n - 1] == '\0' || out[n - 1] == '\n' || out[n - 1] == '\r')) {
                out[--n] = '\0';
            }
            if (n > 0) return;
        }
    }
    snprintf(out, out_size, "Linux (%s)",
#if defined(__aarch64__)
             "ARM64 / aarch64"
#elif defined(__x86_64__)
             "x86_64"
#else
             "unknown arch"
#endif
    );
#elif defined(_WIN32)
    snprintf(out, out_size, "Windows (%s)",
#if defined(_M_ARM64) || defined(__aarch64__)
             "ARM64"
#elif defined(_M_X64) || defined(__x86_64__)
             "x64"
#else
             "x86"
#endif
    );
#elif defined(__APPLE__)
    snprintf(out, out_size, "macOS (ARM64)");
#else
    snprintf(out, out_size, "Unknown Platform");
#endif
}

static bool on_token_v2(const char* piece, size_t len, void* user_data) {
    stream_metrics* m = (stream_metrics*)user_data;
    if (piece && len > 0) {
        fwrite(piece, 1, len, stdout);
        fflush(stdout);
        if (m) {
            m->pieces++;
            m->bytes += len;
        }
    }
    return true;
}

static void on_log(infer_log_level level, const char* text, void* user_data) {
    (void)user_data;
    if (level == INFER_LOG_ERROR || level == INFER_LOG_WARN) {
        fprintf(stderr, "[libinfer] %s", text ? text : "");
    }
}

static void print_usage(const char* prog) {
    printf(
        "libinfer ARM64 / Orange Pi 5 & 6 Chat Example\n\n"
        "Usage:\n"
        "  %s <model.gguf> [options]\n\n"
        "Options:\n"
        "  --cpu                   CPU-only mode (n_gpu_layers = 0; uses big cores on Orange Pi 5/6)\n"
        "  --vulkan                Offload all layers to Vulkan GPU (Mali-G610 / Immortalis-G720 / Adreno)\n"
        "  --gpu-layers <N>        Explicit layer offload count (-1 = auto, 0 = CPU)\n"
        "  --ctx <N>               Context window size (0 = auto-size to free RAM/VRAM, default: 2048)\n"
        "  --threads <N>           Override CPU thread count (Orange Pi 5: 4, Orange Pi 6: 8)\n"
        "  --kv-q8                 Use Q8_0 KV cache (saves ~50%% KV RAM on 4GB/8GB SBCs)\n"
        "  --kv-q4                 Use Q4_0 KV cache (saves ~75%% KV RAM; enables Flash Attention)\n"
        "  --flash-attn            Enable Flash Attention\n"
        "  --max-tokens <N>        Max generated tokens per turn (default: 512)\n"
        "  --temp <F>              Temperature (default: 0.6)\n"
        "  --prompt <text>         Run a single prompt and exit (useful for benchmarking)\n"
        "  --verbose               Print backend warnings and errors\n"
        "  -h, --help              Show this help\n\n"
        "Examples:\n"
        "  Orange Pi 5 (RK3588, 8GB, CPU big cores + Q8 KV cache):\n"
        "    %s qwen2.5-3b-instruct-q4_k_m.gguf --cpu --ctx 2048 --kv-q8\n\n"
        "  Orange Pi 6 (CIX P1, 16GB+, Vulkan Immortalis-G720 GPU):\n"
        "    %s qwen2.5-7b-instruct-q4_k_m.gguf --vulkan --ctx 4096 --flash-attn\n\n"
        "  Windows ARM64 (Snapdragon X / Adreno Vulkan):\n"
        "    %s llama-3.2-3b-instruct-q4_k_m.gguf --vulkan --ctx 4096\n",
        prog, prog, prog, prog
    );
}

int main(int argc, char** argv) {
    const char* model_path    = NULL;
    const char* single_prompt = NULL;
    int32_t     gpu_layers    = -1;   /* auto: uses Vulkan GPU if available, otherwise CPU */
    uint32_t    ctx_size      = 2048; /* conservative default for 4-16GB SBCs */
    int32_t     max_tokens    = 512;
    int32_t     threads       = 0;    /* 0 = auto big-core detection in libinfer */
    int32_t     kv_type       = INFER_KV_TYPE_F16;
    bool        flash_attn    = false;
    bool        verbose       = false;
    float       temperature   = 0.60f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cpu") == 0) {
            gpu_layers = 0;
        } else if (strcmp(argv[i], "--vulkan") == 0) {
            gpu_layers = -1;
        } else if (strcmp(argv[i], "--gpu-layers") == 0 && i + 1 < argc) {
            gpu_layers = (int32_t)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            ctx_size = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = (int32_t)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--kv-q8") == 0) {
            kv_type    = INFER_KV_TYPE_Q8_0;
            flash_attn = true;
        } else if (strcmp(argv[i], "--kv-q4") == 0) {
            kv_type    = INFER_KV_TYPE_Q4_0;
            flash_attn = true;
        } else if (strcmp(argv[i], "--flash-attn") == 0) {
            flash_attn = true;
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = (int32_t)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            single_prompt = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (!model_path && argv[i][0] != '-') {
            model_path = argv[i];
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    if (threads > 0) {
        char tbuf[32];
        snprintf(tbuf, sizeof(tbuf), "%d", threads);
        set_env_var("INFER_N_THREADS", tbuf);
    }

    if (verbose) {
        infer_set_log_callback(on_log, NULL);
    } else {
        infer_set_log_callback(NULL, NULL);
    }

    char device_name[160] = {0};
    detect_device_banner(device_name, sizeof(device_name));

    printf("=============================================================\n");
    printf(" libinfer v%s - ARM64 / Orange Pi 5 & 6 Example\n", INFER_VERSION_STR);
    printf(" Device:     %s\n", device_name);
    printf(" Model:      %s\n", model_path);
    printf(" GPU Layers: %d (%s)\n", gpu_layers,
           gpu_layers == 0 ? "CPU only" : (gpu_layers < 0 ? "Auto Vulkan/GPU" : "Hybrid"));
    printf(" Context:    %u (%s)\n", ctx_size, ctx_size == 0 ? "auto" : "requested");
    printf(" KV Cache:   %s | Flash Attn: %s\n",
           kv_type == INFER_KV_TYPE_Q4_0 ? "Q4_0" : (kv_type == INFER_KV_TYPE_Q8_0 ? "Q8_0" : "F16"),
           flash_attn ? "ON" : "OFF");
    printf("=============================================================\n\n");

    char init_err[256] = {0};
    infer_ctx* ctx = infer_init_ex(init_err, sizeof(init_err));
    if (!ctx) {
        fprintf(stderr, "Failed to initialize libinfer: %s\n", init_err[0] ? init_err : "unknown error");
        return 1;
    }

    infer_model_params mparams = infer_model_default_params();
    mparams.requested_ctx       = ctx_size;
    mparams.allow_ctx_reduction = true; /* Gracefully shrink ctx on 4GB/8GB SBCs */
    mparams.n_gpu_layers        = gpu_layers;
    mparams.use_mmap            = true;
    mparams.flash_attn          = flash_attn;
    mparams.kv_type_k           = kv_type;
    mparams.kv_type_v           = kv_type;

    double t_load_start = monotonic_seconds();
    printf("Loading model...\n");
    if (!infer_load_model_ex(ctx, model_path, &mparams)) {
        char err[512] = {0};
        infer_get_last_error(ctx, err, sizeof(err));
        fprintf(stderr, "Failed to load model: %s\n", err);
        infer_free(ctx);
        return 1;
    }
    double t_load_elapsed = monotonic_seconds() - t_load_start;

    infer_model_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    if (infer_get_model_info(ctx, &info)) {
        printf("Loaded '%s' in %.2fs (layers=%u, embd=%u, active_ctx=%u)\n\n",
               info.description, t_load_elapsed, info.n_layer, info.n_embd, info.n_ctx_active);
    }

    infer_sampler_params sparams = infer_sampler_default_params();
    sparams.temperature = temperature;
    sparams.top_p       = 0.90f;
    sparams.top_k       = 40;
    if (!infer_set_sampler(ctx, &sparams)) {
        fprintf(stderr, "Warning: failed to configure sampler.\n");
    }

    if (single_prompt) {
        infer_chat_msg msgs[] = {
            { "system", "You are a concise, helpful assistant running locally on an ARM64 device." },
            { "user",   single_prompt }
        };
        stream_metrics m = {0, 0};
        double t0 = monotonic_seconds();
        bool ok = infer_generate_chat_v2(ctx, msgs, 2, max_tokens, on_token_v2, &m);
        double dt = monotonic_seconds() - t0;
        printf("\n\n[%.2fs | %zu pieces | %.1f pieces/s | KV=%u]\n",
               dt, m.pieces, dt > 0.0 ? ((double)m.pieces / dt) : 0.0,
               infer_get_kv_cache_token_count(ctx));
        if (!ok) {
            char err[512] = {0};
            infer_get_last_error(ctx, err, sizeof(err));
            fprintf(stderr, "Generation error: %s\n", err);
        }
        infer_free(ctx);
        return ok ? 0 : 1;
    }

    printf("Interactive mode. Commands: /info, /save, /load, /clear, /quit\n");

    char input[2048];
    bool first_turn = true;
    while (true) {
        printf("\nYou> ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) {
            break;
        }
        input[strcspn(input, "\r\n")] = '\0';
        if (input[0] == '\0') continue;

        if (strcmp(input, "/quit") == 0 || strcmp(input, "/exit") == 0) {
            break;
        } else if (strcmp(input, "/clear") == 0) {
            if (infer_memory_clear(ctx)) {
                first_turn = true;
                printf("[Memory & KV cache cleared]\n");
            }
            continue;
        } else if (strcmp(input, "/info") == 0) {
            printf("[Device: %s | Model: %s | KV: %u / %u]\n",
                   device_name, info.description,
                   infer_get_kv_cache_token_count(ctx), info.n_ctx_active);
            continue;
        } else if (strcmp(input, "/save") == 0) {
            if (infer_state_save(ctx, "orangepi_state.bin")) {
                printf("[State saved to orangepi_state.bin]\n");
            } else {
                char err[256] = {0};
                infer_get_last_error(ctx, err, sizeof(err));
                printf("[Save failed: %s]\n", err);
            }
            continue;
        } else if (strcmp(input, "/load") == 0) {
            if (infer_state_load(ctx, "orangepi_state.bin")) {
                first_turn = false;
                printf("[State restored from orangepi_state.bin (KV=%u)]\n",
                       infer_get_kv_cache_token_count(ctx));
            } else {
                char err[256] = {0};
                infer_get_last_error(ctx, err, sizeof(err));
                printf("[Load failed: %s]\n", err);
            }
            continue;
        }

        printf("AI> ");
        fflush(stdout);

        stream_metrics m = {0, 0};
        double t0 = monotonic_seconds();
        bool ok = false;

        if (first_turn) {
            infer_chat_msg msgs[] = {
                { "system", "You are a concise, helpful assistant running locally on an ARM64 device." },
                { "user",   input }
            };
            ok = infer_generate_chat_v2(ctx, msgs, 2, max_tokens, on_token_v2, &m);
            if (ok) first_turn = false;
        } else {
            ok = infer_generate_chat_turn_v2(ctx, "user", input, max_tokens, on_token_v2, &m);
        }

        double dt = monotonic_seconds() - t0;
        printf("\n[%.2fs | %zu pieces | %.1f pieces/s | KV=%u / %u]\n",
               dt, m.pieces, dt > 0.0 ? ((double)m.pieces / dt) : 0.0,
               infer_get_kv_cache_token_count(ctx), info.n_ctx_active);

        if (!ok) {
            char err[512] = {0};
            infer_get_last_error(ctx, err, sizeof(err));
            fprintf(stderr, "[Error: %s]\n", err);
        }
    }

    infer_free(ctx);
    return 0;
}

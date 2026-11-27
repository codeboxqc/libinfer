/* ========================================================================== */
/* examples/main.c - Cross-Platform CLI Example (x64 & ARM64)                 */
/* ========================================================================== */
#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "infer.h"

static bool stream_cb(const char* piece, size_t len, void* user_data) {
    (void)user_data;
    if (piece && len > 0) {
        fwrite(piece, 1, len, stdout);
        fflush(stdout);
    }
    return true;
}

static void print_usage(const char* prog) {
    printf(
        "Usage: %s -m <model.gguf> [options]\n"
        "Options:\n"
        "  -m, --model <path>        Path to GGUF model file (required)\n"
        "  -p, --prompt <text>       Prompt text (default: \"Hello! Introduce yourself briefly.\")\n"
        "  -s, --system <text>       System prompt\n"
        "  -c, --ctx <tokens>        Context size (0 = auto, default: 0)\n"
        "  -n, --max-tokens <count>  Maximum tokens to generate (default: 256)\n"
        "  -ngl, --gpu-layers <n>    GPU layers (-1 = auto/all, 0 = CPU only, default: -1)\n"
        "  --temp <float>            Sampling temperature (default: 0.7)\n"
        "  -h, --help                Show this help\n",
        prog
    );
}

int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* prompt     = "Hello! Introduce yourself briefly.";
    const char* sys_prompt = "You are a helpful AI assistant.";
    uint32_t    ctx_size   = 0;
    int32_t     max_tokens = 256;
    int32_t     gpu_layers = -1;
    float       temp       = 0.7f;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--system") == 0) && i + 1 < argc) {
            sys_prompt = argv[++i];
        } else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--ctx") == 0) && i + 1 < argc) {
            ctx_size = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            max_tokens = (int32_t)strtol(argv[++i], NULL, 10);
        } else if ((strcmp(argv[i], "-ngl") == 0 || strcmp(argv[i], "--gpu-layers") == 0) && i + 1 < argc) {
            gpu_layers = (int32_t)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temp = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (!model_path && argv[i][0] != '-') {
            model_path = argv[i];
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    char init_err[256] = {0};
    infer_ctx* ctx = infer_init_ex(init_err, sizeof(init_err));
    if (!ctx) {
        fprintf(stderr, "infer_init_ex failed: %s\n", init_err[0] ? init_err : "unknown error");
        return 1;
    }

    infer_model_params mparams = infer_model_default_params();
    mparams.requested_ctx       = ctx_size;
    mparams.allow_ctx_reduction = (ctx_size == 0);
    mparams.n_gpu_layers        = gpu_layers;

    if (!infer_load_model_ex(ctx, model_path, &mparams)) {
        char err[512] = {0};
        infer_get_last_error(ctx, err, sizeof(err));
        fprintf(stderr, "Failed to load model '%s': %s\n", model_path, err);
        infer_free(ctx);
        return 1;
    }

    infer_sampler_params sparams = infer_sampler_default_params();
    sparams.temperature = temp;
    if (!infer_set_sampler(ctx, &sparams)) {
        fprintf(stderr, "Warning: failed to set sampler parameters.\n");
    }

    infer_chat_msg msgs[] = {
        { "system", sys_prompt },
        { "user",   prompt }
    };

    bool ok = infer_generate_chat_v2(ctx, msgs, 2, max_tokens, stream_cb, NULL);
    printf("\n");

    if (!ok) {
        char err[512] = {0};
        infer_get_last_error(ctx, err, sizeof(err));
        fprintf(stderr, "Generation error: %s\n", err);
    }

    infer_free(ctx);
    return ok ? 0 : 1;
}

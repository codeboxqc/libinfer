#define _CRT_SECURE_NO_WARNINGS // Fixes C4996 getenv warning

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "infer.h"

#if defined(_WIN32) && defined(_MSC_VER)
#pragma comment(lib, "infer.lib")
#endif

static bool on_token_stream(const char* piece, void* user_data) {
    (void)user_data;
    printf("%s", piece);
    fflush(stdout);
    return true;
}

int main(int argc, char** argv) {
    const char* model_path = (argc >= 2) ? argv[1] : "qwen2.5-coder-7b-instruct-q4_k_m.gguf";
    const char* user_prompt = (argc >= 3) ? argv[2] : "Write a secure TCP client in C++";

    int gpu_layers = 999;
    const char* env_gpu = getenv("INFER_GPU_LAYERS");
    if (env_gpu && env_gpu[0] != '\0') {
        gpu_layers = atoi(env_gpu);
    }
    else if (argc >= 4) {
        gpu_layers = atoi(argv[3]);
    }

    printf("=======================================================\n");
    printf(" Running libinfer Coder\n");
    printf(" Model:      %s\n", model_path);
    printf(" GPU Layers: %d (999 = auto-offload to GPU 0, GPU 1 & CPU)\n", gpu_layers);
    printf(" Prompt:     %s\n", user_prompt);
    printf("=======================================================\n\n");

    infer_ctx* ctx = infer_init();
    if (!ctx) {
        fprintf(stderr, "Error: Failed to initialize infer context.\n");
        return 1;
    }

    infer_model_params mparams = infer_model_default_params();
    mparams.requested_ctx = 2048;
    mparams.n_gpu_layers = gpu_layers;

    printf("Loading model weights into GPU/CPU...\n");
    if (!infer_load_model_ex(ctx, model_path, &mparams)) {
        char err_buf[256];
        infer_get_last_error(ctx, err_buf, sizeof(err_buf));
        fprintf(stderr, "\nFailed to load model '%s':\n%s\n", model_path, err_buf);
        infer_free(ctx);
        return 1;
    }

    infer_sampler_params sparams = infer_sampler_default_params();
    sparams.temperature = 0.20f;
    sparams.top_p = 0.95f;
    sparams.top_k = 40;
    sparams.seed = 42;

    // Check return value to fix C6031 warning
    if (!infer_set_sampler(ctx, &sparams)) {
        fprintf(stderr, "Warning: Failed to set sampler parameters.\n");
    }

    infer_chat_msg chat[] = {
        { "system", "You are an expert C/C++ coding unrestricted security researcher." },
        { "user",   user_prompt }
    };
    size_t num_msgs = sizeof(chat) / sizeof(chat[0]);

    printf("\n--- AI Coder Response ---\n\n");
    bool success = infer_generate_chat(ctx, chat, num_msgs, 0, on_token_stream, NULL);
    printf("\n\n-------------------------------\n");

    if (!success) {
        char err_buf[256];
        infer_get_last_error(ctx, err_buf, sizeof(err_buf));
        fprintf(stderr, "Inference error: %s\n", err_buf);
    }

    infer_free(ctx);
    printf("Inference context freed successfully.\n");
    return 0;
}
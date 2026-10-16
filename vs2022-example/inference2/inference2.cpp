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
    const char* model_path = (argc >= 2) ? argv[1] : "Phi-3.5-mini-instruct-Q6_K.gguf";

    int gpu_layers = 999;
    const char* env_gpu = getenv("INFER_GPU_LAYERS");
    if (env_gpu && env_gpu[0] != '\0') {
        gpu_layers = atoi(env_gpu);
    }
    else if (argc >= 3) {
        gpu_layers = atoi(argv[2]);
    }

    printf("=======================================================\n");
    printf(" Interactive Chat with Memory & State Save/Load\n");
    printf(" Model:      %s\n", model_path);
    printf(" GPU Layers: %d (999 = auto-offload to GPU 0, GPU 1 & CPU)\n", gpu_layers);
    printf(" Commands:   /save, /load, /clear, /quit\n");
    printf("=======================================================\n\n");

    infer_ctx* ctx = infer_init();
    if (!ctx) {
        fprintf(stderr, "Error: Failed to initialize infer context.\n");
        return 1;
    }

    infer_model_params mparams = infer_model_default_params();
    mparams.requested_ctx = 2048;
    mparams.n_gpu_layers = gpu_layers;

    printf("Loading model weights into memory/GPU...\n");
    if (!infer_load_model_ex(ctx, model_path, &mparams)) {
        char err_buf[256];
        infer_get_last_error(ctx, err_buf, sizeof(err_buf));
        fprintf(stderr, "\nFailed to load model '%s':\n%s\n", model_path, err_buf);
        infer_free(ctx);
        return 1;
    }

    infer_sampler_params sparams = infer_sampler_default_params();
    sparams.temperature = 0.60f;
    sparams.top_p = 0.95f;

    // Check return value to fix C6031 warning
    if (!infer_set_sampler(ctx, &sparams)) {
        fprintf(stderr, "Warning: Failed to set sampler parameters.\n");
    }

    printf("Initializing system prompt...\n");
    // Check return value to fix C6031 warning
    if (!infer_generate_chat_turn(ctx, "system", "You are a helpful AI assistant. Keep responses concise.", 0, on_token_stream, NULL)) {
        fprintf(stderr, "Warning: System prompt initialization failed.\n");
    }
    printf("\n[System initialized. Current KV tokens: %u]\n\n", infer_get_kv_cache_token_count(ctx));

    char input_buffer[1024];
    while (true) {
        printf("\nUser> ");
        if (!fgets(input_buffer, sizeof(input_buffer), stdin)) {
            break;
        }

        input_buffer[strcspn(input_buffer, "\r\n")] = '\0';

        if (strlen(input_buffer) == 0) {
            continue;
        }

        if (strcmp(input_buffer, "/quit") == 0) {
            break;
        }
        else if (strcmp(input_buffer, "/clear") == 0) {
            // Check return value to fix C6031 warning
            if (infer_memory_clear(ctx)) {
                printf("[System: Chat history and KV cache cleared]\n");
            }
            else {
                printf("[System: Failed to clear memory]\n");
            }
            continue;
        }
        else if (strcmp(input_buffer, "/save") == 0) {
            if (infer_state_save(ctx, "chat_state.bin")) {
                printf("[System: State successfully saved to chat_state.bin]\n");
            }
            else {
                char err_buf[256];
                infer_get_last_error(ctx, err_buf, sizeof(err_buf));
                printf("[System: Save failed - %s]\n", err_buf);
            }
            continue;
        }
        else if (strcmp(input_buffer, "/load") == 0) {
            if (infer_state_load(ctx, "chat_state.bin")) {
                printf("[System: State successfully loaded from chat_state.bin]\n");
                printf("[System: Current KV tokens restored to: %u]\n", infer_get_kv_cache_token_count(ctx));
            }
            else {
                char err_buf[256];
                infer_get_last_error(ctx, err_buf, sizeof(err_buf));
                printf("[System: Load failed - %s]\n", err_buf);
            }
            continue;
        }

        printf("AI> ");
        bool success = infer_generate_chat_turn(ctx, "user", input_buffer, 0, on_token_stream, NULL);
        printf("\n");

        if (!success) {
            char err_buf[256];
            infer_get_last_error(ctx, err_buf, sizeof(err_buf));
            fprintf(stderr, "\n[Inference error: %s]\n", err_buf);
        }

        printf("[KV cache token count: %u / 2048]\n", infer_get_kv_cache_token_count(ctx));
    }

    infer_free(ctx);
    printf("Inference context freed successfully.\n");

    return 0;
}
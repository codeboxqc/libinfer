#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>




//Phi-3.5 Mini (Q6_K_L)", "Phi-3.5-mini-instruct-Q6_K_L.gguf 
//https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q6_K_L.gguf?download=true 
//DeepSeek-R1 Qwen 7B (Q3_K_M)", "DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf 
//https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf?download=true 
//Qwen 2.5 Coder 7B (Q4_K_M)", "qwen2.5-coder-7b-instruct-q4_k_m.gguf 
//https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf?download=true 
//Llama 3.2 3B (Q4_K_M)", "Llama-3.2-3B-Instruct-Q4_K_M.gguf 
//https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true 
//

#include "infer.h"

// Automatically link infer.lib when compiling with MSVC on Windows
#if defined(_WIN32) && defined(_MSC_VER)
#pragma comment(lib, "infer.lib")
#endif

// Streaming Callback: Prints generated code/text tokens as they arrive
static bool on_token_stream(const char* piece, void* user_data) {
    (void)user_data;
    printf("%s", piece);
    fflush(stdout); // Real-time terminal output[cite: 3]
    return true;
}

int main(int argc, char** argv) {
    // Default model filename add any model .guff
    const char* model_path = (argc >= 2) ? argv[1] : "Phi-3.5-mini-instruct-Q6_K.gguf";

    printf("=======================================================\n");
    printf(" Interactive Chat with Memory & State Save/Load\n");
    printf(" Model:  %s\n", model_path);
    printf(" Commands: /save, /load, /clear, /quit\n");
    printf("=======================================================\n\n");

    // Initialize inference context[cite: 3]
    infer_ctx* ctx = infer_init();
    if (!ctx) {
        fprintf(stderr, "Error: Failed to initialize infer context.\n");
        return 1;
    }

    // Load model with 2048 token context window[cite: 3]
    printf("Loading model weights into memory/GPU...\n");
    if (!infer_load_model(ctx, model_path, 2048)) {
        fprintf(stderr, "\nFailed to load model '%s':\n%s\n", model_path, infer_get_last_error(ctx));
        infer_free(ctx);
        return 1;
    }

    // Set sampling parameters[cite: 3]
    infer_sampler_params sparams = infer_sampler_default_params();
    sparams.temperature = 0.60f;
    sparams.top_p = 0.95f;
    infer_set_sampler(ctx, &sparams);

    // Initial system prompt to establish behavior (processed without echoing)
    printf("Initializing system prompt...\n");
    infer_generate_chat_turn(ctx, "system", "You are a helpful AI assistant. Keep responses concise.", on_token_stream, NULL);
    printf("\n[System initialized. Current KV tokens: %u]\n\n", infer_get_kv_cache_token_count(ctx));

    // Simple interactive command loop
    char input_buffer[1024];
    while (true) {
        printf("\nUser> ");
        if (!fgets(input_buffer, sizeof(input_buffer), stdin)) {
            break;
        }

        // Strip trailing newline character
        input_buffer[strcspn(input_buffer, "\r\n")] = '\0';

        if (strlen(input_buffer) == 0) {
            continue;
        }

        // Handle special commands
        if (strcmp(input_buffer, "/quit") == 0) {
            break;
        }
        else if (strcmp(input_buffer, "/save") == 0) {
            if (infer_state_save(ctx, "chat_state.bin")) { // Saves KV Cache & Session State[cite: 1]
                printf("[System: State successfully saved to chat_state.bin]\n");
            }
            else {
                printf("[System: Save failed - %s]\n", infer_get_last_error(ctx));
            }
            continue;
        }
        else if (strcmp(input_buffer, "/load") == 0) {
            if (infer_state_load(ctx, "chat_state.bin")) { // Loads KV Cache & Session State[cite: 1]
                printf("[System: State successfully loaded from chat_state.bin]\n");
                printf("[System: Current KV tokens restored to: %u]\n", infer_get_kv_cache_token_count(ctx));
            }
            else {
                printf("[System: Load failed - %s]\n", infer_get_last_error(ctx));
            }
            continue;
        }
        else if (strcmp(input_buffer, "/clear") == 0) {
            infer_memory_clear(ctx); // Clears KV cache[cite: 1]
            printf("[System: Memory cleared. KV tokens: %u]\n", infer_get_kv_cache_token_count(ctx));
            continue;
        }

        // Standard user turn: uses infer_generate_chat_turn to persist memory automatically[cite: 1]
        printf("AI> ");
        bool success = infer_generate_chat_turn(ctx, "user", input_buffer, on_token_stream, NULL);
        printf("\n");

        if (!success) {
            fprintf(stderr, "\n[Inference error: %s]\n", infer_get_last_error(ctx));
        }

        // Show live token usage across the conversation[cite: 1]
        printf("[KV cache token count: %u / 2048]\n", infer_get_kv_cache_token_count(ctx));
    }

    // Cleanup resources[cite: 3]
    infer_free(ctx);
    printf("Inference context freed successfully.\n");

    return 0;
}
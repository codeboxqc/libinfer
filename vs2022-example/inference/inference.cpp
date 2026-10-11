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
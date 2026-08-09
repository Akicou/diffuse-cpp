#include "diffuse.h"
#include "diffuse-tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "diffuse-cpp: Inference engine for diffusion LLMs\n\n"
        "Usage: %s [options]\n\n"
        "Required:\n"
        "  -m PATH     Model file (GGUF)\n\n"
        "Input (one required):\n"
        "  -p TEXT     Prompt text (requires embedded tokenizer)\n"
        "  --tokens IDS   Comma-separated token IDs\n\n"
        "Generation:\n"
        "  -n INT      Tokens to generate (default: 256)\n"
        "  -s INT      Denoising steps per block (default: 32)\n"
        "  -t INT      Threads (default: 4)\n"
        "  -ngl INT    GPU layers to offload (default: 0)\n\n"
        "Sampling:\n"
        "  --temp F    Temperature (default: 0 = argmax)\n"
        "  --seed INT  Random seed (default: 42)\n"
        "  --remasking low_confidence|random (default: low_confidence)\n"
        "  --threshold F   Confidence threshold (default: 0.95)\n"
        "  --no-early-stop  Disable EOS early stop\n"
        "  --no-editing     Disable Levenshtein editing\n\n"
        "Other:\n"
        "  --system TEXT  System prompt (default: 'You are a helpful assistant.')\n"
        "  --raw       Don't apply chat template\n"
        "  -h, --help  Show this help\n"
    );
}

static std::vector<int32_t> parse_tokens(const char * str) {
    std::vector<int32_t> tokens;
    const char * p = str;
    while (*p) {
        tokens.push_back(atoi(p));
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return tokens;
}

int main(int argc, char ** argv) {
    std::string model_path, prompt;
    std::string system_prompt = "You are a helpful assistant.";
    std::vector<int32_t> input_tokens;
    int n_generate = 256, n_steps = 32, n_threads = 4, n_gpu_layers = 0;
    float temperature = 0.0f, threshold = 0.95f;
    uint32_t seed = 42;
    diffuse_remasking remasking = diffuse_remasking::LOW_CONFIDENCE;
    bool eos_early_stop = true, editing = true, raw = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i+1 < argc) model_path = argv[++i];
        else if (arg == "-p" && i+1 < argc) prompt = argv[++i];
        else if (arg == "-n" && i+1 < argc) n_generate = atoi(argv[++i]);
        else if (arg == "-s" && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (arg == "-t" && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (arg == "-ngl" && i+1 < argc) n_gpu_layers = atoi(argv[++i]);
        else if (arg == "--temp" && i+1 < argc) temperature = atof(argv[++i]);
        else if (arg == "--seed" && i+1 < argc) seed = atoi(argv[++i]);
        else if (arg == "--tokens" && i+1 < argc) input_tokens = parse_tokens(argv[++i]);
        else if (arg == "--remasking" && i+1 < argc) {
            std::string r = argv[++i];
            if (r == "random") remasking = diffuse_remasking::RANDOM;
        }
        else if (arg == "--threshold" && i+1 < argc) threshold = atof(argv[++i]);
        else if (arg == "--no-early-stop") eos_early_stop = false;
        else if (arg == "--no-editing") editing = false;
        else if (arg == "--system" && i+1 < argc) system_prompt = argv[++i];
        else if (arg == "--raw") raw = true;
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
    }

    if (model_path.empty()) { fprintf(stderr, "Error: -m required\n"); return 1; }
    if (input_tokens.empty() && prompt.empty()) { fprintf(stderr, "Error: -p or --tokens required\n"); return 1; }

    fprintf(stderr, "Loading model...\n");
    diffuse_model * model = diffuse_model_load(model_path, n_threads);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    diffuse_tokenizer * tokenizer = diffuse_tokenizer_load(model_path);

    if (input_tokens.empty()) {
        if (!diffuse_tokenizer_ready(tokenizer)) {
            fprintf(stderr, "Error: no embedded tokenizer. Use --tokens or re-convert.\n");
            diffuse_model_free(model);
            return 1;
        }
        if (raw) {
            input_tokens = diffuse_tokenize(tokenizer, prompt, true);
        } else {
            std::vector<diffuse_chat_message> messages;
            if (!system_prompt.empty()) messages.push_back({"system", system_prompt});
            messages.push_back({"user", prompt});
            input_tokens = diffuse_apply_chat_template(tokenizer, messages, true);
        }
        fprintf(stderr, "Prompt: %zu tokens\n", input_tokens.size());
    }

    const auto & hp = diffuse_model_hparams(model);
    int n_ctx = (int)input_tokens.size() + n_generate + 64;
    diffuse_context * ctx = diffuse_context_new_gpu(model, n_ctx, n_threads, n_gpu_layers);

    diffuse_sampler_params sp;
    sp.n_steps        = n_steps;
    sp.temperature    = temperature;
    sp.seed           = seed;
    sp.remasking      = remasking;
    sp.threshold      = threshold;
    sp.eos_early_stop = eos_early_stop;
    sp.enable_editing = editing;
    sp.stop_token_2   = diffuse_token_id(tokenizer, "<|role_end|>");  // LLaDA2 turn end

    fprintf(stderr, "Generating %d tokens...\n", n_generate);

    auto result = diffuse_generate(ctx, input_tokens, n_generate, sp,
        [](int blk, int total_blk, int step, int total_step, const std::vector<int32_t> &) {
            if (total_blk > 1)
                fprintf(stderr, "\r  block %d/%d, step %d/%d", blk+1, total_blk, step, total_step);
            else
                fprintf(stderr, "\r  step %d/%d", step, total_step);
        });
    fprintf(stderr, "\n\n");

    if (diffuse_tokenizer_ready(tokenizer)) {
        printf("%s\n", diffuse_detokenize(tokenizer, result, true).c_str());
    } else {
        for (size_t i = 0; i < result.size(); i++) {
            if (i > 0) printf(",");
            printf("%d", result[i]);
        }
        printf("\n");
    }

    diffuse_context_free(ctx);
    diffuse_model_free(model);
    if (tokenizer) diffuse_tokenizer_free(tokenizer);
    return 0;
}

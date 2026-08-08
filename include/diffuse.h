#pragma once

// diffuse-cpp public API

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

// Forward declarations
struct diffuse_model;
struct diffuse_context;

// ── Hyperparameters ────────────────────────────────────────────
struct diffuse_hparams {
    uint32_t n_vocab      = 0;
    uint32_t n_embd       = 0;   // hidden_size
    uint32_t n_head       = 0;
    uint32_t n_head_kv    = 0;   // for GQA; equals n_head for MHA
    uint32_t n_layer      = 0;
    uint32_t n_ff         = 0;   // intermediate_size
    uint32_t n_ctx_max    = 0;   // max sequence length
    float    rope_theta   = 500000.0f;
    float    rms_norm_eps = 1e-5f;
    uint32_t mask_token_id = 0;

    // Derived
    uint32_t n_embd_head() const { return n_head > 0 ? n_embd / n_head : 0; }
};

// ── Sampler parameters ─────────────────────────────────────────
enum class diffuse_schedule {
    COSINE,
    LINEAR,
};

enum class diffuse_remasking {
    LOW_CONFIDENCE,
    RANDOM,
    ENTROPY_EXIT,   // Unmask all low-entropy tokens early (semantic scheduling)
    MASKGIT_PLUS,   // Dream: unmask highest top-1 confidence (similar to LOW_CONFIDENCE)
    TOPK_MARGIN,    // Dream: unmask by margin between top-1 and top-2 logits
};

struct diffuse_sampler_params {
    int      n_steps     = 32;
    float    temperature = 0.0f;  // 0 = argmax
    diffuse_schedule   schedule   = diffuse_schedule::COSINE;
    diffuse_remasking  remasking  = diffuse_remasking::LOW_CONFIDENCE;
    uint32_t seed       = 42;
    float    entropy_threshold = 1.5f;  // For ENTROPY_EXIT: unmask tokens below this
    bool     use_cache   = true;        // Inter-step KV cache (--no-cache to disable)
    int      cache_refresh    = 0;      // Force full forward every N steps (0 = never)
    int      cache_keep_active = 0;     // Keep recently-changed positions active N extra steps
};

// ── Generation parameters ──────────────────────────────────────
struct diffuse_params {
    std::string model_path;
    std::string prompt;
    int         n_generate  = 128;  // tokens to generate
    int         n_threads   = 4;
    int         n_gpu_layers = 0;   // layers to offload to GPU (0 = CPU only)
    diffuse_sampler_params sampler;
};

// ── Token callback (called after each diffusion step) ──────────
using diffuse_step_callback = std::function<void(
    int step, int total_steps, const std::vector<int32_t>& tokens)>;

// ── Model API ──────────────────────────────────────────────────
diffuse_model * diffuse_model_load(const std::string & path, int n_threads);
void            diffuse_model_free(diffuse_model * model);
const diffuse_hparams & diffuse_model_hparams(const diffuse_model * model);

// ── Context (holds compute buffers) ────────────────────────────
diffuse_context * diffuse_context_new(const diffuse_model * model, int n_ctx, int n_threads);
// Create context with GPU offload: n_gpu_layers layers on GPU (0 = CPU only)
diffuse_context * diffuse_context_new_gpu(const diffuse_model * model, int n_ctx, int n_threads, int n_gpu_layers);
void              diffuse_context_free(diffuse_context * ctx);

// ── Forward pass ───────────────────────────────────────────────
// tokens: [n_tokens], logits_out: [n_tokens * n_vocab]
bool diffuse_forward(diffuse_context * ctx,
                     const int32_t * tokens, int n_tokens,
                     float * logits_out);

// ── Generation (full diffusion loop) ───────────────────────────
std::vector<int32_t> diffuse_generate(
    diffuse_context * ctx,
    const std::vector<int32_t> & prompt_tokens,
    int n_generate,
    const diffuse_sampler_params & params,
    diffuse_step_callback callback = nullptr);

// ═══════════════════════════════════════════════════════════════
// ── Tokenizer API ─────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

struct diffuse_tokenizer;

// Load tokenizer from a GGUF file (returns nullptr if no tokenizer found)
diffuse_tokenizer * diffuse_tokenizer_load(const std::string & gguf_path);

// Free a tokenizer
void diffuse_tokenizer_free(diffuse_tokenizer * tok);

// Encode text → token IDs
std::vector<int32_t> diffuse_tokenize(
    diffuse_tokenizer * tok,
    const std::string & text,
    bool add_special = true);

// Decode token IDs → text
std::string diffuse_detokenize(
    diffuse_tokenizer * tok,
    const std::vector<int32_t> & ids,
    bool skip_special = true);

// Check if tokenizer was successfully loaded
bool diffuse_tokenizer_ready(const diffuse_tokenizer * tok);

// Get tokenizer vocab size
size_t diffuse_tokenizer_size(const diffuse_tokenizer * tok);

// end of diffuse.h

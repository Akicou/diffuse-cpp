#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

struct diffuse_model;
struct diffuse_context;

// ── Hyperparameters ────────────────────────────────────────────
struct diffuse_hparams {
    uint32_t n_vocab       = 0;
    uint32_t n_embd        = 0;
    uint32_t n_head        = 0;
    uint32_t n_head_kv     = 0;
    uint32_t n_layer       = 0;
    uint32_t n_ff          = 0;
    uint32_t n_ctx_max     = 0;
    float    rope_theta    = 500000.0f;
    float    rms_norm_eps  = 1e-5f;
    uint32_t mask_token_id = 0;
    uint32_t eos_token_id  = 0;

    // Block diffusion (LLaDA2.X)
    uint32_t block_length  = 0;     // 0 = flat diffusion (LLaDA 1.0/Dream)

    uint32_t n_embd_head() const { return n_head > 0 ? n_embd / n_head : 0; }
};

// ── Schedule and remasking ─────────────────────────────────────
enum class diffuse_schedule { COSINE, LINEAR };

enum class diffuse_remasking {
    LOW_CONFIDENCE,   // Commit highest-probability tokens first
    RANDOM,           // Commit tokens in random order
};

// ── Sampler parameters ─────────────────────────────────────────
struct diffuse_sampler_params {
    int      n_steps           = 32;
    float    temperature       = 0.0f;       // 0 = argmax
    diffuse_schedule schedule  = diffuse_schedule::COSINE;
    diffuse_remasking remasking = diffuse_remasking::LOW_CONFIDENCE;
    uint32_t seed              = 42;
    float    threshold         = 0.95f;      // Confidence threshold for token commit
    bool     eos_early_stop    = true;       // Stop when EOS is committed and confirmed
    bool     enable_editing    = true;       // Levenshtein editing (DELETE/INSERT)
};

// ── Step callback ──────────────────────────────────────────────
using diffuse_step_callback = std::function<void(
    int block, int total_blocks,
    int step, int total_steps,
    const std::vector<int32_t>& tokens)>;

// ── Model API ──────────────────────────────────────────────────
diffuse_model * diffuse_model_load(const std::string & path, int n_threads);
void            diffuse_model_free(diffuse_model * model);
const diffuse_hparams & diffuse_model_hparams(const diffuse_model * model);

// ── Context ────────────────────────────────────────────────────
diffuse_context * diffuse_context_new(const diffuse_model * model, int n_ctx, int n_threads);
diffuse_context * diffuse_context_new_gpu(const diffuse_model * model, int n_ctx, int n_threads, int n_gpu_layers);
void              diffuse_context_free(diffuse_context * ctx);

// ── Forward pass ───────────────────────────────────────────────
bool diffuse_forward(diffuse_context * ctx,
                     const int32_t * tokens, int n_tokens,
                     float * logits_out);

// ── Generation ─────────────────────────────────────────────────
std::vector<int32_t> diffuse_generate(
    diffuse_context * ctx,
    const std::vector<int32_t> & prompt_tokens,
    int n_generate,
    const diffuse_sampler_params & params,
    diffuse_step_callback callback = nullptr);

// ── Tokenizer API ──────────────────────────────────────────────
struct diffuse_tokenizer;
diffuse_tokenizer * diffuse_tokenizer_load(const std::string & gguf_path);
void diffuse_tokenizer_free(diffuse_tokenizer * tok);
std::vector<int32_t> diffuse_tokenize(diffuse_tokenizer * tok, const std::string & text, bool add_special = true);
std::string diffuse_detokenize(diffuse_tokenizer * tok, const std::vector<int32_t> & ids, bool skip_special = true);
bool diffuse_tokenizer_ready(const diffuse_tokenizer * tok);
size_t diffuse_tokenizer_size(const diffuse_tokenizer * tok);

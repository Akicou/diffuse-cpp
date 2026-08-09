#pragma once

#include "diffuse.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Logging ────────────────────────────────────────────────────
#define DIFFUSE_LOG(fmt, ...) fprintf(stderr, "[diffuse] " fmt "\n", ##__VA_ARGS__)
#define DIFFUSE_DIE(fmt, ...) do { \
    fprintf(stderr, "[diffuse] FATAL: " fmt "\n", ##__VA_ARGS__); \
    exit(1); \
} while(0)

// ── Model architecture type ────────────────────────────────────
enum class diffuse_arch {
    LLaDA,          // LLaDA 1.0 (flat diffusion, dense, MHA)
    DREAM,          // Dream (flat diffusion, dense, GQA, Qwen2.5 backbone)
    LLaDA2_MoE,     // LLaDA2.X (block diffusion, MoE, GQA, QK-norm)
};

// ── Tensor name helpers ────────────────────────────────────────
static inline std::string fmt_layer(const char * pattern, int i) {
    char buf[128];
    snprintf(buf, sizeof(buf), pattern, i);
    return buf;
}

// ── Dense per-layer weight struct (LLaDA 1.0 / Dream) ──────────
struct diffuse_layer {
    struct ggml_tensor * attn_norm   = nullptr;
    struct ggml_tensor * wq          = nullptr;
    struct ggml_tensor * wk          = nullptr;
    struct ggml_tensor * wv          = nullptr;
    struct ggml_tensor * wo          = nullptr;
    struct ggml_tensor * bq          = nullptr;  // Q bias (Dream/Qwen2.5)
    struct ggml_tensor * bk          = nullptr;
    struct ggml_tensor * bv          = nullptr;
    struct ggml_tensor * ffn_norm    = nullptr;
    struct ggml_tensor * ffn_gate    = nullptr;
    struct ggml_tensor * ffn_up      = nullptr;
    struct ggml_tensor * ffn_down    = nullptr;
};

// ── LLaDA2 MoE layer ───────────────────────────────────────────
struct diffuse_moe_layer {
    struct ggml_tensor * attn_norm       = nullptr;
    struct ggml_tensor * post_attn_norm  = nullptr;
    struct ggml_tensor * qkv             = nullptr;
    struct ggml_tensor * q_norm          = nullptr;
    struct ggml_tensor * k_norm          = nullptr;
    struct ggml_tensor * wo              = nullptr;

    // Dense MLP (for first_k_dense_replace layers)
    struct ggml_tensor * ffn_gate        = nullptr;
    struct ggml_tensor * ffn_up          = nullptr;
    struct ggml_tensor * ffn_down        = nullptr;

    // MoE router + experts
    struct ggml_tensor * gate_weight     = nullptr;  // [n_embd, n_experts]
    struct ggml_tensor * gate_bias       = nullptr;  // [n_experts]
    struct ggml_tensor * expert_gate     = nullptr;
    struct ggml_tensor * expert_up       = nullptr;
    struct ggml_tensor * expert_down     = nullptr;

    // Shared expert
    struct ggml_tensor * shared_gate     = nullptr;
    struct ggml_tensor * shared_up       = nullptr;
    struct ggml_tensor * shared_down     = nullptr;

    bool is_moe = false;
};

// ── Full model ─────────────────────────────────────────────────
struct diffuse_model {
    diffuse_hparams hparams;
    diffuse_arch    arch = diffuse_arch::LLaDA2_MoE;
    std::string     model_type;

    // Embeddings
    struct ggml_tensor * tok_embd    = nullptr;
    struct ggml_tensor * output_norm = nullptr;
    struct ggml_tensor * output      = nullptr;

    // Layers (dense or MoE depending on arch)
    std::vector<diffuse_layer>     layers;      // for LLaDA/Dream
    std::vector<diffuse_moe_layer> moe_layers;  // for LLaDA2

    // MoE hyperparameters (LLaDA2)
    uint32_t n_experts          = 0;
    uint32_t n_experts_per_tok  = 0;
    uint32_t n_shared_experts   = 0;
    uint32_t moe_intermediate   = 0;
    uint32_t first_k_dense      = 0;
    uint32_t head_dim           = 0;
    uint32_t rotary_dim         = 0;
    bool     use_qk_norm        = false;
    float    routed_scaling     = 1.0f;
    bool     norm_topk_prob     = true;
    uint32_t n_group            = 8;
    uint32_t topk_group         = 4;
    uint32_t expert_capacity    = 48;   // experts allowed per 32-token routing block
    uint32_t moe_block_size     = 32;   // token block size used by the MoE router
    uint32_t delete_token_id    = 0;
    uint32_t split_token_id     = 0;

    // Backend
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buf     = nullptr;
    struct ggml_context    * ctx     = nullptr;
};

// ── Compute context ────────────────────────────────────────────
struct diffuse_context {
    const diffuse_model * model;
    int n_ctx;
    int n_threads;
    int n_gpu_layers = 0;

    ggml_backend_t        backend_cpu  = nullptr;
    ggml_backend_t        backend_gpu  = nullptr;
    ggml_backend_sched_t  sched        = nullptr;
    int                   n_backends   = 0;

    ggml_backend_buffer_t buf          = nullptr;
    struct ggml_context  * ctx         = nullptr;

    void  * compute_buf     = nullptr;
    size_t compute_buf_size = 0;

    bool sched_initialized = false;
};

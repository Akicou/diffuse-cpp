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

// MSVC does not define M_PI by default
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Logging ────────────────────────────────────────────────────
#define DIFFUSE_LOG(fmt, ...) fprintf(stderr, "[diffuse] " fmt "\n", ##__VA_ARGS__)
#define DIFFUSE_DIE(fmt, ...) do { \
    fprintf(stderr, "[diffuse] FATAL: " fmt "\n", ##__VA_ARGS__); \
    exit(1); \
} while(0)

// ── Tensor name helpers ────────────────────────────────────────
static inline std::string fmt_layer(const char * pattern, int i) {
    char buf[128];
    snprintf(buf, sizeof(buf), pattern, i);
    return buf;
}

// ── Per-layer weight struct ────────────────────────────────────
struct diffuse_layer {
    // Attention
    struct ggml_tensor * attn_norm;   // RMSNorm weight
    struct ggml_tensor * wq;          // Q projection
    struct ggml_tensor * wk;          // K projection
    struct ggml_tensor * wv;          // V projection
    struct ggml_tensor * wo;          // output projection

    // QKV biases (optional, nullptr for models without them e.g. LLaDA)
    struct ggml_tensor * bq = nullptr;  // Q bias (Dream/Qwen2.5)
    struct ggml_tensor * bk = nullptr;  // K bias
    struct ggml_tensor * bv = nullptr;  // V bias

    // FFN (SwiGLU)
    struct ggml_tensor * ffn_norm;    // RMSNorm weight
    struct ggml_tensor * ffn_gate;    // gate projection (w1)
    struct ggml_tensor * ffn_up;      // up projection (w3)
    struct ggml_tensor * ffn_down;    // down projection (w2)
};

// ── LLaDA2 MoE layer (diffusion MoE) ───────────────────────────
struct diffuse_moe_layer {
    // Attention (fused QKV + QK norm + partial rotary)
    struct ggml_tensor * attn_norm       = nullptr;  // input_layernorm
    struct ggml_tensor * post_attn_norm  = nullptr;  // post_attention_layernorm
    struct ggml_tensor * qkv             = nullptr;  // fused query_key_value.weight
    struct ggml_tensor * q_norm          = nullptr;  // query_layernorm (RMSNorm on head_dim)
    struct ggml_tensor * k_norm          = nullptr;  // key_layernorm
    struct ggml_tensor * wo              = nullptr;  // attention.dense.weight

    // Dense MLP (for first_k_dense_replace layers)
    struct ggml_tensor * ffn_gate        = nullptr;
    struct ggml_tensor * ffn_up          = nullptr;
    struct ggml_tensor * ffn_down        = nullptr;

    // MoE router + experts
    struct ggml_tensor * gate_weight     = nullptr;  // [n_embd, n_experts]
    struct ggml_tensor * gate_bias       = nullptr;  // [n_experts] (expert_bias)
    struct ggml_tensor * expert_gate     = nullptr;  // [n_embd, moe_ff, n_experts] (stacked)
    struct ggml_tensor * expert_up       = nullptr;
    struct ggml_tensor * expert_down     = nullptr;

    // Shared expert (always active)
    struct ggml_tensor * shared_gate     = nullptr;
    struct ggml_tensor * shared_up       = nullptr;
    struct ggml_tensor * shared_down     = nullptr;

    bool is_moe = false;  // true for MoE layers, false for dense
};

// ── Full model struct ──────────────────────────────────────────
struct diffuse_model {
    diffuse_hparams hparams;
    std::string model_type;  // "llada", "dream", etc. (from GGUF metadata)

    // Embeddings
    struct ggml_tensor * tok_embd;    // token embeddings
    struct ggml_tensor * output_norm; // final RMSNorm
    struct ggml_tensor * output;      // lm_head (may be tied to tok_embd)

    // Layers
    std::vector<diffuse_layer> layers;
    std::vector<diffuse_moe_layer> moe_layers;  // LLaDA2 MoE layers

    // MoE hyperparameters (LLaDA2)
    uint32_t n_experts          = 0;
    uint32_t n_experts_per_tok  = 0;
    uint32_t n_shared_experts   = 0;
    uint32_t moe_intermediate   = 0;
    uint32_t first_k_dense      = 0;
    uint32_t head_dim           = 0;  // head_dim (may differ from n_embd/n_head)
    uint32_t rotary_dim         = 0;  // partial rotary dimension
    bool     use_qk_norm        = false;
    float    routed_scaling     = 1.0f;
    bool     norm_topk_prob     = true;
    uint32_t moe_block_size     = 32;
    uint32_t expert_capacity    = 48;
    uint32_t delete_token_id    = 0;
    uint32_t split_token_id     = 0;

    // GGML backend
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buf     = nullptr;
    struct ggml_context    * ctx     = nullptr;     // weight context
};

// ── Compute context ────────────────────────────────────────────
//
// Optimized after studying llama.cpp's compute buffer reuse pattern:
// llama.cpp pre-allocates compute buffers once in llama_context and reuses
// them across forward passes via ggml_backend_sched. We achieve a similar
// effect by caching a raw memory arena and re-initializing a fresh GGML
// context on top of it for each forward pass — avoiding the malloc/free
// overhead that the original diffuse-cpp incurred on every diffusion step.

struct diffuse_context {
    const diffuse_model * model;
    int n_ctx;
    int n_threads;

    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    struct ggml_context  * ctx    = nullptr;     // compute context

    // Persistent compute buffer (avoids alloc/free every forward pass)
    void  * compute_buf     = nullptr;
    size_t compute_buf_size = 0;
};

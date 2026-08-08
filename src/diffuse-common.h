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

#pragma once

#include "diffuse-common.h"
#include "diffuse-cache.h"

// ── Compute buffer helpers (used by diffuse-graph.cpp and diffuse-moe-graph.cpp) ──
size_t diffuse_compute_buf_size(const diffuse_hparams & hp, int n_tokens);
struct ggml_context * diffuse_new_compute_ctx(diffuse_context * dctx, size_t needed);
void diffuse_ensure_compute_buf(diffuse_context * dctx, size_t needed);

// ── LLaDA2 MoE forward pass ─────────────────────────────────────
bool diffuse_forward_moe(diffuse_context * ctx,
                         const int32_t * tokens, int n_tokens,
                         float * logits_out);

// Full MoE forward with KV cache extraction
bool diffuse_forward_moe_full(diffuse_context * ctx,
                              const int32_t * tokens, int n_tokens,
                              float * logits_out,
                              diffuse_step_cache * cache);
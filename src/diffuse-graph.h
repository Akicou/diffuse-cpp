#pragma once

#include "diffuse-common.h"

// ── Compute buffer helpers ─────────────────────────────────────
size_t diffuse_compute_buf_size(const diffuse_hparams & hp, int n_tokens);
struct ggml_context * diffuse_new_compute_ctx(diffuse_context * dctx, size_t needed);
void diffuse_ensure_compute_buf(diffuse_context * dctx, size_t needed);

// ── Dense forward (LLaDA 1.0 / Dream) ──────────────────────────
bool diffuse_forward_dense(diffuse_context * ctx,
                           const int32_t * tokens, int n_tokens,
                           float * logits_out);

// ── MoE forward (LLaDA2.X) ─────────────────────────────────────
// Build MoE graph with block-causal attention mask
// block_start and block_end define the current generation window within n_tokens
bool diffuse_forward_moe(diffuse_context * ctx,
                         const int32_t * tokens, int n_tokens,
                         float * logits_out);

// MoE forward with explicit attention mask support
// attn_mask: [n_tokens, n_tokens] additive mask (0 = attend, -inf = block)
// If attn_mask is nullptr, uses block-causal pattern from hp.block_length
bool diffuse_forward_moe_masked(diffuse_context * ctx,
                                 const int32_t * tokens, int n_tokens,
                                 const float * attn_mask,  // [n_tokens * n_tokens] or nullptr
                                 float * logits_out);

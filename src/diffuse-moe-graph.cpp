#include "diffuse-graph.h"
#include "diffuse-backend.h"

#include <algorithm>
#include <cmath>

// ── Helper ──
static struct ggml_tensor * ensure_f32_moe(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (!t) return nullptr;
    if (t->type != GGML_TYPE_F32) return ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// ── Build block-causal attention mask ──────────────────────────
// Returns a tensor [n_kv, n_batch, 1, 1] suitable for ggml_flash_attn_ext
// mask[i][j] = 0.0 if block_id(i) >= block_id(j), else -INFINITY
//
// When block_length == 0, creates a full bidirectional mask (all zeros).
static struct ggml_tensor * build_block_causal_mask(
        struct ggml_context * ctx,
        int n_tokens,
        int block_length,
        int n_prompt) {

    // Block ids: the whole prompt is block 0 (fully bidirectional within itself),
    // then generation blocks tile the span after it. Tiling from absolute 0
    // instead would split the prompt across blocks and stop its own tokens from
    // attending to each other.
    auto block_id = [&](int i) {
        if (i < n_prompt) return 0;
        return 1 + (i - n_prompt) / block_length;
    };

    // Mask shape for flash_attn_ext: [n_kv, n_batch, ne32, ne33] = [kv_len, q_len, 1, 1].
    // flash_attn_ext reads the mask as F16 (ggml-cpu/ops.cpp), so it MUST be F16 — an F32
    // mask decodes to NaN halves for the -INFINITY entries and breaks attention.
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_tokens, n_tokens);
    ggml_set_name(mask, "block_causal_mask");
    ggml_set_input(mask);

    ggml_fp16_t * mask_data = (ggml_fp16_t *)mask->data;
    const ggml_fp16_t zero    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);

    if (block_length <= 0) {
        // Full bidirectional: all zeros
        for (int i = 0; i < n_tokens * n_tokens; i++) {
            mask_data[i] = zero;
        }
    } else {
        // Block-causal: query i can attend to key j iff block_id(i) >= block_id(j).
        // Row-major data[q*ne0 + kv], ne0 = kv_len, so data[i*n_tokens + j] = mask[query i][key j].
        for (int i = 0; i < n_tokens; i++) {
            int block_i = block_id(i);
            for (int j = 0; j < n_tokens; j++) {
                int block_j = block_id(j);
                mask_data[i * n_tokens + j] = (block_i >= block_j) ? zero : neg_inf;
            }
        }
    }

    return mask;
}

// ── Build MoE transformer graph with block-causal attention ────
//
// Architecture (LLaDA2.X):
// 1. Token embedding
// 2. Per layer:
//    a. Input layernorm (RMSNorm)
//    b. Fused QKV projection → split into Q, K, V
//    c. QK normalization (RMSNorm on head_dim)
//    d. Partial RoPE (rotary_dim out of head_dim)
//    e. GQA expansion (n_kv_heads → n_heads)
//    f. Flash attention with block-causal mask
//    g. Output projection
//    h. Post-attention layernorm + residual
//    i. Dense MLP (first_k_dense layers) or MoE (rest)
// 3. Final RMSNorm + output projection (logits)
static struct ggml_cgraph * diffuse_build_graph_moe(
        diffuse_context * dctx,
        struct ggml_context * ctx,
        const int32_t * tokens,
        int n_tokens) {

    const auto & model = *dctx->model;
    const auto & hp    = model.hparams;
    const int n_embd      = (int)hp.n_embd;
    const int n_head      = (int)hp.n_head;
    const int n_head_kv   = (int)hp.n_head_kv;
    const int head_dim    = (int)model.head_dim;
    const int rotary_dim  = (int)model.rotary_dim;
    const int n_layer     = (int)hp.n_layer;
    const int N           = n_tokens;
    const int block_len   = (int)hp.block_length;
    const float rms_eps   = hp.rms_norm_eps;
    const float rope_base = hp.rope_theta;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, (size_t)(8192 * n_layer), false);

    // ── Inputs ─────────────────────────────────────────────────
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    memcpy(inp_tokens->data, tokens, N * sizeof(int32_t));

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    {
        int32_t * pos_data = (int32_t *)inp_pos->data;
        for (int i = 0; i < N; i++) pos_data[i] = i;
    }

    // Block-causal attention mask (clamped: n_prompt may exceed this window on prefill)
    struct ggml_tensor * attn_mask = build_block_causal_mask(ctx, N, block_len,
                                                             std::min(dctx->n_prompt, N));

    // ── Embedding ──────────────────────────────────────────────
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_tokens);

    // ── Transformer layers ─────────────────────────────────────
    for (int il = 0; il < n_layer; il++) {
        const auto & ml = model.moe_layers[il];
        struct ggml_tensor * residual = cur;

        // ── Attention ──────────────────────────────────────────
        // Input layernorm
        cur = ggml_rms_norm(ctx, cur, rms_eps);
        cur = ggml_mul(ctx, cur, ensure_f32_moe(ctx, ml.attn_norm));

        // Fused QKV: [hidden, (n_head + 2*n_kv_head) * head_dim]
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.qkv), cur);

        // Split: Q = [head_dim, n_head, N], K = [head_dim, n_kv_head, N], V = [head_dim, n_kv_head, N]
        // The fused weight stores [Q_heads; K_heads; V_heads] concatenated along dim 1
        // After mul_mat: qkv shape = [(n_head + 2*n_kv_head)*head_dim, N]
        const int q_dim = n_head * head_dim;
        const int kv_dim = n_head_kv * head_dim;

        struct ggml_tensor * Q = ggml_view_2d(ctx, qkv, head_dim * n_head,  N, head_dim * (n_head + 2*n_head_kv) * sizeof(float), 0);
        struct ggml_tensor * K = ggml_view_2d(ctx, qkv, head_dim * n_head_kv, N, head_dim * (n_head + 2*n_head_kv) * sizeof(float), q_dim * sizeof(float));
        struct ggml_tensor * V = ggml_view_2d(ctx, qkv, head_dim * n_head_kv, N, head_dim * (n_head + 2*n_head_kv) * sizeof(float), (q_dim + kv_dim) * sizeof(float));

        // Reshape for multi-head (need cont since view_2d is non-contiguous)
        Q = ggml_reshape_3d(ctx, ggml_cont(ctx, Q), head_dim, n_head, N);
        K = ggml_reshape_3d(ctx, ggml_cont(ctx, K), head_dim, n_head_kv, N);
        V = ggml_reshape_3d(ctx, ggml_cont(ctx, V), head_dim, n_head_kv, N);

        // QK normalization (RMSNorm on head_dim)
        if (model.use_qk_norm && ml.q_norm) {
            Q = ggml_rms_norm(ctx, Q, rms_eps);
            Q = ggml_mul(ctx, Q, ensure_f32_moe(ctx, ml.q_norm));
            K = ggml_rms_norm(ctx, K, rms_eps);
            K = ggml_mul(ctx, K, ensure_f32_moe(ctx, ml.k_norm));
        }

        // Partial RoPE: rotate first rotary_dim, pass through rest
        // ggml_rope_ext(ctx, a, pos, freq_factors, n_dims, mode, n_ctx_orig, n_freq_orig, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
        Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, rotary_dim, 0, 0, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, inp_pos, nullptr, rotary_dim, 0, 0, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute to flash-attn layout [head_dim, n_tokens, n_head].
        // flash_attn_ext broadcasts KV heads for GQA (query head h → kv head h/(n_head/n_head_kv)),
        // so no manual repeat is needed — and a manual ggml_repeat tiles the heads in the wrong order.
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);

        // Flash attention with block-causal mask
        struct ggml_tensor * attn = ggml_flash_attn_ext(
            ctx, Q, K, V, attn_mask,
            1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);

        attn = ggml_reshape_2d(ctx, attn, n_embd, N);

        // Output projection
        attn = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.wo), attn);

        // Residual + post-attention layernorm
        cur = ggml_add(ctx, attn, residual);
        residual = cur;
        cur = ggml_rms_norm(ctx, cur, rms_eps);
        cur = ggml_mul(ctx, cur, ensure_f32_moe(ctx, ml.post_attn_norm));

        // ── MLP / MoE ──────────────────────────────────────────
        if (!ml.is_moe) {
            // Dense MLP (SwiGLU)
            struct ggml_tensor * g = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.ffn_gate), cur);
            g = ggml_silu(ctx, g);
            struct ggml_tensor * u = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.ffn_up), cur);
            cur = ggml_mul(ctx, g, u);
            cur = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.ffn_down), cur);
        } else {
            // ── MoE layer (LLaDA2 / Ling — DeepSeek-V3 style group-limited routing) ──
            // Mirrors llama.cpp build_moe_ffn: sigmoid gating, bias-corrected group
            // selection (choice only), unbiased weights, normalize + scale, weighted
            // sum of the top-k experts, plus a shared expert on the same input.
            const int n_experts = (int)model.n_experts;
            const int top_k     = (int)model.n_experts_per_tok;
            const int n_group   = (int)model.n_group;
            const int topk_grp  = (int)model.topk_group;

            struct ggml_tensor * moe_in = cur;  // [n_embd, N] — input to routed AND shared experts

            // Router logits → sigmoid gating probs (unbiased)
            struct ggml_tensor * logits = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.gate_weight), moe_in);
            struct ggml_tensor * probs  = ggml_sigmoid(ctx, logits);  // [n_experts, N]

            // Selection scores: add expert bias (affects choice only, not the applied weights)
            struct ggml_tensor * sel = probs;
            if (ml.gate_bias) {
                sel = ggml_add(ctx, probs, ensure_f32_moe(ctx, ml.gate_bias));
            }

            // Group-limited routing: keep only experts in the top `topk_grp` groups,
            // where a group's score is the sum of its top-2 experts' selection scores.
            const bool use_groups = (n_group > 1) && (topk_grp > 0) && (topk_grp < n_group)
                                    && (n_experts % n_group == 0) && (n_experts / n_group >= 2);
            if (use_groups) {
                const int eg = n_experts / n_group;                                        // experts per group
                struct ggml_tensor * selg = ggml_reshape_3d(ctx, sel, eg, n_group, N);      // [eg, n_group, N]
                struct ggml_tensor * g2   = ggml_top_k(ctx, selg, 2);                       // [2, n_group, N] idx
                struct ggml_tensor * g2v  = ggml_get_rows(ctx,
                        ggml_reshape_4d(ctx, selg, 1, eg, n_group, N), g2);                 // [1, 2, n_group, N] vals
                struct ggml_tensor * gscore = ggml_sum_rows(ctx,
                        ggml_reshape_3d(ctx, g2v, 2, n_group, N));                          // [1, n_group, N]
                gscore = ggml_reshape_2d(ctx, gscore, n_group, N);                          // [n_group, N]
                struct ggml_tensor * gsel   = ggml_top_k(ctx, gscore, topk_grp);           // [topk_grp, N] group idx
                struct ggml_tensor * selg_k = ggml_get_rows(ctx, selg, gsel);              // [eg, topk_grp, N]
                sel = ggml_set_rows(ctx,
                        ggml_fill(ctx, selg, -INFINITY), selg_k, gsel);                    // mask non-selected groups
                sel = ggml_reshape_2d(ctx, sel, n_experts, N);                             // [n_experts, N]
            }

            // Final expert selection + gating weights (weights gathered from UNBIASED probs)
            struct ggml_tensor * idx = ggml_top_k(ctx, sel, top_k);                        // [top_k, N] i32
            struct ggml_tensor * weights = ggml_get_rows(ctx,
                    ggml_reshape_3d(ctx, probs, 1, n_experts, N), idx);                    // [1, top_k, N]
            if (model.norm_topk_prob) {
                struct ggml_tensor * w2 = ggml_reshape_2d(ctx, weights, top_k, N);
                w2 = ggml_div(ctx, w2, ggml_sum_rows(ctx, w2));                            // renormalize over top_k
                weights = ggml_reshape_3d(ctx, w2, 1, top_k, N);
            }
            if (model.routed_scaling != 1.0f) {
                weights = ggml_scale(ctx, weights, model.routed_scaling);
            }

            // Expert FFN (SwiGLU) via mul_mat_id — expert weights stay quantized
            struct ggml_tensor * cur3   = ggml_reshape_3d(ctx, moe_in, n_embd, 1, N);      // [n_embd, 1, N]
            struct ggml_tensor * eg_out = ggml_silu(ctx, ggml_mul_mat_id(ctx, ml.expert_gate, cur3, idx)); // [moe_ff, top_k, N]
            struct ggml_tensor * eu_out = ggml_mul_mat_id(ctx, ml.expert_up, cur3, idx);
            struct ggml_tensor * inter  = ggml_mul(ctx, eg_out, eu_out);                   // [moe_ff, top_k, N]
            struct ggml_tensor * experts = ggml_mul_mat_id(ctx, ml.expert_down, inter, idx); // [n_embd, top_k, N]
            experts = ggml_mul(ctx, experts, weights);                                     // apply gating

            // Aggregate the top_k expert outputs → [n_embd, N]
            struct ggml_tensor * moe_out = ggml_view_2d(ctx, experts, n_embd, N, experts->nb[2], 0);
            for (int e = 1; e < top_k; e++) {
                moe_out = ggml_add(ctx, moe_out,
                        ggml_view_2d(ctx, experts, n_embd, N, experts->nb[2], (size_t)e * experts->nb[1]));
            }
            if (top_k == 1) moe_out = ggml_cont(ctx, moe_out);

            // Shared expert runs on the MoE input (not the routed output), then is added
            if (model.n_shared_experts > 0 && ml.shared_gate) {
                struct ggml_tensor * sg = ggml_silu(ctx, ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.shared_gate), moe_in));
                struct ggml_tensor * su = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.shared_up), moe_in);
                struct ggml_tensor * sd = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.shared_down), ggml_mul(ctx, sg, su));
                moe_out = ggml_add(ctx, moe_out, sd);
            }

            cur = moe_out;
        }

        // Residual connection
        cur = ggml_add(ctx, cur, residual);
    }

    // ── Output norm + logits ───────────────────────────────────
    cur = ggml_rms_norm(ctx, cur, rms_eps);
    cur = ggml_mul(ctx, cur, ensure_f32_moe(ctx, model.output_norm));

    struct ggml_tensor * logits = ggml_mul_mat(ctx, ensure_f32_moe(ctx, model.output), cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);
    return gf;
}

// ── MoE forward pass ───────────────────────────────────────────
bool diffuse_forward_moe(diffuse_context * ctx,
                         const int32_t * tokens, int n_tokens,
                         float * logits_out) {
    return diffuse_forward_moe_masked(ctx, tokens, n_tokens, nullptr, logits_out);
}

// ── MoE forward with explicit mask ─────────────────────────────
bool diffuse_forward_moe_masked(diffuse_context * ctx,
                                 const int32_t * tokens, int n_tokens,
                                 const float * attn_mask_data,
                                 float * logits_out) {
    const auto & hp = ctx->model->hparams;
    size_t needed = diffuse_compute_buf_size(hp, n_tokens);
    struct ggml_context * ctx_compute = diffuse_new_compute_ctx(ctx, needed);

    struct ggml_cgraph * gf = diffuse_build_graph_moe(ctx, ctx_compute, tokens, n_tokens);

    if (!diffuse_sched_compute(ctx, ctx_compute, gf)) {
        DIFFUSE_LOG("MoE graph compute failed");
        ggml_free(ctx_compute);
        return false;
    }

    struct ggml_tensor * logits = ggml_graph_node(gf, -1);
    memcpy(logits_out, logits->data, (size_t)n_tokens * hp.n_vocab * sizeof(float));

    ggml_free(ctx_compute);
    return true;
}

#include "diffuse-graph.h"
#include "diffuse-cache.h"

#include <algorithm>
#include <cmath>

// ── Helper: ensure tensor is F32 ──
static struct ggml_tensor * ensure_f32_moe(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (!t) return nullptr;
    if (t->type != GGML_TYPE_F32) return ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// ── Build LLaDA2 MoE transformer forward graph ──────────────────
//
// Key differences from LLaDA-8B / Dream:
//   1. Fused QKV projection (single weight, split via views)
//   2. QK normalization (RMSNorm on head_dim after projection, before RoPE)
//   3. Partial rotary (only first rotary_dim of head_dim get RoPE)
//   4. GQA (n_head=32, n_head_kv=4)
//   5. MoE FFN (256 experts, top-8) + shared expert for layers >= first_k_dense
//   6. Dense FFN for first_k_dense layers
//   7. Bidirectional attention (non-causal, no mask)
//   8. Sigmoid router scoring

struct ggml_cgraph * diffuse_build_graph_moe(
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
    const int moe_ff      = (int)model.moe_intermediate;
    const int n_experts   = (int)model.n_experts;
    const int n_experts_used = (int)model.n_experts_per_tok;
    const float routed_scale = model.routed_scaling;

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

    // ── Embedding lookup ───────────────────────────────────────
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_tokens);

    // ── Transformer layers ─────────────────────────────────────
    for (int il = 0; il < n_layer; il++) {
        const auto & ml = model.moe_layers[il];
        struct ggml_tensor * residual = cur;

        // Pre-attention RMSNorm
        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = ml.attn_norm;
            if (norm_w->type != GGML_TYPE_F32) norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            cur = ggml_mul(ctx, cur, norm_w);
        }

        // ── Fused QKV projection ───────────────────────────────
        // qkv weight: GGML ne[0]=n_embd, ne[1]=(n_head+2*n_head_kv)*head_dim
        // Result: [(n_head+2*n_head_kv)*head_dim, N]
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, ml.qkv, cur);

        // Split Q, K, V using view_3d
        // qkv output layout: [head_dim, n_head+2*n_head_kv, N]
        // where dim 1 is: [n_head Q heads, n_head_kv K heads, n_head_kv V heads]
        const int q_dim = n_head * head_dim;
        const int kv_dim = n_head_kv * head_dim;
        const int qkv_dim = q_dim + 2 * kv_dim;
        const size_t es = ggml_element_size(qkv);

        // Q: [head_dim, n_head, N] — view into first n_head columns of dim 1
        struct ggml_tensor * Q = ggml_view_3d(ctx, qkv,
            head_dim, n_head, N,
            qkv_dim * es,       // nb[1] = stride per row (full qkv_dim)
            qkv_dim * es * N,   // nb[2] = stride per N
            0);                  // offset = 0
        // K: [head_dim, n_head_kv, N] — view starting at q_dim
        struct ggml_tensor * K = ggml_view_3d(ctx, qkv,
            head_dim, n_head_kv, N,
            qkv_dim * es,
            qkv_dim * es * N,
            q_dim * es);
        // V: [head_dim, n_head_kv, N] — view starting at q_dim + kv_dim
        struct ggml_tensor * V = ggml_view_3d(ctx, qkv,
            head_dim, n_head_kv, N,
            qkv_dim * es,
            qkv_dim * es * N,
            (q_dim + kv_dim) * es);

        // Make contiguous copies for subsequent ops
        Q = ggml_cont(ctx, Q);
        K = ggml_cont(ctx, K);
        V = ggml_cont(ctx, V);

        // ── QK normalization (RMSNorm on head_dim) ───────────
        if (model.use_qk_norm) {
            // Q: [head_dim, n_head, N] → norm on dim 0 (head_dim)
            Q = ggml_rms_norm(ctx, Q, hp.rms_norm_eps);
            {
                struct ggml_tensor * qw = ml.q_norm;
                if (qw->type != GGML_TYPE_F32) qw = ggml_cast(ctx, qw, GGML_TYPE_F32);
                Q = ggml_mul(ctx, Q, qw);
            }
            K = ggml_rms_norm(ctx, K, hp.rms_norm_eps);
            {
                struct ggml_tensor * kw = ml.k_norm;
                if (kw->type != GGML_TYPE_F32) kw = ggml_cast(ctx, kw, GGML_TYPE_F32);
                K = ggml_mul(ctx, K, kw);
            }
        }

        // ── Partial RoPE (only first rotary_dim of head_dim) ──
        Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, rotary_dim,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, inp_pos, nullptr, rotary_dim,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // ── Flash Attention (bidirectional, GQA-native) ───────
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);  // [d, N, n_head]
        K = ggml_permute(ctx, K, 0, 2, 1, 3);  // [d, N, n_head_kv]
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        float attn_scale = 1.0f / sqrtf((float)head_dim);
        struct ggml_tensor * attn_out = ggml_flash_attn_ext(
                ctx, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
        attn_out = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_out), n_embd, N);

        // Output projection
        cur = ggml_mul_mat(ctx, ml.wo, attn_out);
        cur = ggml_add(ctx, cur, residual);

        // ── FFN: Dense or MoE ──────────────────────────────────
        residual = cur;
        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = ml.post_attn_norm;
            if (norm_w->type != GGML_TYPE_F32) norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            cur = ggml_mul(ctx, cur, norm_w);
        }

        // Save FFN input (after norm) for shared expert
        struct ggml_tensor * cur_ffn_input = cur;

        if (!ml.is_moe) {
            // ── Dense SwiGLU FFN ────────────────────────────────
            struct ggml_tensor * gate = ggml_mul_mat(ctx, ml.ffn_gate, cur);
            struct ggml_tensor * up   = ggml_mul_mat(ctx, ml.ffn_up,   cur);
            gate = ggml_silu(ctx, gate);
            cur  = ggml_mul(ctx, gate, up);
            cur  = ggml_mul_mat(ctx, ml.ffn_down, cur);
        } else {
            // ── MoE FFN ─────────────────────────────────────────
            //
            // 1. Router: logits = hidden @ gate_weight → [n_experts, N]
            // 2. Sigmoid scoring: scores = sigmoid(logits)
            // 3. Top-k selection: select top n_experts_used per token
            // 4. Normalize weights
            // 5. Expert computation via mul_mat_id
            // 6. Shared expert (always active)
            // 7. Sum routed + shared

            
            // Router logits: [n_experts, N]
            struct ggml_tensor * logits = ggml_mul_mat(ctx, ml.gate_weight, cur);

            // Sigmoid scores
            struct ggml_tensor * probs = ggml_sigmoid(ctx, logits);

            // Add expert bias if present
            if (ml.gate_bias) {
                probs = ggml_add(ctx, probs, ensure_f32_moe(ctx, ml.gate_bias));
            }

            // Top-k expert selection: argsort_top_k operates on ne[0]
            // probs is [n_experts, N], so result is [n_experts_used, N]
            struct ggml_tensor * selected_experts = ggml_argsort_top_k(ctx, probs, n_experts_used);

            // Get weights for selected experts
            // probs: [n_experts, N] → reshape to [1, n_experts, N] for get_rows
            probs = ggml_reshape_3d(ctx, ggml_cont(ctx, probs), 1, n_experts, N);
            struct ggml_tensor * weights = ggml_get_rows(ctx, probs, selected_experts);  // [1, n_experts_used, N]

            // Normalize weights (if norm_topk_prob)
            if (model.norm_topk_prob) {
                weights = ggml_reshape_2d(ctx, ggml_cont(ctx, weights), n_experts_used, N);
                struct ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights);  // [1, N]
                weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5f, INFINITY);
                weights = ggml_div(ctx, weights, weights_sum);
                weights = ggml_reshape_3d(ctx, ggml_cont(ctx, weights), 1, n_experts_used, N);
            }

            // Scale weights
            if (routed_scale != 0.0f && routed_scale != 1.0f) {
                weights = ggml_scale(ctx, weights, routed_scale);
            }

            // Expert computation via mul_mat_id
            // cur: [n_embd, N] → reshape to [n_embd, 1, N] for mul_mat_id
            struct ggml_tensor * cur_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, cur), n_embd, 1, N);

            // Gate path
            struct ggml_tensor * exp_gate = ggml_mul_mat_id(ctx, ml.expert_gate, cur_3d, selected_experts);
            // Up path
            struct ggml_tensor * exp_up   = ggml_mul_mat_id(ctx, ml.expert_up,   cur_3d, selected_experts);
            // SwiGLU: silu(gate) * up
            exp_gate = ggml_silu(ctx, exp_gate);
            struct ggml_tensor * exp_inter = ggml_mul(ctx, exp_gate, exp_up);
            // Down path
            struct ggml_tensor * exp_down  = ggml_mul_mat_id(ctx, ml.expert_down, exp_inter, selected_experts);

            // Weight and sum: result = sum_k(weight_k * exp_down_k)
            // exp_down: [n_embd, n_experts_used, N]
            // weights:  [1, n_experts_used, N]
            // Weight expert outputs and combine (like llama.cpp build_moe_ffn)
            // exp_down: [n_embd, n_experts_used, N]
            // weights:  [1, n_experts_used, N]
            struct ggml_tensor * weighted = ggml_mul(ctx, exp_down, weights);

            // Aggregate experts: view each expert as 2D [n_embd, N] and sum
            struct ggml_tensor * moe_out = nullptr;
            for (int e = 0; e < n_experts_used; e++) {
                struct ggml_tensor * exp_view = ggml_view_2d(ctx, weighted,
                    n_embd, N,
                    weighted->nb[2],      // stride per N
                    (size_t)e * weighted->nb[1]);  // offset for expert e
                if (e == 0) {
                    moe_out = exp_view;
                } else {
                    moe_out = ggml_add(ctx, moe_out, exp_view);
                }
            }
            // Make contiguous (moe_out may be a view)
            cur = ggml_cont(ctx, moe_out);

            // Shared expert (always active) — takes the same normalized input as MoE
            if (ml.shared_gate) {
                // 'cur_ffn_input' is the post-attention-norm normalized output
                // (computed before MoE routing)
                struct ggml_tensor * s_gate = ggml_mul_mat(ctx, ml.shared_gate, cur_ffn_input);
                struct ggml_tensor * s_up   = ggml_mul_mat(ctx, ml.shared_up,   cur_ffn_input);
                s_gate = ggml_silu(ctx, s_gate);
                struct ggml_tensor * s_inter = ggml_mul(ctx, s_gate, s_up);
                struct ggml_tensor * s_down  = ggml_mul_mat(ctx, ml.shared_down, s_inter);
                cur = ggml_add(ctx, cur, s_down);
            }
        }

        cur = ggml_add(ctx, cur, residual);
    }

    // ── Final norm + logits ────────────────────────────────────
    cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
    {
        struct ggml_tensor * norm_w = model.output_norm;
        if (norm_w->type != GGML_TYPE_F32) norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
        cur = ggml_mul(ctx, cur, norm_w);
    }
    cur = ggml_mul_mat(ctx, model.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ── MoE forward pass execution ──────────────────────────────────
bool diffuse_forward_moe(diffuse_context * ctx,
                         const int32_t * tokens, int n_tokens,
                         float * logits_out) {
    return diffuse_forward_moe_full(ctx, tokens, n_tokens, logits_out, nullptr);
}

bool diffuse_forward_moe_full(diffuse_context * ctx,
                              const int32_t * tokens, int n_tokens,
                              float * logits_out,
                              diffuse_step_cache * cache) {
    const auto & hp = ctx->model->hparams;

    // Use persistent compute buffer (MoE needs more for expert computation)
    size_t needed = diffuse_compute_buf_size(hp, n_tokens);
    // MoE needs extra for expert weights — scale up
    needed += (size_t)n_tokens * ctx->model->moe_intermediate
              * ctx->model->n_experts_per_tok * sizeof(float) * 6
              * ctx->model->hparams.n_layer;
    struct ggml_context * ctx_compute = diffuse_new_compute_ctx(ctx, needed);

    struct ggml_cgraph * gf = diffuse_build_graph_moe(ctx, ctx_compute, tokens, n_tokens);

    enum ggml_status status = ggml_graph_compute_with_ctx(ctx_compute, gf, ctx->n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        DIFFUSE_LOG("MoE graph compute failed with status %d", (int)status);
        ggml_free(ctx_compute);
        return false;
    }

    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        DIFFUSE_LOG("logits tensor not found in MoE graph");
        ggml_free(ctx_compute);
        return false;
    }
    memcpy(logits_out, logits->data, (size_t)n_tokens * hp.n_vocab * sizeof(float));

    ggml_free(ctx_compute);
    return true;
}

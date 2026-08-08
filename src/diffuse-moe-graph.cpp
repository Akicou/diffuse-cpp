#include "diffuse-graph.h"
#include "diffuse-backend.h"

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
        int block_length) {

    // Mask shape for flash_attn_ext: [n_kv, n_batch, ne32, ne33]
    // = [n_tokens_kv, n_tokens_q, 1, 1]
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(mask, "block_causal_mask");
    ggml_set_input(mask);

    float * mask_data = (float *)mask->data;

    if (block_length <= 0) {
        // Full bidirectional: all zeros
        for (int i = 0; i < n_tokens * n_tokens; i++) {
            mask_data[i] = 0.0f;
        }
    } else {
        // Block-causal: position i can attend to j iff block_id(i) >= block_id(j)
        for (int i = 0; i < n_tokens; i++) {
            int block_i = i / block_length;
            for (int j = 0; j < n_tokens; j++) {
                int block_j = j / block_length;
                // mask_data layout: [n_kv, n_batch] → [j, i] since ne[0]=n_tokens_kv=n_tokens
                // Wait: ggml 2D tensor ne[0] = n_tokens (columns/fast), ne[1] = n_tokens (rows)
                // Data is row-major: data[i * ne[0] + j] = mask[i][j]
                // For flash_attn_ext: mask is [n_kv, n_batch, ...] = [kv_len, q_len, ...]
                // So ne[0] = kv_len = n_tokens, ne[1] = q_len = n_tokens
                // data[q_idx * n_tokens + kv_idx]
                mask_data[i * n_tokens + j] = (block_i >= block_j) ? 0.0f : -INFINITY;
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
    const int n_kv_div    = n_head / n_head_kv;
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

    // Block-causal attention mask
    struct ggml_tensor * attn_mask = build_block_causal_mask(ctx, N, block_len);

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

        // Reshape for multi-head
        Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, N);
        K = ggml_reshape_3d(ctx, K, head_dim, n_head_kv, N);
        V = ggml_reshape_3d(ctx, V, head_dim, n_head_kv, N);

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

        // GQA: repeat KV heads
        if (n_kv_div > 1) {
            struct ggml_tensor * k_expanded = ggml_reshape_3d(ctx, K, K->ne[0], n_head, N);
            K = ggml_repeat(ctx, K, k_expanded);
            struct ggml_tensor * v_expanded = ggml_reshape_3d(ctx, V, V->ne[0], n_head, N);
            V = ggml_repeat(ctx, V, v_expanded);
        }

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
            // MoE layer
            // Router: sigmoid scoring → group-limited top-k selection
            const int n_experts = (int)model.n_experts;
            const int top_k = (int)model.n_experts_per_tok;

            // Router logits: [n_experts, N]
            struct ggml_tensor * router_logits = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.gate_weight), cur);

            // Sigmoid scores
            struct ggml_tensor * scores = ggml_sigmoid(ctx, router_logits);

            // Add expert_bias if present
            if (ml.gate_bias) {
                scores = ggml_add(ctx, scores, ensure_f32_moe(ctx, ml.gate_bias));
            }

            // Select top-k experts per token
            // ggml_argsort_top_k returns indices sorted by score descending
            struct ggml_tensor * topk_indices = ggml_argsort_top_k(ctx, scores, top_k);
            struct ggml_tensor * topk_scores  = ggml_get_rows(ctx, scores, topk_indices);

            // Normalize top-k scores
            struct ggml_tensor * topk_sum = ggml_sum_rows(ctx, topk_scores);
            if (model.norm_topk_prob) {
                topk_scores = ggml_div(ctx, topk_scores, topk_sum);
            }
            topk_scores = ggml_scale(ctx, topk_scores, model.routed_scaling);

            // Compute expert outputs via batched matmul
            // Expert weights are 3D: [n_embd_in, n_embd_out, n_experts]
            // For each token, select its top_k experts and compute their FFN

            // Reshape topk_indices for mul_mat_id: [top_k, N] → need as I32 1D [N * top_k]
            // Then use mul_mat_id which does: for each token t, for each selected expert e:
            //   result[t] += weight[e] @ input[t] * topk_weight[t, e]

            // gate_proj for selected experts
            struct ggml_tensor * expert_gate_out = ggml_mul_mat_id(ctx, ensure_f32_moe(ctx, ml.expert_gate), cur, topk_indices);
            expert_gate_out = ggml_silu(ctx, expert_gate_out);

            struct ggml_tensor * expert_up_out = ggml_mul_mat_id(ctx, ensure_f32_moe(ctx, ml.expert_up), cur, topk_indices);

            struct ggml_tensor * expert_inter = ggml_mul(ctx, expert_gate_out, expert_up_out);

            struct ggml_tensor * expert_down_out = ggml_mul_mat_id(ctx, ensure_f32_moe(ctx, ml.expert_down), expert_inter, topk_indices);

            // Weight by topk_scores and sum
            // expert_down_out shape: [n_embd, top_k, N]
            // topk_scores shape: [top_k, N]
            expert_down_out = ggml_mul(ctx, expert_down_out, topk_scores);

            // Sum over top_k dimension → [n_embd, N]
            // Reshape to [n_embd, top_k * N] then sum... 
            // Actually ggml mul_mat_id returns [n_embd, top_k, N] which is already 3D
            // We need to sum along the top_k dimension
            cur = ggml_sum_rows(ctx, expert_down_out);

            // Shared expert (always active)
            if (model.n_shared_experts > 0 && ml.shared_gate) {
                struct ggml_tensor * sg = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.shared_gate), cur);
                sg = ggml_silu(ctx, sg);
                struct ggml_tensor * su = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.shared_up), cur);
                struct ggml_tensor * si = ggml_mul(ctx, sg, su);
                struct ggml_tensor * sd = ggml_mul_mat(ctx, ensure_f32_moe(ctx, ml.shared_down), si);
                cur = ggml_add(ctx, cur, sd);
            }
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

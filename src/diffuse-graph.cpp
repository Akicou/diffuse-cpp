#include "diffuse-graph.h"
#include "diffuse-backend.h"

// ── Helper: ensure tensor is F32 (for bias add after quantization) ──
static struct ggml_tensor * ensure_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (!t) return nullptr;
    if (t->type != GGML_TYPE_F32) return ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// ── Compute buffer helpers ──
// (declared in diffuse-graph.h, defined below)
// Used by diffuse-graph.cpp, diffuse-moe-graph.cpp

// ── Build transformer forward graph ────────────────────────────
struct ggml_cgraph * diffuse_build_graph(
        diffuse_context * dctx,
        struct ggml_context * ctx,
        const int32_t * tokens,
        int n_tokens) {

    const auto & model = *dctx->model;
    const auto & hp    = model.hparams;
    const int n_embd      = (int)hp.n_embd;
    const int n_head      = (int)hp.n_head;
    const int n_head_kv   = (int)hp.n_head_kv;
    const int n_embd_head = (int)hp.n_embd_head();
    const int n_layer     = (int)hp.n_layer;
    const int N           = n_tokens;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, (size_t)(4096 * n_layer), false);

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
        const auto & layer = model.layers[il];
        struct ggml_tensor * residual = cur;

        // Pre-attention RMSNorm
        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.attn_norm;
            if (norm_w->type != GGML_TYPE_F32) {
                norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            }
            cur = ggml_mul(ctx, cur, norm_w);
        }

        // QKV projections
        struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.wq, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, layer.wk, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, layer.wv, cur);

        // Add QKV biases if present (Dream/Qwen2.5)
        if (layer.bq) Q = ggml_add(ctx, Q, ensure_f32(ctx, layer.bq));
        if (layer.bk) K = ggml_add(ctx, K, ensure_f32(ctx, layer.bk));
        if (layer.bv) V = ggml_add(ctx, V, ensure_f32(ctx, layer.bv));

        // Reshape for multi-head: [n_embd_head, n_head, N]
        Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head,    N);
        K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, N);
        V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, N);

        // RoPE on Q and K (NEOX style = non-interleaved, like LLaDA/OLMo)
        Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // ── Flash Attention (fused bidirectional attention) ──────
        // Replaces the manual K^T@Q → scale → softmax → V^T@attn path.
        // Adopted from llama.cpp's build_attn() which uses ggml_flash_attn_ext
        // for diffusion models (LLaDA, Dream) with non-causal attention.
        //
        // Key advantages over the old manual attention:
        //   1. Avoids materializing the N×N×n_head attention matrix
        //   2. Handles GQA natively — no need to expand K,V from n_head_kv to n_head
        //   3. mask=nullptr → no masking = full bidirectional attention
        //
        // flash_attn_ext expects:
        //   Q: [n_embd_head, N, n_head,    1]
        //   K: [n_embd_head, N, n_head_kv, 1]  (GQA broadcast handled by kernel)
        //   V: [n_embd_head, N, n_head_kv, 1]  (same layout as K, contiguous)
        //   mask: nullptr (bidirectional — all positions attend to all)
        //   result: [n_embd_head, n_head, N, 1] → reshape to [n_embd, N]
        //
        // No GQA expansion needed — flash_attn_ext broadcasts n_head_kv → n_head.

        // Q: [n_embd_head, n_head, N] → [n_embd_head, N, n_head]
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        // K,V: [n_embd_head, n_head_kv, N] → [n_embd_head, N, n_head_kv]
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        float attn_scale = 1.0f / sqrtf((float)n_embd_head);
        struct ggml_tensor * attn_out = ggml_flash_attn_ext(
                ctx, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);

        // Result: [n_embd_head, n_head, N, 1] → [n_embd, N]
        attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, N);

        // Output projection
        cur = ggml_mul_mat(ctx, layer.wo, attn_out);
        cur = ggml_add(ctx, cur, residual);

        // ── FFN (SwiGLU) ───────────────────────────────────────
        residual = cur;
        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.ffn_norm;
            if (norm_w->type != GGML_TYPE_F32) {
                norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            }
            cur = ggml_mul(ctx, cur, norm_w);
        }

        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up,   cur);
        gate = ggml_silu(ctx, gate);
        cur  = ggml_mul(ctx, gate, up);
        cur  = ggml_mul_mat(ctx, layer.ffn_down, cur);

        cur = ggml_add(ctx, cur, residual);
    }

    // ── Final norm + logits ────────────────────────────────────
    cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
    {
        struct ggml_tensor * norm_w = model.output_norm;
        if (norm_w->type != GGML_TYPE_F32) {
            norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
        }
        cur = ggml_mul(ctx, cur, norm_w);
    }
    cur = ggml_mul_mat(ctx, model.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ── Build full graph WITH named K,V for cache extraction ────────
// Same as diffuse_build_graph but marks K,V tensors as outputs so
// they can be extracted after execution to populate the cache.
static struct ggml_cgraph * diffuse_build_graph_extractable(
        diffuse_context * dctx,
        struct ggml_context * ctx,
        const int32_t * tokens,
        int n_tokens) {

    const auto & model = *dctx->model;
    const auto & hp    = model.hparams;
    const int n_embd      = (int)hp.n_embd;
    const int n_head      = (int)hp.n_head;
    const int n_head_kv   = (int)hp.n_head_kv;
    const int n_embd_head = (int)hp.n_embd_head();
    const int n_layer     = (int)hp.n_layer;
    const int N           = n_tokens;

    // Extra nodes for the K,V output markers
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx,
            (size_t)(4096 * n_layer + n_layer * 4), false);

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

    struct ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_tokens);

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        struct ggml_tensor * residual = cur;

        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.attn_norm;
            if (norm_w->type != GGML_TYPE_F32) {
                norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            }
            cur = ggml_mul(ctx, cur, norm_w);
        }

        struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.wq, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, layer.wk, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, layer.wv, cur);

        // Add QKV biases if present (Dream/Qwen2.5)
        if (layer.bq) Q = ggml_add(ctx, Q, ensure_f32(ctx, layer.bq));
        if (layer.bk) K = ggml_add(ctx, K, ensure_f32(ctx, layer.bk));
        if (layer.bv) V = ggml_add(ctx, V, ensure_f32(ctx, layer.bv));

        Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head,    N);
        K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, N);
        V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, N);

        Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // GQA: repeat each KV head n_rep times (grouped, not interleaved)
        if (n_head_kv < n_head) {
            const int n_rep = n_head / n_head_kv;
            K = ggml_reshape_4d(ctx, K, n_embd_head, 1, n_head_kv, N);
            K = ggml_repeat(ctx, K,
                    ggml_new_tensor_4d(ctx, K->type, n_embd_head, n_rep, n_head_kv, N));
            K = ggml_reshape_3d(ctx, K, n_embd_head, n_head, N);

            V = ggml_reshape_4d(ctx, V, n_embd_head, 1, n_head_kv, N);
            V = ggml_repeat(ctx, V,
                    ggml_new_tensor_4d(ctx, V->type, n_embd_head, n_rep, n_head_kv, N));
            V = ggml_reshape_3d(ctx, V, n_embd_head, n_head, N);
        }

        // ── Name K,V for cache extraction (BEFORE permute) ──────
        // Shape at this point: [n_embd_head, n_head, N]
        {
            char name_buf[32];
            snprintf(name_buf, sizeof(name_buf), "Kc.%02d", il);
            ggml_set_name(K, name_buf);
            ggml_set_output(K);

            snprintf(name_buf, sizeof(name_buf), "Vc.%02d", il);
            ggml_set_name(V, name_buf);
            ggml_set_output(V);
        }

        // ── Flash Attention (fused bidirectional) ──────────────
        // K,V are GQA-expanded here; flash_attn_ext processes them as
        // n_head==n_head_kv (MHA). Still avoids N×N matrix materialization.
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);  // [d, N, n_head]
        K = ggml_permute(ctx, K, 0, 2, 1, 3);  // [d, N, n_head]
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));  // [d, N, n_head]

        float attn_scale = 1.0f / sqrtf((float)n_embd_head);
        struct ggml_tensor * attn_out = ggml_flash_attn_ext(
                ctx, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
        attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, N);

        cur = ggml_mul_mat(ctx, layer.wo, attn_out);
        cur = ggml_add(ctx, cur, residual);

        residual = cur;
        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.ffn_norm;
            if (norm_w->type != GGML_TYPE_F32) {
                norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            }
            cur = ggml_mul(ctx, cur, norm_w);
        }

        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up,   cur);
        gate = ggml_silu(ctx, gate);
        cur  = ggml_mul(ctx, gate, up);
        cur  = ggml_mul_mat(ctx, layer.ffn_down, cur);
        cur = ggml_add(ctx, cur, residual);
    }

    cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
    {
        struct ggml_tensor * norm_w = model.output_norm;
        if (norm_w->type != GGML_TYPE_F32) {
            norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
        }
        cur = ggml_mul(ctx, cur, norm_w);
    }
    cur = ggml_mul_mat(ctx, model.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ── Full forward pass with cache extraction ─────────────────────
bool diffuse_forward_full(diffuse_context * ctx,
                          const int32_t * tokens, int n_tokens,
                          float * logits_out,
                          diffuse_step_cache * cache) {
    const auto & hp = ctx->model->hparams;

    // Use persistent compute buffer (avoids multi-GB malloc/free each step)
    size_t needed = diffuse_compute_buf_size(hp, n_tokens);
    struct ggml_context * ctx_compute = diffuse_new_compute_ctx(ctx, needed);

    struct ggml_cgraph * gf = (cache != nullptr)
        ? diffuse_build_graph_extractable(ctx, ctx_compute, tokens, n_tokens)
        : diffuse_build_graph(ctx, ctx_compute, tokens, n_tokens);

    if (!diffuse_sched_compute(ctx, ctx_compute, gf)) {
        DIFFUSE_LOG("graph compute failed");
        ggml_free(ctx_compute);
        return false;
    }

    // Extract logits
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        DIFFUSE_LOG("logits tensor not found");
        ggml_free(ctx_compute);
        return false;
    }
    memcpy(logits_out, logits->data, (size_t)n_tokens * hp.n_vocab * sizeof(float));

    // Extract K,V into cache
    if (cache) {
        const int n_layer = (int)hp.n_layer;
        size_t kv_bytes = cache->pos_stride() * n_tokens * sizeof(float);
        char name_buf[32];

        for (int il = 0; il < n_layer; il++) {
            snprintf(name_buf, sizeof(name_buf), "Kc.%02d", il);
            struct ggml_tensor * K_t = ggml_graph_get_tensor(gf, name_buf);
            snprintf(name_buf, sizeof(name_buf), "Vc.%02d", il);
            struct ggml_tensor * V_t = ggml_graph_get_tensor(gf, name_buf);

            if (K_t && V_t) {
                memcpy(cache->K[il].data(), K_t->data, kv_bytes);
                memcpy(cache->V[il].data(), V_t->data, kv_bytes);
            } else {
                DIFFUSE_LOG("WARNING: could not extract K/V for layer %d", il);
            }
        }
    }

    ggml_free(ctx_compute);
    return true;
}

// ── Build CACHED graph for active positions only ────────────────
//
// Architecture:
//   - Compute Q, K, V projections ONLY for active_tokens
//   - For attention: K_full = concat(K_cached, K_active), V_full = concat(V_cached, V_active)
//   - K_cached contains the cached K,V for INACTIVE positions (reordered)
//   - The attention is Q_active × K_full, giving scores [n_total, n_active, n_head]
//   - FFN and logits computed only for n_active positions
//
// Position ordering:
//   K_full[0..n_cached-1] = cached positions (in order of cached_positions[])
//   K_full[n_cached..n_total-1] = active positions (in order of active_positions[])
//   RoPE uses the ORIGINAL absolute positions.

struct ggml_cgraph * diffuse_build_graph_cached(
        diffuse_context * dctx,
        struct ggml_context * ctx,
        const int32_t * active_tokens,
        const int32_t * active_pos_indices,
        int n_active,
        int n_total,
        diffuse_step_cache * cache,
        const std::vector<int> & cached_positions) {

    const auto & model = *dctx->model;
    const auto & hp    = model.hparams;
    const int n_embd      = (int)hp.n_embd;
    const int n_head      = (int)hp.n_head;
    const int n_head_kv   = (int)hp.n_head_kv;
    const int n_embd_head = (int)hp.n_embd_head();
    const int n_layer     = (int)hp.n_layer;
    const int n_cached    = n_total - n_active;
    const size_t kv_stride = cache->pos_stride();  // n_embd_head * n_head

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx,
            (size_t)(4096 * n_layer + n_layer * 8 + 256), false);

    // ── Active position inputs ───────────────────────────────────
    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_active);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    memcpy(inp_tokens->data, active_tokens, n_active * sizeof(int32_t));

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_active);
    ggml_set_name(inp_pos, "inp_pos_active");
    ggml_set_input(inp_pos);
    memcpy(inp_pos->data, active_pos_indices, n_active * sizeof(int32_t));

    // We also need position indices for the cached positions (for V permutation ordering)
    // — not needed for RoPE since cached K,V already have RoPE applied

    // ── Embedding for active positions ───────────────────────────
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, inp_tokens);

    // ── Transformer layers ───────────────────────────────────────
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        struct ggml_tensor * residual = cur;

        // Pre-attention RMSNorm
        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.attn_norm;
            if (norm_w->type != GGML_TYPE_F32) {
                norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            }
            cur = ggml_mul(ctx, cur, norm_w);
        }

        // QKV projections for active positions only
        struct ggml_tensor * Q_active = ggml_mul_mat(ctx, layer.wq, cur);
        struct ggml_tensor * K_active = ggml_mul_mat(ctx, layer.wk, cur);
        struct ggml_tensor * V_active = ggml_mul_mat(ctx, layer.wv, cur);

        // Add QKV biases if present (Dream/Qwen2.5)
        if (layer.bq) Q_active = ggml_add(ctx, Q_active, ensure_f32(ctx, layer.bq));
        if (layer.bk) K_active = ggml_add(ctx, K_active, ensure_f32(ctx, layer.bk));
        if (layer.bv) V_active = ggml_add(ctx, V_active, ensure_f32(ctx, layer.bv));

        // Reshape: [n_embd, n_active] → [n_embd_head, n_head(_kv), n_active]
        Q_active = ggml_reshape_3d(ctx, Q_active, n_embd_head, n_head,    n_active);
        K_active = ggml_reshape_3d(ctx, K_active, n_embd_head, n_head_kv, n_active);
        V_active = ggml_reshape_3d(ctx, V_active, n_embd_head, n_head_kv, n_active);

        // RoPE on Q and K with ORIGINAL position indices
        Q_active = ggml_rope_ext(ctx, Q_active, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_active = ggml_rope_ext(ctx, K_active, inp_pos, nullptr, n_embd_head,
                          GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // GQA: repeat each KV head n_rep times (grouped, not interleaved)
        if (n_head_kv < n_head) {
            const int n_rep = n_head / n_head_kv;
            K_active = ggml_reshape_4d(ctx, K_active, n_embd_head, 1, n_head_kv, n_active);
            K_active = ggml_repeat(ctx, K_active,
                    ggml_new_tensor_4d(ctx, K_active->type, n_embd_head, n_rep, n_head_kv, n_active));
            K_active = ggml_reshape_3d(ctx, K_active, n_embd_head, n_head, n_active);

            V_active = ggml_reshape_4d(ctx, V_active, n_embd_head, 1, n_head_kv, n_active);
            V_active = ggml_repeat(ctx, V_active,
                    ggml_new_tensor_4d(ctx, V_active->type, n_embd_head, n_rep, n_head_kv, n_active));
            V_active = ggml_reshape_3d(ctx, V_active, n_embd_head, n_head, n_active);
        }

        // ── Name K_active, V_active for extraction ──────────────
        {
            char name_buf[32];
            snprintf(name_buf, sizeof(name_buf), "Ka.%02d", il);
            ggml_set_name(K_active, name_buf);
            ggml_set_output(K_active);

            snprintf(name_buf, sizeof(name_buf), "Va.%02d", il);
            ggml_set_name(V_active, name_buf);
            ggml_set_output(V_active);
        }

        // ── Load cached K,V for inactive positions ──────────────
        // Shape: [n_embd_head, n_head, n_cached]
        struct ggml_tensor * K_cached = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                n_embd_head, n_head, n_cached);
        ggml_set_input(K_cached);

        struct ggml_tensor * V_cached = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                n_embd_head, n_head, n_cached);
        ggml_set_input(V_cached);

        // Fill from cache: gather cached positions into contiguous tensor
        {
            float * K_dst = (float *)K_cached->data;
            float * V_dst = (float *)V_cached->data;
            for (int c = 0; c < n_cached; c++) {
                int orig_pos = cached_positions[c];
                memcpy(K_dst + c * kv_stride,
                       cache->K[il].data() + orig_pos * kv_stride,
                       kv_stride * sizeof(float));
                memcpy(V_dst + c * kv_stride,
                       cache->V[il].data() + orig_pos * kv_stride,
                       kv_stride * sizeof(float));
            }
        }

        // ── Concatenate: K_full = [K_cached | K_active] ─────────
        // dim=2 is the sequence/position dimension
        struct ggml_tensor * K_full = ggml_concat(ctx, K_cached, K_active, 2);
        struct ggml_tensor * V_full = ggml_concat(ctx, V_cached, V_active, 2);
        // K_full shape: [n_embd_head, n_head, n_total]

        // ── Flash Attention (fused, with cached K,V) ────────────
        // Q: [n_embd_head, n_head, n_active] → [d, n_active, n_head]
        Q_active = ggml_permute(ctx, Q_active, 0, 2, 1, 3);
        // K_full, V_full: [d, n_head, n_total] → [d, n_total, n_head]
        K_full = ggml_permute(ctx, K_full, 0, 2, 1, 3);
        V_full = ggml_cont(ctx, ggml_permute(ctx, V_full, 0, 2, 1, 3));

        float attn_scale = 1.0f / sqrtf((float)n_embd_head);
        struct ggml_tensor * attn_out = ggml_flash_attn_ext(
                ctx, Q_active, K_full, V_full, nullptr, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
        attn_out = ggml_reshape_2d(ctx, attn_out, n_embd, n_active);

        // Output projection
        cur = ggml_mul_mat(ctx, layer.wo, attn_out);
        cur = ggml_add(ctx, cur, residual);

        // ── FFN (SwiGLU) — only for active positions ────────────
        residual = cur;
        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.ffn_norm;
            if (norm_w->type != GGML_TYPE_F32) {
                norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            }
            cur = ggml_mul(ctx, cur, norm_w);
        }

        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up,   cur);
        gate = ggml_silu(ctx, gate);
        cur  = ggml_mul(ctx, gate, up);
        cur  = ggml_mul_mat(ctx, layer.ffn_down, cur);
        cur = ggml_add(ctx, cur, residual);
    }

    // ── Final norm + logits (active positions only) ──────────────
    cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
    {
        struct ggml_tensor * norm_w = model.output_norm;
        if (norm_w->type != GGML_TYPE_F32) {
            norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
        }
        cur = ggml_mul(ctx, cur, norm_w);
    }
    cur = ggml_mul_mat(ctx, model.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ── Cached forward pass execution ───────────────────────────────
bool diffuse_forward_cached(
        diffuse_context * ctx,
        const int32_t * active_tokens,
        const int32_t * active_pos_indices,
        int n_active,
        int n_total,
        diffuse_step_cache * cache,
        const std::vector<int> & cached_positions,
        const std::vector<int> & active_positions,
        float * logits_out) {

    const auto & hp = ctx->model->hparams;
    const int n_cached = n_total - n_active;

    // Use persistent compute buffer (reuse across cached steps)
    // The cached path needs less memory than the full path, so the
    // pre-allocated buffer from diffuse_context_new will suffice.
    size_t needed = diffuse_compute_buf_size(hp, std::max(n_total, n_active));
    struct ggml_context * ctx_compute = diffuse_new_compute_ctx(ctx, needed);

    struct ggml_cgraph * gf = diffuse_build_graph_cached(
            ctx, ctx_compute,
            active_tokens, active_pos_indices,
            n_active, n_total,
            cache, cached_positions);

    if (!diffuse_sched_compute(ctx, ctx_compute, gf)) {
        DIFFUSE_LOG("cached graph compute failed");
        ggml_free(ctx_compute);
        return false;
    }

    // Extract logits for active positions
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        DIFFUSE_LOG("logits tensor not found in cached graph");
        ggml_free(ctx_compute);
        return false;
    }
    memcpy(logits_out, logits->data, (size_t)n_active * hp.n_vocab * sizeof(float));

    // Update cache: store K_active, V_active for active positions
    {
        char name_buf[32];
        for (int il = 0; il < (int)hp.n_layer; il++) {
            snprintf(name_buf, sizeof(name_buf), "Ka.%02d", il);
            struct ggml_tensor * K_t = ggml_graph_get_tensor(gf, name_buf);
            snprintf(name_buf, sizeof(name_buf), "Va.%02d", il);
            struct ggml_tensor * V_t = ggml_graph_get_tensor(gf, name_buf);

            if (K_t && V_t) {
                cache->update_kv(il, (const float *)K_t->data,
                                     (const float *)V_t->data,
                                     active_positions);
            }
        }
    }

    ggml_free(ctx_compute);
    return true;
}

// ── Forward pass dispatcher ─────────────────────────────────────
// Routes to MoE or standard forward based on model type.
bool diffuse_forward(diffuse_context * ctx,
                     const int32_t * tokens, int n_tokens,
                     float * logits_out) {
    if (ctx->model->model_type == "llada2_moe") {
        return diffuse_forward_moe(ctx, tokens, n_tokens, logits_out);
    }
    return diffuse_forward_full(ctx, tokens, n_tokens, logits_out, nullptr);
}

// ── Compute buffer sizing ──────────────────────────────────────
// Estimate the GGML arena needed for one forward pass. The buffer must
// hold all intermediate tensors for n_layer transformer blocks plus the
// final logits projection. We size conservatively; the persistent buffer
// is reused across steps, so over-allocation is one-time.
size_t diffuse_compute_buf_size(const diffuse_hparams & hp, int n_tokens) {
    const size_t n_embd = hp.n_embd;
    const size_t n_head = hp.n_head;
    const size_t n_ff   = hp.n_ff;
    const size_t n_layer = hp.n_layer;

    // Per layer: QKV projections, attention intermediates, norms, FFN
    // Note: flash attention avoids the N*N*H attention matrix, so we
    // only need O(N*d) for attention instead of O(N^2*H).
    // The terms below over-estimate slightly to stay safe.
    size_t per_layer =
        (size_t)n_tokens * n_embd * sizeof(float) * 10     // QKV proj + norms + residuals
      + (size_t)n_tokens * n_ff   * sizeof(float) * 3      // FFN (gate, up, down, intermediate)
      + (size_t)n_tokens * n_embd * sizeof(float) * 2      // attention (Q,K,V + output) — flash path
      + n_embd * sizeof(float) * 2                          // norm weights
      ;

    size_t buf_size = per_layer * n_layer;
    buf_size += (size_t)n_tokens * hp.n_vocab * sizeof(float) * 2;  // logits
    buf_size += 256ull * 1024 * 1024;                               // overhead
    buf_size = (size_t)(buf_size * 1.5);                            // headroom

    return buf_size;
}

// ── Allocate or grow the persistent compute buffer ──────────────
void diffuse_ensure_compute_buf(diffuse_context * dctx, size_t needed) {
    if (dctx->compute_buf_size >= needed) return;

    // Free old buffer if present
    if (dctx->compute_buf) {
        free(dctx->compute_buf);
    }

    dctx->compute_buf = malloc(needed);
    if (!dctx->compute_buf) {
        DIFFUSE_DIE("failed to allocate compute buffer (%zu MB)", needed / (1024*1024));
    }
    dctx->compute_buf_size = needed;

    DIFFUSE_LOG("compute buffer: %zu MB (persistent)", needed / (1024*1024));
}

// ── Create a GGML context on the persistent buffer ──────────────
// This re-initializes a fresh GGML arena on top of the pre-allocated
// memory, effectively "resetting" it for the next graph build. This is
// the key optimization: we avoid malloc/free of multi-GB buffers on
// every diffusion step.
struct ggml_context * diffuse_new_compute_ctx(diffuse_context * dctx, size_t needed) {
    diffuse_ensure_compute_buf(dctx, needed);

    struct ggml_init_params cparams = {
        /*.mem_size   = */ dctx->compute_buf_size,
        /*.mem_buffer = */ dctx->compute_buf,
        /*.no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(cparams);
    if (!ctx) {
        DIFFUSE_DIE("failed to initialize compute context (%zu MB)", dctx->compute_buf_size / (1024*1024));
    }
    return ctx;
}

// ── Context management ─────────────────────────────────────────
diffuse_context * diffuse_context_new(const diffuse_model * model, int n_ctx, int n_threads) {
    return diffuse_context_new_gpu(model, n_ctx, n_threads, 0);
}

diffuse_context * diffuse_context_new_gpu(const diffuse_model * model, int n_ctx, int n_threads, int n_gpu_layers) {
    auto * ctx = new diffuse_context();
    ctx->model       = model;
    ctx->n_ctx       = n_ctx;
    ctx->n_threads   = n_threads;
    ctx->n_gpu_layers = n_gpu_layers;

    // Initialize backends (CPU always, GPU if requested)
    diffuse_init_backends(ctx, n_gpu_layers);

    // Pre-allocate compute buffer for CPU-only fallback path.
    // When GPU backend is active, the scheduler manages its own buffers.
    if (n_gpu_layers == 0) {
        const auto & hp = model->hparams;
        size_t max_tokens = (size_t)n_ctx + 256;
        size_t needed = diffuse_compute_buf_size(hp, (int)max_tokens);
        diffuse_ensure_compute_buf(ctx, needed);
    }

    return ctx;
}

void diffuse_context_free(diffuse_context * ctx) {
    if (!ctx) return;
    if (ctx->sched)      ggml_backend_sched_free(ctx->sched);
    if (ctx->buf)        ggml_backend_buffer_free(ctx->buf);
    if (ctx->backend_gpu) ggml_backend_free(ctx->backend_gpu);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    if (ctx->ctx)        ggml_free(ctx->ctx);
    if (ctx->compute_buf) free(ctx->compute_buf);
    delete ctx;
}

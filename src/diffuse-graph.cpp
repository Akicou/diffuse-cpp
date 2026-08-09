#include "diffuse-graph.h"
#include "diffuse-backend.h"

// ── Helper: ensure tensor is F32 ──
static struct ggml_tensor * ensure_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (!t) return nullptr;
    if (t->type != GGML_TYPE_F32) return ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// ── Build dense transformer graph (LLaDA 1.0 / Dream) ─────────
// Uses full bidirectional attention (flat diffusion)
static struct ggml_cgraph * diffuse_build_graph_dense(
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

    const int n_kv_div = n_head / n_head_kv;

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        struct ggml_tensor * residual = cur;

        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.attn_norm;
            if (norm_w->type != GGML_TYPE_F32) norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            cur = ggml_mul(ctx, cur, norm_w);
        }

        struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.wq, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx, layer.wk, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx, layer.wv, cur);

        if (layer.bq) Q = ggml_add(ctx, Q, ensure_f32(ctx, layer.bq));
        if (layer.bk) K = ggml_add(ctx, K, ensure_f32(ctx, layer.bk));
        if (layer.bv) V = ggml_add(ctx, V, ensure_f32(ctx, layer.bv));

        Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head, N);
        K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, N);
        V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, N);

        // RoPE
        cur = ggml_rope_ext(ctx, Q, inp_pos, nullptr, n_embd_head, 0, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        struct ggml_tensor * Kr = ggml_rope_ext(ctx, K, inp_pos, nullptr, n_embd_head, 0, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // GQA: repeat KV heads
        if (n_kv_div > 1) {
            Kr = ggml_cont(ctx, Kr);
            V  = ggml_cont(ctx, V);
            Kr = ggml_repeat(ctx, Kr, ggml_new_tensor_3d(ctx, Kr->type, Kr->ne[0], n_head, N));
            V  = ggml_repeat(ctx, V,  ggml_new_tensor_3d(ctx, V->type,  V->ne[0], n_head, N));
        }

        // Full bidirectional attention (flat diffusion — no causal mask)
        cur = ggml_flash_attn_ext(ctx, cur, Kr, V, nullptr, 1.0f / sqrtf((float)n_embd_head), 0.0f, 0.0f);
        cur = ggml_reshape_2d(ctx, cur, n_embd, N);

        {
            struct ggml_tensor * wo = layer.wo;
            if (wo->type != GGML_TYPE_F32) wo = ggml_cast(ctx, wo, GGML_TYPE_F32);
            cur = ggml_mul_mat(ctx, wo, cur);
        }
        cur = ggml_add(ctx, cur, residual);

        // FFN
        residual = cur;
        cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
        {
            struct ggml_tensor * norm_w = layer.ffn_norm;
            if (norm_w->type != GGML_TYPE_F32) norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
            cur = ggml_mul(ctx, cur, norm_w);
        }

        struct ggml_tensor * gate = layer.ffn_gate;
        struct ggml_tensor * up   = layer.ffn_up;
        struct ggml_tensor * down = layer.ffn_down;
        if (gate->type != GGML_TYPE_F32) gate = ggml_cast(ctx, gate, GGML_TYPE_F32);
        if (up->type != GGML_TYPE_F32)   up   = ggml_cast(ctx, up, GGML_TYPE_F32);
        if (down->type != GGML_TYPE_F32) down = ggml_cast(ctx, down, GGML_TYPE_F32);

        struct ggml_tensor * g = ggml_mul_mat(ctx, gate, cur);
        g = ggml_silu(ctx, g);
        struct ggml_tensor * u = ggml_mul_mat(ctx, up, cur);
        cur = ggml_mul(ctx, g, u);
        cur = ggml_mul_mat(ctx, down, cur);
        cur = ggml_add(ctx, cur, residual);
    }

    // Output norm + projection
    cur = ggml_rms_norm(ctx, cur, hp.rms_norm_eps);
    {
        struct ggml_tensor * norm_w = model.output_norm;
        if (norm_w->type != GGML_TYPE_F32) norm_w = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
        cur = ggml_mul(ctx, cur, norm_w);
    }

    struct ggml_tensor * logits = ggml_mul_mat(ctx, model.output, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);
    return gf;
}

// ── Dense forward pass ─────────────────────────────────────────
bool diffuse_forward_dense(diffuse_context * ctx,
                           const int32_t * tokens, int n_tokens,
                           float * logits_out) {
    const auto & hp = ctx->model->hparams;
    size_t needed = diffuse_compute_buf_size(hp, n_tokens);
    struct ggml_context * ctx_compute = diffuse_new_compute_ctx(ctx, needed);

    struct ggml_cgraph * gf = diffuse_build_graph_dense(ctx, ctx_compute, tokens, n_tokens);

    if (!diffuse_sched_compute(ctx, ctx_compute, gf)) {
        DIFFUSE_LOG("dense graph compute failed");
        ggml_free(ctx_compute);
        return false;
    }

    struct ggml_tensor * logits = ggml_graph_node(gf, -1);
    memcpy(logits_out, logits->data, (size_t)n_tokens * hp.n_vocab * sizeof(float));

    ggml_free(ctx_compute);
    return true;
}

// ── Public forward dispatcher ──────────────────────────────────
bool diffuse_forward(diffuse_context * ctx,
                     const int32_t * tokens, int n_tokens,
                     float * logits_out) {
    switch (ctx->model->arch) {
        case diffuse_arch::LLaDA2_MoE:
            return diffuse_forward_moe(ctx, tokens, n_tokens, logits_out);
        case diffuse_arch::LLaDA:
        case diffuse_arch::DREAM:
            return diffuse_forward_dense(ctx, tokens, n_tokens, logits_out);
    }
    return false;
}

// ── Compute buffer sizing ──────────────────────────────────────
size_t diffuse_compute_buf_size(const diffuse_hparams & hp, int n_tokens) {
    const size_t n_embd = hp.n_embd;
    const size_t n_ff   = hp.n_ff;
    const size_t n_layer = hp.n_layer;

    size_t per_layer =
        (size_t)n_tokens * n_embd * sizeof(float) * 12 +
        (size_t)n_tokens * n_ff   * sizeof(float) * 3 +
        n_embd * sizeof(float) * 2;

    // Block-causal mask
    if (hp.block_length > 0) {
        per_layer += (size_t)n_tokens * n_tokens * sizeof(float);
    }

    size_t buf_size = per_layer * n_layer;
    buf_size += (size_t)n_tokens * hp.n_vocab * sizeof(float) * 2;
    buf_size += 256ull * 1024 * 1024;
    buf_size = (size_t)(buf_size * 1.5);

    return buf_size;
}

// ── Compute buffer management ──────────────────────────────────
void diffuse_ensure_compute_buf(diffuse_context * dctx, size_t needed) {
    if (dctx->compute_buf_size >= needed) return;
    if (dctx->compute_buf) free(dctx->compute_buf);
    dctx->compute_buf = malloc(needed);
    if (!dctx->compute_buf) DIFFUSE_DIE("failed to allocate compute buffer (%zu MB)", needed / (1024*1024));
    dctx->compute_buf_size = needed;
    DIFFUSE_LOG("compute buffer: %zu MB", needed / (1024*1024));
}

struct ggml_context * diffuse_new_compute_ctx(diffuse_context * dctx, size_t needed) {
    diffuse_ensure_compute_buf(dctx, needed);
    struct ggml_init_params cparams = { dctx->compute_buf_size, dctx->compute_buf, false };
    struct ggml_context * ctx = ggml_init(cparams);
    if (!ctx) DIFFUSE_DIE("failed to init compute context (%zu MB)", dctx->compute_buf_size / (1024*1024));
    return ctx;
}

// ── Context management ─────────────────────────────────────────
diffuse_context * diffuse_context_new(const diffuse_model * model, int n_ctx, int n_threads) {
    return diffuse_context_new_gpu(model, n_ctx, n_threads, 0);
}

diffuse_context * diffuse_context_new_gpu(const diffuse_model * model, int n_ctx, int n_threads, int n_gpu_layers) {
    auto * ctx = new diffuse_context();
    ctx->model = model;
    ctx->n_ctx = n_ctx;
    ctx->n_threads = n_threads;
    ctx->n_gpu_layers = n_gpu_layers;
    diffuse_init_backends(ctx, n_gpu_layers);

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

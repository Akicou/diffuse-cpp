#include "diffuse-graph.h"
#include "diffuse-backend.h"

// ── Helper: ensure tensor is F32 (for bias add after quantization) ──
static struct ggml_tensor * ensure_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (!t) return nullptr;
    if (t->type != GGML_TYPE_F32) return ggml_cast(ctx, t, GGML_TYPE_F32);
    return t;
}

// ── Public forward pass (MoE only) ─────────────────────────────
bool diffuse_forward(diffuse_context * ctx,
                     const int32_t * tokens, int n_tokens,
                     float * logits_out) {
    return diffuse_forward_moe(ctx, tokens, n_tokens, logits_out);
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

    // Per layer: QKV projections, attention intermediates, norms, FFN/MoE
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

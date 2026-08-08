#include "diffuse-model.h"

// ── GGUF metadata key helpers ──────────────────────────────────
static int64_t find_key(const struct gguf_context * gctx, const char * key) {
    int64_t id = gguf_find_key(gctx, key);
    if (id < 0) {
        DIFFUSE_DIE("missing GGUF metadata key: %s", key);
    }
    return id;
}

static uint32_t get_u32(const struct gguf_context * gctx, const char * key) {
    return gguf_get_val_u32(gctx, find_key(gctx, key));
}

static float get_f32(const struct gguf_context * gctx, const char * key, float def) {
    int64_t id = gguf_find_key(gctx, key);
    if (id < 0) return def;
    return gguf_get_val_f32(gctx, id);
}

// ── Multi-arch key resolution ────────────────────────────────────
// Try diffuse.*, then qwen2.*, then llama.* prefixes.
// This allows loading GGUF files produced by llama.cpp's converter.
// required = true: die if not found and no default
// required = false: return def if not found
static uint32_t get_u32_multi(const struct gguf_context * gctx,
                               const char * suffix,
                               uint32_t def = 0,
                               bool required = true) {
    static const char * prefixes[] = { "diffuse", "qwen2", "llama", nullptr };
    char buf[128];
    for (int i = 0; prefixes[i]; i++) {
        snprintf(buf, sizeof(buf), "%s.%s", prefixes[i], suffix);
        int64_t id = gguf_find_key(gctx, buf);
        if (id >= 0) {
            return gguf_get_val_u32(gctx, id);
        }
    }
    if (!required) return def;
    DIFFUSE_DIE("missing GGUF metadata key: *.%s (tried diffuse/qwen2/llama)", suffix);
    return 0;  // unreachable
}

static float get_f32_multi(const struct gguf_context * gctx,
                            const char * suffix, float def) {
    static const char * prefixes[] = { "diffuse", "qwen2", "llama", nullptr };
    char buf[128];
    for (int i = 0; prefixes[i]; i++) {
        snprintf(buf, sizeof(buf), "%s.%s", prefixes[i], suffix);
        int64_t id = gguf_find_key(gctx, buf);
        if (id >= 0) {
            return gguf_get_val_f32(gctx, id);
        }
    }
    return def;
}

// Detect the architecture prefix used in this GGUF file
static std::string detect_arch_prefix(const struct gguf_context * gctx) {
    // Check general.architecture first (standard llama.cpp key)
    int64_t id = gguf_find_key(gctx, "general.architecture");
    if (id >= 0) {
        return gguf_get_val_str(gctx, id);
    }
    // Fallback: check which prefix has block_count
    if (gguf_find_key(gctx, "diffuse.block_count") >= 0) return "diffuse";
    if (gguf_find_key(gctx, "qwen2.block_count") >= 0) return "qwen2";
    if (gguf_find_key(gctx, "llama.block_count") >= 0) return "llama";
    return "unknown";
}

// ── Tensor lookup helper ───────────────────────────────────────
static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const char * name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        DIFFUSE_DIE("tensor not found: %s", name);
    }
    return t;
}

// ── Load model from GGUF ───────────────────────────────────────
diffuse_model * diffuse_model_load_impl(const std::string & path, int n_threads) {
    DIFFUSE_LOG("loading model from %s", path.c_str());

    // Open GGUF file — gguf_init_from_file allocates a ggml_context with tensors
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gparams = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &meta_ctx,
    };
    struct gguf_context * gctx = gguf_init_from_file(path.c_str(), gparams);
    if (!gctx) {
        DIFFUSE_DIE("failed to open GGUF file: %s", path.c_str());
    }

    // Parse hyperparameters — supports diffuse.*, qwen2.*, llama.* prefixes
    auto * model = new diffuse_model();
    auto & hp = model->hparams;

    // Detect architecture
    std::string arch = detect_arch_prefix(gctx);
    DIFFUSE_LOG("  detected arch prefix: %s", arch.c_str());

    hp.n_layer       = get_u32_multi(gctx, "block_count");
    hp.n_head        = get_u32_multi(gctx, "attention.head_count");
    hp.n_head_kv     = get_u32_multi(gctx, "attention.head_count_kv");
    hp.n_embd        = get_u32_multi(gctx, "embedding_length");
    hp.n_ff          = get_u32_multi(gctx, "feed_forward_length");
    hp.n_ctx_max     = get_u32_multi(gctx, "context_length", 4096);
    hp.n_vocab       = get_u32_multi(gctx, "vocab_size", 0, false);  // inferred from tensor if absent
    hp.mask_token_id = get_u32_multi(gctx, "mask_token_id", 0, false);  // 0 = no mask token (AR models)
    hp.rope_theta    = get_f32_multi(gctx, "rope.freq_base", 1000000.0f);
    hp.rms_norm_eps  = get_f32_multi(gctx, "attention.layer_norm_rms_epsilon", 1e-6f);

    // Read model type
    {
        int64_t id = gguf_find_key(gctx, "diffuse.model_type");
        if (id >= 0) {
            model->model_type = gguf_get_val_str(gctx, id);
        } else if (arch == "qwen2") {
            model->model_type = "qwen2";  // autoregressive Qwen2.5
        } else if (arch == "llama") {
            model->model_type = "llama";  // autoregressive Llama
        } else {
            model->model_type = "llada";  // default for older diffuse GGUF files
        }
    }

    const char * arch_name = "unknown";
    if (model->model_type == "llada")  arch_name = "LLaDA (Llama backbone, diffusion)";
    else if (model->model_type == "dream")  arch_name = "Dream (Qwen2.5 backbone, diffusion)";
    else if (model->model_type == "llada2_moe")  arch_name = "LLaDA2 MoE (diffusion)";
    else if (model->model_type == "qwen2")  arch_name = "Qwen2.5 (autoregressive)";
    else if (model->model_type == "llama")  arch_name = "Llama (autoregressive)";

    DIFFUSE_LOG("  arch: %s", arch_name);
    DIFFUSE_LOG("  n_vocab=%u, n_embd=%u, n_head=%u/%u(kv), n_layer=%u, n_ff=%u",
                hp.n_vocab, hp.n_embd, hp.n_head, hp.n_head_kv, hp.n_layer, hp.n_ff);
    DIFFUSE_LOG("  rope_theta=%.0f, rms_norm_eps=%.1e, mask_token=%u",
                hp.rope_theta, hp.rms_norm_eps, hp.mask_token_id);

    // The weight context is the one that gguf_init populated
    model->ctx = meta_ctx;

    // Map tensors to model struct
    model->tok_embd    = get_tensor(meta_ctx, "token_embd.weight");
    model->output_norm = get_tensor(meta_ctx, "output_norm.weight");

    // Infer n_vocab from token_embd if not in metadata
    // In GGML: token_embd.weight has shape [n_embd, n_vocab] (ne[0]=n_embd, ne[1]=n_vocab)
    if (hp.n_vocab == 0) {
        hp.n_vocab = (uint32_t)model->tok_embd->ne[1];
        DIFFUSE_LOG("  n_vocab inferred from token_embd: %u", hp.n_vocab);
    }

    // lm_head — may or may not be present (tied embeddings)
    struct ggml_tensor * lm_head = ggml_get_tensor(meta_ctx, "output.weight");
    model->output = lm_head ? lm_head : model->tok_embd;

    // Layers
    model->layers.resize(hp.n_layer);
    // Skip standard layer loading for MoE models (loaded below)
    if (model->model_type != "llada2_moe") {
    for (uint32_t i = 0; i < hp.n_layer; i++) {
        auto & l = model->layers[i];
        l.attn_norm = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_norm.weight", i).c_str());
        l.wq       = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_q.weight", i).c_str());
        l.wk       = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_k.weight", i).c_str());
        l.wv       = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_v.weight", i).c_str());
        l.wo       = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_output.weight", i).c_str());
        l.ffn_norm = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_norm.weight", i).c_str());
        l.ffn_gate = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_gate.weight", i).c_str());
        l.ffn_up   = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_up.weight", i).c_str());
        l.ffn_down = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_down.weight", i).c_str());

        // Optional QKV biases (Dream/Qwen2.5 has them, LLaDA does not)
        l.bq = ggml_get_tensor(meta_ctx, fmt_layer("blk.%d.attn_q.bias", i).c_str());
        l.bk = ggml_get_tensor(meta_ctx, fmt_layer("blk.%d.attn_k.bias", i).c_str());
        l.bv = ggml_get_tensor(meta_ctx, fmt_layer("blk.%d.attn_v.bias", i).c_str());
    }  // close for loop
    }  // close if (not llada2_moe)

    // ── Load LLaDA2 MoE layers (if model is llada2_moe) ──────────
    if (model->model_type == "llada2_moe") {
        // Parse MoE hyperparameters from metadata
        model->n_experts         = get_u32_multi(gctx, "expert_count", 0, false);
        model->n_experts_per_tok = get_u32_multi(gctx, "expert_used_count", 0, false);
        model->n_shared_experts  = get_u32_multi(gctx, "expert_shared_count", 0, false);
        model->moe_intermediate  = get_u32_multi(gctx, "expert_feed_forward_length", 0, false);
        model->first_k_dense     = get_u32_multi(gctx, "first_k_dense_replace", 0, false);
        model->head_dim          = get_u32_multi(gctx, "head_dim", hp.n_embd / hp.n_head, false);
        model->rotary_dim        = get_u32_multi(gctx, "rotary_dim", model->head_dim, false);
        model->use_qk_norm       = get_u32_multi(gctx, "use_qk_norm", 0, false);
        model->routed_scaling    = get_f32_multi(gctx, "routed_scaling_factor", 1.0f);
        model->norm_topk_prob    = get_u32_multi(gctx, "norm_topk_prob", 1, false);
        model->moe_block_size    = get_u32_multi(gctx, "moe_block_size", 32, false);
        model->expert_capacity   = get_u32_multi(gctx, "expert_capacity", 48, false);
        model->delete_token_id   = get_u32_multi(gctx, "delete_token_id", 0, false);
        model->split_token_id    = get_u32_multi(gctx, "split_token_id", 0, false);

        DIFFUSE_LOG("  MoE: %u experts (top-%u), %u shared, moe_ff=%u, first_dense=%u",
                    model->n_experts, model->n_experts_per_tok,
                    model->n_shared_experts, model->moe_intermediate,
                    model->first_k_dense);
        DIFFUSE_LOG("  head_dim=%u, rotary_dim=%u, qk_norm=%s, routed_scale=%.1f",
                    model->head_dim, model->rotary_dim,
                    model->use_qk_norm ? "yes" : "no",
                    model->routed_scaling);

        model->moe_layers.resize(hp.n_layer);
        for (uint32_t i = 0; i < hp.n_layer; i++) {
            auto & ml = model->moe_layers[i];
            ml.is_moe = (i >= model->first_k_dense);

            // Norms
            ml.attn_norm      = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_norm.weight", i).c_str());
            ml.post_attn_norm = get_tensor(meta_ctx, fmt_layer("blk.%d.post_attn_norm.weight", i).c_str());

            // Fused QKV
            ml.qkv = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_qkv.weight", i).c_str());
            // Attention output
            ml.wo = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_output.weight", i).c_str());

            // QK norm (optional)
            if (model->use_qk_norm) {
                ml.q_norm = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_q_norm.weight", i).c_str());
                ml.k_norm = get_tensor(meta_ctx, fmt_layer("blk.%d.attn_k_norm.weight", i).c_str());
            }

            if (!ml.is_moe) {
                // Dense MLP
                ml.ffn_gate = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_gate.weight", i).c_str());
                ml.ffn_up   = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_up.weight", i).c_str());
                ml.ffn_down = get_tensor(meta_ctx, fmt_layer("blk.%d.ffn_down.weight", i).c_str());
            } else {
                // MoE
                ml.gate_weight = get_tensor(meta_ctx, fmt_layer("blk.%d.moe_gate.weight", i).c_str());
                ml.gate_bias   = ggml_get_tensor(meta_ctx, fmt_layer("blk.%d.moe_gate_bias.weight", i).c_str());

                ml.expert_gate = get_tensor(meta_ctx, fmt_layer("blk.%d.moe_experts_gate.weight", i).c_str());
                ml.expert_up   = get_tensor(meta_ctx, fmt_layer("blk.%d.moe_experts_up.weight", i).c_str());
                ml.expert_down = get_tensor(meta_ctx, fmt_layer("blk.%d.moe_experts_down.weight", i).c_str());

                if (model->n_shared_experts > 0) {
                    ml.shared_gate = get_tensor(meta_ctx, fmt_layer("blk.%d.moe_shared_gate.weight", i).c_str());
                    ml.shared_up   = get_tensor(meta_ctx, fmt_layer("blk.%d.moe_shared_up.weight", i).c_str());
                    ml.shared_down = get_tensor(meta_ctx, fmt_layer("blk.%d.moe_shared_down.weight", i).c_str());
                }
            }
        }
    }

    DIFFUSE_LOG("model loaded: %lld tensors",
                (long long)gguf_get_n_tensors(gctx));
    gguf_free(gctx);  // frees metadata, but ggml_context (weights) stays alive

    return model;
}

void diffuse_model_free_impl(diffuse_model * model) {
    if (!model) return;
    if (model->buf)     ggml_backend_buffer_free(model->buf);
    if (model->ctx)     ggml_free(model->ctx);
    if (model->backend) ggml_backend_free(model->backend);
    delete model;
}

// ── Public API wrappers ────────────────────────────────────────
diffuse_model * diffuse_model_load(const std::string & path, int n_threads) {
    return diffuse_model_load_impl(path, n_threads);
}

void diffuse_model_free(diffuse_model * model) {
    diffuse_model_free_impl(model);
}

const diffuse_hparams & diffuse_model_hparams(const diffuse_model * model) {
    return model->hparams;
}

// diffuse-quantize: Convert F16/F32 GGUF to quantized GGUF
// Streaming version — processes one tensor at a time, low memory usage.
//
// Supported: q8_0, q4_k_m, q4_0, q6_k

#include <ggml.h>
#include <gguf.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// ── Tensor classification ─────────────────────────────────────

enum tensor_class {
    TC_NORM,
    TC_EMBED,
    TC_OUTPUT,
    TC_ATTN,
    TC_FFN,
    TC_FFN_EDGE,
    TC_MOE_GATE,     // Router gate weight — keep F32
};

static tensor_class classify_tensor(const char * name, int n_layers) {
    if (strstr(name, "_norm") || strstr(name, "norm.")) return TC_NORM;
    if (strstr(name, ".bias")) return TC_NORM;

    // MoE router gate — small, keep as F32
    if (strstr(name, "moe_gate.weight") || strstr(name, "moe_gate_bias")) return TC_MOE_GATE;

    // Token embeddings
    if (strstr(name, "token_embd") || strstr(name, "embed")) return TC_EMBED;

    int layer = -1;
    const char * blk = strstr(name, "blk.");
    if (blk) layer = atoi(blk + 4);

    // Attention weights
    if (strstr(name, "attn_q") || strstr(name, "attn_k") ||
        strstr(name, "attn_v") || strstr(name, "attn_output")) {
        return TC_ATTN;
    }

    // MoE expert weights (3D stacked: gate/up/down)
    if (strstr(name, "moe_experts_") || strstr(name, "moe_shared_")) {
        return TC_FFN;
    }

    // Dense FFN weights
    if (strstr(name, "ffn_gate") || strstr(name, "ffn_up") || strstr(name, "ffn_down")) {
        if (layer == 0 || layer == n_layers - 1) return TC_FFN_EDGE;
        return TC_FFN;
    }

    // Output head
    if (strstr(name, "output.weight") || strstr(name, "lm_head")) return TC_OUTPUT;

    return TC_ATTN;
}

// ── Quantization scheme ───────────────────────────────────────

struct quant_scheme {
    std::string name;
    ggml_type norm_type, embed_type, output_type, attn_type, ffn_type, ffn_edge_type, moe_gate_type;
};

static quant_scheme get_scheme(const std::string & type_str) {
    if (type_str == "q8_0")
        return {"Q8_0", GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0,
                GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32};
    if (type_str == "q4_k_m" || type_str == "q4_k")
        return {"Q4_K_M", GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q6_K,
                GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_F32};
    if (type_str == "q4_0")
        return {"Q4_0", GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0,
                GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_F32};
    if (type_str == "q6_k")
        return {"Q6_K", GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q6_K,
                GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_F32};
    fprintf(stderr, "Unknown: %s\nSupported: q8_0, q4_k_m, q4_0, q6_k\n", type_str.c_str());
    exit(1);
}

static ggml_type target_type_for(tensor_class tc, const quant_scheme & scheme) {
    switch (tc) {
        case TC_NORM:      return scheme.norm_type;
        case TC_EMBED:     return scheme.embed_type;
        case TC_OUTPUT:    return scheme.output_type;
        case TC_ATTN:      return scheme.attn_type;
        case TC_FFN:       return scheme.ffn_type;
        case TC_FFN_EDGE:  return scheme.ffn_edge_type;
        case TC_MOE_GATE:  return scheme.moe_gate_type;
    }
    return GGML_TYPE_F32;
}

static const char * tc_name(tensor_class tc) {
    switch (tc) {
        case TC_NORM:      return "norm";
        case TC_EMBED:     return "embed";
        case TC_OUTPUT:    return "output";
        case TC_ATTN:      return "attn";
        case TC_FFN:       return "ffn";
        case TC_FFN_EDGE:  return "ffn_edge";
        case TC_MOE_GATE:  return "moe_gate";
    }
    return "?";
}

// ── Main ──────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <type>\n", argv[0]);
        return 1;
    }

    const char * input_path  = argv[1];
    const char * output_path = argv[2];
    const quant_scheme scheme = get_scheme(argv[3]);

    fprintf(stderr, "Quantizing: %s → %s [%s]\n", input_path, output_path, scheme.name.c_str());

    // ── Load input GGUF metadata ──────────────────────────────
    struct ggml_context * src_ctx = nullptr;
    struct gguf_init_params params = { false, &src_ctx };
    struct gguf_context * src_gctx = gguf_init_from_file(input_path, params);
    if (!src_gctx) { fprintf(stderr, "Failed to open: %s\n", input_path); return 1; }

    const int n_tensors = gguf_get_n_tensors(src_gctx);
    int n_layers = 32;
    int64_t key_id = gguf_find_key(src_gctx, "diffuse.block_count");
    if (key_id >= 0) n_layers = (int)gguf_get_val_u32(src_gctx, key_id);
    fprintf(stderr, "Input: %d tensors, %d layers\n", n_tensors, n_layers);

    // ── Pass 1: Build output GGUF with tensor info (no data) ──
    struct gguf_context * dst_gctx = gguf_init_empty();
    gguf_set_kv(dst_gctx, src_gctx);

    struct ggml_init_params ti_params = { (size_t)(ggml_tensor_overhead() * (n_tensors + 128)), nullptr, true };
    struct ggml_context * ti_ctx = ggml_init(ti_params);

    size_t total_src_bytes = 0;
    size_t total_dst_bytes = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(src_gctx, i);
        struct ggml_tensor * src_t = ggml_get_tensor(src_ctx, name);
        if (!src_t) continue;

        tensor_class tc = classify_tensor(name, n_layers);
        ggml_type dst_type = target_type_for(tc, scheme);

        // If target type is same or larger, keep original
        if (dst_type == src_t->type ||
            (!ggml_is_quantized(dst_type) && dst_type >= src_t->type)) {
            dst_type = src_t->type;
        }

        struct ggml_tensor * ti = ggml_new_tensor(ti_ctx, dst_type, ggml_n_dims(src_t), src_t->ne);
        ggml_set_name(ti, name);
        gguf_add_tensor(dst_gctx, ti);

        size_t src_nb = ggml_nbytes(src_t);
        size_t dst_nb;
        if (ggml_is_quantized(dst_type) && dst_type != src_t->type) {
            int64_t nelem = ggml_nelements(src_t);
            int64_t npr = src_t->ne[0];
            int64_t nr = nelem / npr;
            dst_nb = ggml_row_size(dst_type, npr) * nr;
        } else {
            dst_nb = src_nb;
        }
        total_src_bytes += src_nb;
        total_dst_bytes += dst_nb;
    }

    // ── Write header + KV + tensor info ───────────────────────
    fprintf(stderr, "Writing header...\n");
    // Write metadata + tensor info first, then tensor data
    // Use the 3-step approach: write only meta, then append data
    gguf_write_to_file(dst_gctx, output_path, true);  // only_meta = true

    // Now reopen and append tensor data
    FILE * fout = fopen(output_path, "ab");
    if (!fout) { fprintf(stderr, "Failed to reopen %s\n", output_path); return 1; }

    // ── Pass 2: Process and write each tensor ─────────────────
    fprintf(stderr, "Processing tensors...\n");
    ggml_quantize_init(GGML_TYPE_Q4_K);  // init all quantization tables

    int n_quantized = 0, n_kept = 0;
    const size_t alignment = 32;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(src_gctx, i);
        struct ggml_tensor * src_t = ggml_get_tensor(src_ctx, name);
        if (!src_t) continue;

        const ggml_type src_type = src_t->type;
        const int64_t nelements = ggml_nelements(src_t);
        tensor_class tc = classify_tensor(name, n_layers);
        ggml_type dst_type = target_type_for(tc, scheme);

        if (dst_type == src_type || (!ggml_is_quantized(dst_type) && dst_type >= src_type)) {
            dst_type = src_type;
        }

        // Write padding to alignment
        long cur_pos = ftell(fout);
        size_t pad = (alignment - (cur_pos % alignment)) % alignment;
        if (pad > 0) {
            std::vector<uint8_t> zeros(pad, 0);
            fwrite(zeros.data(), 1, pad, fout);
        }

        if (dst_type == src_type) {
            // Copy as-is
            fwrite(src_t->data, 1, ggml_nbytes(src_t), fout);
            n_kept++;
            if (i % 20 == 0)
                fprintf(stderr, "  [%d/%d] %-45s keep %s\n", i+1, n_tensors, name, ggml_type_name(src_type));
        } else {
            // Dequantize to F32, then quantize to target
            std::vector<float> f32_data(nelements);

            if (src_type == GGML_TYPE_F32) {
                memcpy(f32_data.data(), src_t->data, ggml_nbytes(src_t));
            } else if (src_type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *)src_t->data, f32_data.data(), nelements);
            } else {
                // Unsupported source type — keep as-is
                fwrite(src_t->data, 1, ggml_nbytes(src_t), fout);
                n_kept++;
                continue;
            }

            // Quantize
            int64_t n_per_row = src_t->ne[0];
            int64_t nrows = nelements / n_per_row;

            int64_t blk_size = ggml_blck_size(dst_type);
            if (n_per_row % blk_size != 0) {
                // Can't quantize — keep original
                fwrite(src_t->data, 1, ggml_nbytes(src_t), fout);
                n_kept++;
                continue;
            }

            size_t dst_nbytes = ggml_row_size(dst_type, n_per_row) * nrows;
            std::vector<uint8_t> dst_data(dst_nbytes);

            size_t actual = ggml_quantize_chunk(dst_type, f32_data.data(),
                dst_data.data(), 0, nrows, n_per_row, nullptr);

            fwrite(dst_data.data(), 1, actual, fout);
            n_quantized++;

            if (i % 20 == 0) {
                float ratio = (float)ggml_nbytes(src_t) / actual;
                fprintf(stderr, "  [%d/%d] %-45s %s→%s %.1fx\n", i+1, n_tensors, name,
                    ggml_type_name(src_type), ggml_type_name(dst_type), ratio);
            }
        }
    }

    fclose(fout);

    float compression = (float)total_src_bytes / total_dst_bytes;
    fprintf(stderr, "\nDone! %d quantized, %d kept\n", n_quantized, n_kept);
    fprintf(stderr, "  %.1f GB → %.1f GB (%.1fx)\n",
            total_src_bytes/1e9, total_dst_bytes/1e9, compression);

    ggml_quantize_free();
    gguf_free(dst_gctx);
    gguf_free(src_gctx);
    ggml_free(src_ctx);
    ggml_free(ti_ctx);

    return 0;
}

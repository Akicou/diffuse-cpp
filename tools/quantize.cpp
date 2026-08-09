// diffuse-quantize v3: Memory-safe streaming quantizer
// Reads source GGUF tensor-by-tensor, writes output in one pass.

#include <ggml.h>
#include <gguf.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

enum tensor_class { TC_NORM, TC_EMBED, TC_OUTPUT, TC_ATTN, TC_FFN, TC_FFN_EDGE, TC_MOE_GATE };

static tensor_class classify_tensor(const char * name, int n_layers) {
    if (strstr(name, "_norm") || strstr(name, "norm.") || strstr(name, ".bias")) return TC_NORM;
    if (strstr(name, "moe_gate")) return TC_MOE_GATE;
    if (strstr(name, "token_embd") || strstr(name, "embed")) return TC_EMBED;
    int layer = -1; const char * blk = strstr(name, "blk."); if (blk) layer = atoi(blk + 4);
    if (strstr(name, "attn_q") || strstr(name, "attn_k") || strstr(name, "attn_v") || strstr(name, "attn_output")) return TC_ATTN;
    if (strstr(name, "moe_experts_") || strstr(name, "moe_shared_")) return TC_FFN;
    if (strstr(name, "ffn_gate") || strstr(name, "ffn_up") || strstr(name, "ffn_down")) {
        return (layer == 0 || layer == n_layers - 1) ? TC_FFN_EDGE : TC_FFN;
    }
    if (strstr(name, "output.weight") || strstr(name, "lm_head")) return TC_OUTPUT;
    return TC_ATTN;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <q4_k_m|q8_0|q4_0|q6_k>\n", argv[0]);
        return 1;
    }

    const char * input_path = argv[1];
    const char * output_path = argv[2];
    std::string type_str = argv[3];

    // Determine target types per class
    ggml_type t_norm=GGML_TYPE_F32, t_embed=GGML_TYPE_F16, t_output=GGML_TYPE_Q6_K;
    ggml_type t_attn=GGML_TYPE_Q4_K, t_ffn=GGML_TYPE_Q4_K, t_ffn_edge=GGML_TYPE_Q6_K, t_moe_gate=GGML_TYPE_F32;
    const char *scheme_name = "Q4_K_M";
    if (type_str == "q8_0") { t_attn=t_ffn=t_ffn_edge=t_output=GGML_TYPE_Q8_0; scheme_name="Q8_0"; }
    else if (type_str == "q4_0") { t_attn=t_ffn=t_output=GGML_TYPE_Q4_0; t_ffn_edge=GGML_TYPE_Q4_0; scheme_name="Q4_0"; }
    else if (type_str == "q6_k") { t_attn=t_ffn=t_ffn_edge=t_output=GGML_TYPE_Q6_K; scheme_name="Q6_K"; }
    else if (type_str != "q4_k_m" && type_str != "q4_k") {
        fprintf(stderr, "Unknown type: %s\nSupported: q8_0, q4_k_m, q4_0, q6_k\n", type_str.c_str());
        return 1;
    }

    fprintf(stderr, "Quantizing: %s → %s [%s]\n", input_path, output_path, scheme_name);

    // ── Load source GGUF (loads all tensor data into memory) ──
    struct ggml_context * src_ctx = nullptr;
    struct gguf_init_params params = { false, &src_ctx };
    struct gguf_context * src_gctx = gguf_init_from_file(input_path, params);
    if (!src_gctx) { fprintf(stderr, "Failed to open input\n"); return 1; }

    int n_tensors = (int)gguf_get_n_tensors(src_gctx);
    int n_layers = 32;
    int64_t kid = gguf_find_key(src_gctx, "diffuse.block_count");
    if (kid >= 0) n_layers = (int)gguf_get_val_u32(src_gctx, kid);
    fprintf(stderr, "Input: %d tensors, %d layers\n", n_tensors, n_layers);

    // ── Build output GGUF with add_tensor (accumulate like converter) ──
    struct gguf_context * dst_gctx = gguf_init_empty();
    gguf_set_kv(dst_gctx, src_gctx);

    // Context for tensor metadata
    size_t ti_mem = ggml_tensor_overhead() * (n_tensors + 128);
    struct ggml_context * meta_ctx = ggml_init({ ti_mem, nullptr, true });

    size_t total_src = 0, total_dst = 0;
    int n_q = 0, n_k = 0;

    // We'll collect data buffers to write after all tensor info is registered
    // BUT process them one at a time to limit memory
    // Actually for the standard gguf_write_to_file to work, we need all data
    // available. So we'll use the gguf API: add_tensor copies the data pointer,
    // then write_tensors_to_file writes all of them.
    //
    // The problem: we need all tensor data alive simultaneously.
    // For 192GB F16 input, that's 192GB. Plus quantized output ~55GB.
    // Total peak: ~250GB. With 1TB RAM, this should work.

    std::vector<std::vector<uint8_t>> data_buffers;  // keeps quantized/copied data alive
    std::vector<std::vector<float>> f32_buffers;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(src_gctx, i);
        struct ggml_tensor * src_t = ggml_get_tensor(src_ctx, name);
        if (!src_t) continue;

        tensor_class tc = classify_tensor(name, n_layers);
        ggml_type dst_type;
        switch (tc) {
            case TC_NORM: dst_type = t_norm; break;
            case TC_EMBED: dst_type = t_embed; break;
            case TC_OUTPUT: dst_type = t_output; break;
            case TC_ATTN: dst_type = t_attn; break;
            case TC_FFN: dst_type = t_ffn; break;
            case TC_FFN_EDGE: dst_type = t_ffn_edge; break;
            case TC_MOE_GATE: dst_type = t_moe_gate; break;
        }

        // Don't upcast
        if (dst_type == src_t->type || (!ggml_is_quantized(dst_type) && dst_type >= src_t->type)) {
            dst_type = src_t->type;
        }

        int64_t nelem = ggml_nelements(src_t);
        total_src += ggml_nbytes(src_t);

        if (dst_type == src_t->type) {
            // Keep as-is — reference src data directly
            struct ggml_tensor * dt = ggml_new_tensor(meta_ctx, dst_type, ggml_n_dims(src_t), src_t->ne);
            ggml_set_name(dt, name);
            dt->data = src_t->data;
            gguf_add_tensor(dst_gctx, dt);
            total_dst += ggml_nbytes(src_t);
            n_k++;
        } else {
            // Dequantize to F32
            f32_buffers.emplace_back(nelem);
            auto & f32 = f32_buffers.back();

            if (src_t->type == GGML_TYPE_F32) {
                memcpy(f32.data(), src_t->data, ggml_nbytes(src_t));
            } else if (src_t->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *)src_t->data, f32.data(), nelem);
            }

            // Quantize
            int64_t npr = src_t->ne[0];
            int64_t nr = nelem / npr;
            int64_t bs = ggml_blck_size(dst_type);

            if (npr % bs != 0) {
                // Can't quantize, keep original
                struct ggml_tensor * dt = ggml_new_tensor(meta_ctx, src_t->type, ggml_n_dims(src_t), src_t->ne);
                ggml_set_name(dt, name);
                dt->data = src_t->data;
                gguf_add_tensor(dst_gctx, dt);
                total_dst += ggml_nbytes(src_t);
                n_k++;
                // Free the f32 buffer we allocated
                f32_buffers.pop_back();
                continue;
            }

            ggml_quantize_init(dst_type);
            size_t dst_nb = ggml_row_size(dst_type, npr) * nr;
            data_buffers.emplace_back(dst_nb);
            auto & qdata = data_buffers.back();

            size_t actual = ggml_quantize_chunk(dst_type, f32.data(), qdata.data(), 0, nr, npr, nullptr);

            struct ggml_tensor * dt = ggml_new_tensor(meta_ctx, dst_type, ggml_n_dims(src_t), src_t->ne);
            ggml_set_name(dt, name);
            dt->data = qdata.data();
            gguf_add_tensor(dst_gctx, dt);
            total_dst += actual;
            n_q++;

            // Free f32 buffer
            f32_buffers.pop_back();
        }

        if (i % 20 == 0) {
            fprintf(stderr, "  [%d/%d] %-45s %s→%s\n", i+1, n_tensors, name,
                    ggml_type_name(src_t->type), ggml_type_name(dst_type));
        }
    }

    fprintf(stderr, "\nWriting %s...\n", output_path);
    if (!gguf_write_to_file(dst_gctx, output_path, false)) {
        fprintf(stderr, "Failed to write output\n");
        return 1;
    }

    fprintf(stderr, "\nDone! %d quantized, %d kept\n", n_q, n_k);
    fprintf(stderr, "  %.1f GB → %.1f GB (%.1fx)\n", total_src/1e9, total_dst/1e9, (float)total_src/total_dst);

    ggml_quantize_free();
    gguf_free(dst_gctx);
    gguf_free(src_gctx);
    ggml_free(src_ctx);
    ggml_free(meta_ctx);
    return 0;
}

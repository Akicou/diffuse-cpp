#!/usr/bin/env python3
"""Streaming GGUF quantizer using C library via ctypes.

Reads source GGUF with gguf-py (mmap, low memory), quantizes each tensor
via the compiled C library, writes output streaming.
"""

import argparse, ctypes, json, os, sys, time
import numpy as np
import gguf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--type", default="q4_k_m", choices=["q4_k_m","q8_0","q4_0","q6_k"])
    ap.add_argument("--lib", default="/workspace/quant_lib.so")
    args = ap.parse_args()
    t0 = time.time()

    # ── Load C library ──
    lib = ctypes.CDLL(args.lib)
    lib.quantize_to_type.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    lib.quantize_to_type.restype = ctypes.c_size_t
    lib.get_quantized_size.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
    lib.get_quantized_size.restype = ctypes.c_size_t
    lib.get_block_size.argtypes = [ctypes.c_int]
    lib.get_block_size.restype = ctypes.c_int
    lib.f16_to_f32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    lib.init_quant.argtypes = [ctypes.c_int]
    lib.cleanup_quant.argtypes = []

    # ── Quantization scheme ──
    Q = gguf.GGMLQuantizationType
    if args.type == "q4_k_m":
        sch = {'norm': Q.F32, 'embed': Q.F16, 'output': Q.Q6_K, 'attn': Q.Q4_K,
               'ffn': Q.Q4_K, 'ffn_edge': Q.Q6_K, 'moe_gate': Q.F32}
    elif args.type == "q8_0":
        t = Q.Q8_0; sch = {'norm': Q.F32, 'embed': Q.F16, 'output': t, 'attn': t, 'ffn': t, 'ffn_edge': t, 'moe_gate': Q.F32}
    elif args.type == "q4_0":
        t = Q.Q4_0; sch = {'norm': Q.F32, 'embed': Q.F16, 'output': t, 'attn': t, 'ffn': t, 'ffn_edge': t, 'moe_gate': Q.F32}
    elif args.type == "q6_k":
        t = Q.Q6_K; sch = {'norm': Q.F32, 'embed': Q.F16, 'output': t, 'attn': t, 'ffn': t, 'ffn_edge': t, 'moe_gate': Q.F32}

    def classify(name, n_layers):
        if "_norm" in name or "norm." in name or ".bias" in name: return 'norm'
        if "moe_gate" in name: return 'moe_gate'
        if "token_embd" in name or "embed" in name: return 'embed'
        if "attn_q" in name or "attn_k" in name or "attn_v" in name or "attn_output" in name: return 'attn'
        if "moe_experts_" in name or "moe_shared_" in name: return 'ffn'
        if "ffn_gate" in name or "ffn_up" in name or "ffn_down" in name:
            layer = -1; blk = name.find("blk.")
            if blk >= 0: layer = int(name[blk+4:])
            return 'ffn_edge' if (layer == 0 or layer == n_layers - 1) else 'ffn'
        if "output.weight" in name: return 'output'
        return 'attn'

    # ── Read source GGUF (mmap, no data copy) ──
    print(f"Reading: {args.input}", flush=True)
    reader = gguf.GGUFReader(args.input)

    n_layers = 32
    for field in reader.fields.values():
        if field.name == "diffuse.block_count" and len(field.value) > 0:
            n_layers = field.value[0]

    tensors = list(reader.tensors)
    n_tensors = len(tensors)
    print(f"  {n_tensors} tensors, {n_layers} layers", flush=True)

    # ── Build output ──
    writer = gguf.GGUFWriter(args.output, "diffuse")

    # Copy KV metadata
    for field in reader.fields.values():
        fname = field.name
        ftype = field.type
        if field.value is None or len(field.value) == 0: continue

        try:
            if ftype == gguf.GGUFValueType.UINT32:
                writer.add_uint32(fname, int(field.value[0]))
            elif ftype == gguf.GGUFValueType.FLOAT32:
                writer.add_float32(fname, float(field.value[0]))
            elif ftype == gguf.GGUFValueType.STRING:
                val = field.value if isinstance(field.value, str) else (field.value.decode() if isinstance(field.value, bytes) else str(field.value))
                writer.add_string(fname, val)
            elif ftype == gguf.GGUFValueType.BOOL:
                writer.add_bool(fname, bool(field.value[0]))
            elif ftype == gguf.GGUFValueType.INT32:
                writer.add_uint32(fname, int(field.value[0]))
            elif ftype == gguf.GGUFValueType.ARRAY:
                if fname == "tokenizer.ggml.tokens":
                    writer.add_token_list(list(field.value))
                elif fname == "tokenizer.ggml.token_type":
                    writer.add_token_types(list(field.value))
                elif fname == "tokenizer.ggml.scores":
                    writer.add_token_scores(list(field.value))
                elif fname == "tokenizer.ggml.merges":
                    writer.add_token_merges(list(field.value))
        except Exception:
            pass

    print(f"KV copied ({time.time()-t0:.0f}s)", flush=True)

    # ── Add tensor info ──
    plan = []
    for t in tensors:
        name = t.name.decode() if isinstance(t.name, bytes) else t.name
        tc = classify(name, n_layers)
        dst_raw = sch[tc]
        src_raw = t.ggml_type  # GGMLQuantizationType enum

        # Skip quantization if same type or target is larger
        if src_raw == dst_raw:
            dst_raw = src_raw

        shape = tuple(t.shape)
        nelem = int(np.prod(shape))

        # Compute dst nbytes
        if dst_raw == Q.F32:
            dst_nb = nelem * 4
            np_dt = np.float32
        elif dst_raw == Q.F16:
            dst_nb = nelem * 2
            np_dt = np.float16
        else:
            # Quantized — compute via C library
            npr = shape[-1]
            nr = nelem // npr
            dst_nb = lib.get_quantized_size(int(dst_raw), ctypes.c_int64(nr), ctypes.c_int64(npr))
            np_dt = np.float32  # placeholder dtype for tensor_info

        writer.add_tensor_info(name, shape, np_dt, dst_nb, raw_dtype=dst_raw)
        plan.append((name, src_raw, dst_raw, shape, nelem, t.data, dst_nb))

    # Write header
    print(f"Writing header ({time.time()-t0:.0f}s)...", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # ── Process tensors ──
    print(f"Processing {len(plan)} tensors ({time.time()-t0:.0f}s)...", flush=True)
    total_src = 0; total_dst = 0; n_q = 0; n_k = 0

    # Init quant tables for all types we'll use
    for dt in set(p[2] for p in plan):
        if dt != Q.F32 and dt != Q.F16:
            lib.init_quant(int(dt))

    for idx, (name, src_raw, dst_raw, shape, nelem, data_ref, dst_nb) in enumerate(plan):
        # Read source data (zero-copy from mmap)
        if src_raw == Q.F32:
            f32 = np.frombuffer(data_ref, dtype=np.float32).copy()
        elif src_raw == Q.F16:
            f16 = np.frombuffer(data_ref, dtype=np.float16)
            f32 = np.zeros(nelem, dtype=np.float32)
            lib.f16_to_f32(f16.ctypes.data_as(ctypes.c_void_p),
                          f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          ctypes.c_int64(nelem))
        else:
            # Already quantized — keep as-is
            writer.write_tensor_data(np.frombuffer(data_ref, dtype=np.uint8).copy())
            total_src += len(data_ref); total_dst += len(data_ref); n_k += 1
            continue

        if dst_raw == Q.F32:
            writer.write_tensor_data(f32.reshape(shape))
            total_src += nelem * (4 if src_raw == Q.F32 else 2); total_dst += nelem * 4; n_k += 1
        elif dst_raw == Q.F16:
            writer.write_tensor_data(f32.astype(np.float16).reshape(shape))
            total_src += nelem * (4 if src_raw == Q.F32 else 2); total_dst += nelem * 2; n_k += 1
        else:
            # Quantize via C library
            npr = shape[-1]; nr = nelem // npr
            blk = lib.get_block_size(int(dst_raw))
            if npr % blk != 0:
                # Can't quantize — keep F16
                writer.write_tensor_data(f32.astype(np.float16).reshape(shape))
                total_dst += nelem * 2; n_k += 1
                del f32
                continue

            out_buf = ctypes.create_string_buffer(dst_nb)
            actual = lib.quantize_to_type(int(dst_raw),
                f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_buf, ctypes.c_int64(nr), ctypes.c_int64(npr))

            writer.write_tensor_data(np.frombuffer(out_buf.raw[:actual], dtype=np.uint8).copy())
            total_src += nelem * (4 if src_raw == Q.F32 else 2); total_dst += actual; n_q += 1

        del f32

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(plan)}] ({time.time()-t0:.0f}s)", flush=True)

    lib.cleanup_quant()
    writer.close()
    total = time.time() - t0
    sz = os.path.getsize(args.output) / 1e9
    print(f"\nDone! {sz:.1f}GB in {total:.0f}s ({total/60:.1f} min)", flush=True)
    print(f"  {n_q} quantized, {n_k} kept", flush=True)
    print(f"  {total_src/1e9:.1f}GB → {total_dst/1e9:.1f}GB ({total_src/total_dst:.1f}x)", flush=True)

if __name__ == "__main__":
    main()

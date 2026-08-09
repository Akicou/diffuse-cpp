#!/usr/bin/env python3
"""Streaming GGUF quantizer using C library via ctypes."""

import argparse, ctypes, os, time, struct
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

    lib = ctypes.CDLL(args.lib)
    lib.quantize_to_type.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    lib.quantize_to_type.restype = ctypes.c_size_t
    lib.get_quantized_size.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
    lib.get_quantized_size.restype = ctypes.c_size_t
    lib.get_block_size.argtypes = [ctypes.c_int]
    lib.get_block_size.restype = ctypes.c_int
    lib.f16_to_f32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    lib.init_quant.argtypes = [ctypes.c_int]

    Q = gguf.GGMLQuantizationType
    if args.type == "q4_k_m":
        sch = {'norm': Q.F32, 'embed': Q.F16, 'output': Q.Q6_K, 'attn': Q.Q4_K, 'ffn': Q.Q4_K, 'ffn_edge': Q.Q6_K, 'moe_gate': Q.F32}
    elif args.type == "q8_0":
        t = Q.Q8_0; sch = {'norm': Q.F32, 'embed': Q.F16, 'output': t, 'attn': t, 'ffn': t, 'ffn_edge': t, 'moe_gate': Q.F32}
    elif args.type == "q4_0":
        t = Q.Q4_0; sch = {'norm': Q.F32, 'embed': Q.F16, 'output': t, 'attn': t, 'ffn': t, 'ffn_edge': t, 'moe_gate': Q.F32}
    elif args.type == "q6_k":
        t = Q.Q6_K; sch = {'norm': Q.F32, 'embed': Q.F16, 'output': t, 'attn': t, 'ffn': t, 'ffn_edge': t, 'moe_gate': Q.F32}

    # Map Q enum to int values for C library
    QI = {Q.F32: 0, Q.F16: 1, Q.Q4_0: 2, Q.Q8_0: 8, Q.Q4_K: 12, Q.Q6_K: 14}
    # Actually, let me use the actual enum integer values
    def q_int(q):
        return int(q)

    def classify(name, n_layers):
        if "_norm" in name or "norm." in name or ".bias" in name: return 'norm'
        if "moe_gate" in name: return 'moe_gate'
        if "token_embd" in name or "embed" in name: return 'embed'
        if "attn_q" in name or "attn_k" in name or "attn_v" in name or "attn_output" in name: return 'attn'
        if "moe_experts_" in name or "moe_shared_" in name: return 'ffn'
        if "ffn_gate" in name or "ffn_up" in name or "ffn_down" in name:
            layer = -1; blk = name.find("blk.")
            if blk >= 0:
                try: layer = int(name[blk+4:name.find(".", blk+4)])
                except: pass
            return 'ffn_edge' if (layer == 0 or layer == n_layers - 1) else 'ffn'
        if "output.weight" in name: return 'output'
        return 'attn'

    print(f"Reading: {args.input}", flush=True)
    reader = gguf.GGUFReader(args.input)

    # Get n_layers
    n_layers = 32
    bc = reader.fields.get("diffuse.block_count")
    if bc and len(bc.parts) > 0:
        n_layers = int(bc.parts[-1][0])  # last part is the value

    tensors = list(reader.tensors)
    n_tensors = len(tensors)
    print(f"  {n_tensors} tensors, {n_layers} layers", flush=True)

    writer = gguf.GGUFWriter(args.output, "diffuse")

    # Copy KV metadata
    for fname, field in reader.fields.items():
        if not field.types: continue
        ft = field.types[0]
        try:
            if ft == gguf.GGUFValueType.UINT32:
                writer.add_uint32(fname, int(field.parts[-1][0]))
            elif ft == gguf.GGUFValueType.FLOAT32:
                writer.add_float32(fname, float(field.parts[-1][0]))
            elif ft == gguf.GGUFValueType.STRING:
                # String is in parts as bytes
                raw = bytes(field.parts[-1])
                writer.add_string(fname, raw.decode('utf-8', errors='replace'))
            elif ft == gguf.GGUFValueType.BOOL:
                writer.add_bool(fname, bool(field.parts[-1][0]))
            elif ft == gguf.GGUFValueType.INT32:
                writer.add_uint32(fname, int(field.parts[-1][0]))
            elif ft == gguf.GGUFValueType.ARRAY:
                if fname == "tokenizer.ggml.tokens":
                    vals = [v.decode() if isinstance(v, bytes) else str(v) for v in field.parts[1:]]
                    writer.add_token_list(vals)
                elif fname == "tokenizer.ggml.token_type":
                    writer.add_token_types([int(v) for v in field.parts[1:]])
                elif fname == "tokenizer.ggml.scores":
                    writer.add_token_scores([float(v) for v in field.parts[1:]])
                elif fname == "tokenizer.ggml.merges":
                    vals = [v.decode() if isinstance(v, bytes) else str(v) for v in field.parts[1:]]
                    writer.add_token_merges(vals)
        except Exception:
            pass

    print(f"KV copied ({time.time()-t0:.0f}s)", flush=True)

    # Add tensor info
    plan = []
    for t in tensors:
        name = t.name.decode() if isinstance(t.name, bytes) else t.name
        tc = classify(name, n_layers)
        dst_raw = sch[tc]
        src_raw = t.tensor_type
        if src_raw == dst_raw: dst_raw = src_raw

        shape = tuple(t.shape)
        nelem = int(np.prod(shape))

        if dst_raw == Q.F32:
            dst_nb = nelem * 4; np_dt = np.float32
        elif dst_raw == Q.F16:
            dst_nb = nelem * 2; np_dt = np.float16
        else:
            npr = shape[-1]; nr = nelem // npr
            dst_nb = lib.get_quantized_size(q_int(dst_raw), ctypes.c_int64(nr), ctypes.c_int64(npr))
            np_dt = np.float32

        writer.add_tensor_info(name, shape, np_dt, dst_nb, raw_dtype=dst_raw)
        plan.append((name, src_raw, dst_raw, shape, nelem, t.data, dst_nb))

    print(f"Writing header ({time.time()-t0:.0f}s)...", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Process tensors
    print(f"Processing {len(plan)} tensors ({time.time()-t0:.0f}s)...", flush=True)
    total_src = 0; total_dst = 0; n_q = 0; n_k = 0

    for dt in set(p[2] for p in plan):
        if dt != Q.F32 and dt != Q.F16:
            lib.init_quant(q_int(dt))

    for idx, (name, src_raw, dst_raw, shape, nelem, data_ref, dst_nb) in enumerate(plan):
        if src_raw == dst_raw:
            # Keep as-is
            dt_size = {Q.F32: 4, Q.F16: 2}.get(src_raw, 1)
            if src_raw == Q.F32:
                arr = np.frombuffer(data_ref, dtype=np.float32).reshape(shape)
            elif src_raw == Q.F16:
                arr = np.frombuffer(data_ref, dtype=np.float16).reshape(shape)
            else:
                arr = np.frombuffer(data_ref, dtype=np.uint8)
            writer.write_tensor_data(np.array(arr, copy=True))
            total_src += nelem * dt_size; total_dst += nelem * dt_size; n_k += 1
            del arr
        else:
            # Dequantize to F32
            if src_raw == Q.F16:
                f32 = np.zeros(nelem, dtype=np.float32)
                lib.f16_to_f32(ctypes.c_void_p(data_ref.ctypes.data if hasattr(data_ref, 'ctypes') else data_ref.__array_interface__['data'][0]),
                              f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              ctypes.c_int64(nelem))
            elif src_raw == Q.F32:
                f32 = np.frombuffer(data_ref, dtype=np.float32).astype(np.float32).copy()
            else:
                f32 = None

            if f32 is None:
                writer.write_tensor_data(np.frombuffer(data_ref, dtype=np.uint8, copy=True))
                total_src += len(data_ref); total_dst += len(data_ref); n_k += 1
                continue

            if dst_raw == Q.F32:
                writer.write_tensor_data(f32.reshape(shape))
                total_dst += nelem * 4; n_k += 1
            elif dst_raw == Q.F16:
                writer.write_tensor_data(f32.astype(np.float16).reshape(shape))
                total_dst += nelem * 2; n_k += 1
            else:
                npr = shape[-1]; nr = nelem // npr
                blk = lib.get_block_size(q_int(dst_raw))
                if npr % blk != 0:
                    writer.write_tensor_data(f32.astype(np.float16).reshape(shape))
                    total_dst += nelem * 2; n_k += 1; del f32; continue

                out_buf = ctypes.create_string_buffer(dst_nb)
                actual = lib.quantize_to_type(q_int(dst_raw),
                    f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out_buf, ctypes.c_int64(nr), ctypes.c_int64(npr))

                out_arr = np.frombuffer(out_buf.raw[:actual], dtype=np.uint8).copy()
                writer.write_tensor_data(out_arr)
                total_src += nelem * 2; total_dst += actual; n_q += 1
                del out_arr, out_buf

            del f32

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(plan)}] ({time.time()-t0:.0f}s)", flush=True)

    writer.close()
    total = time.time() - t0
    sz = os.path.getsize(args.output) / 1e9
    print(f"\nDone! {sz:.1f}GB in {total:.0f}s ({total/60:.1f} min)", flush=True)
    print(f"  {n_q} quantized, {n_k} kept", flush=True)

if __name__ == "__main__":
    main()

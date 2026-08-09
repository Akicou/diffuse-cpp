#!/usr/bin/env python3
"""Pure streaming GGUF quantizer — reads via file I/O, no mmap.

Combines the streaming converter pattern with C-library quantization.
Memory: ~2× largest tensor (~4GB peak).
"""

import argparse, ctypes, json, os, struct, time
import numpy as np

# ── GGUF binary reader ─────────────────────────────────────────

class GGUFHeader:
    def __init__(self, path):
        self.f = open(path, 'rb')
        magic = self.f.read(4)
        self.version = struct.unpack('<I', self.f.read(4))[0]
        n_tensors = struct.unpack('<Q', self.f.read(8))[0]
        n_kv = struct.unpack('<Q', self.f.read(8))[0]
        self.n_kv = n_kv
        self.n_tensors = n_tensors

        # Parse KV pairs (we need values for the output)
        self.kv = {}  # key → (type_enum, raw_value)
        for _ in range(n_kv):
            klen = struct.unpack('<Q', self.f.read(8))[0]
            key = self.f.read(klen).decode('utf-8')
            vtype = struct.unpack('<I', self.f.read(4))[0]
            val = self._read_value(vtype)
            self.kv[key] = (vtype, val)

        # Parse tensor info
        self.tensors = []
        for _ in range(n_tensors):
            nlen = struct.unpack('<Q', self.f.read(8))[0]
            name = self.f.read(nlen).decode('utf-8')
            n_dims = struct.unpack('<I', self.f.read(4))[0]
            ne = []
            for _ in range(n_dims):
                ne.append(struct.unpack('<q', self.f.read(8))[0])
            ttype = struct.unpack('<I', self.f.read(4))[0]
            offset = struct.unpack('<Q', self.f.read(8))[0]
            self.tensors.append({'name': name, 'n_dims': n_dims, 'ne': ne, 'type': ttype, 'offset': offset})

        self.data_start = self.f.tell()
        # Align
        align = self.kv.get('general.alignment', (0, 32))[1] or 32
        if self.data_start % align != 0:
            self.data_start += align - (self.data_start % align)

    def _read_value(self, vtype):
        if vtype <= 1: return self.f.read(1)[0]  # u8/i8
        elif vtype <= 3: return struct.unpack('<H', self.f.read(2))[0]  # u16/i16
        elif vtype == 4: return struct.unpack('<I', self.f.read(4))[0]  # u32
        elif vtype == 5: return struct.unpack('<i', self.f.read(4))[0]  # i32
        elif vtype == 6: return struct.unpack('<f', self.f.read(4))[0]  # f32
        elif vtype == 7: return self.f.read(1)[0] != 0  # bool
        elif vtype == 8:  # string
            slen = struct.unpack('<Q', self.f.read(8))[0]
            return self.f.read(slen).decode('utf-8')
        elif vtype == 9:  # array
            arr_type = struct.unpack('<I', self.f.read(4))[0]
            arr_n = struct.unpack('<Q', self.f.read(8))[0]
            if arr_type == 8:  # string array
                return [self._read_value(8) for _ in range(arr_n)]
            else:
                return [self._read_value(arr_type) for _ in range(arr_n)]
        elif vtype == 10: return struct.unpack('<Q', self.f.read(8))[0]  # u64
        elif vtype == 11: return struct.unpack('<q', self.f.read(8))[0]  # i64
        elif vtype == 12: return struct.unpack('<d', self.f.read(8))[0]  # f64
        return None

    def read_tensor_data(self, idx):
        """Read raw bytes of tensor data from file."""
        t = self.tensors[idx]
        nelem = 1
        for d in t['ne']: nelem *= d

        # Compute byte size based on type
        type_sizes = {0: 4, 1: 2}  # F16=8
        elem_sz = type_sizes.get(t['type'], 4)
        nbytes = nelem * elem_sz

        self.f.seek(self.data_start + t['offset'])
        return self.f.read(nbytes), t['ne'], t['type']

    def close(self):
        self.f.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--type", default="q4_k_m", choices=["q4_k_m","q8_0","q4_0","q6_k"])
    ap.add_argument("--lib", default="/workspace/quant_lib.so")
    args = ap.parse_args()
    t0 = time.time()

    # Load C quantization library
    lib = ctypes.CDLL(args.lib)
    lib.quantize_to_type.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    lib.quantize_to_type.restype = ctypes.c_size_t
    lib.get_quantized_size.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
    lib.get_quantized_size.restype = ctypes.c_size_t
    lib.get_block_size.argtypes = [ctypes.c_int]
    lib.init_quant.argtypes = [ctypes.c_int]

    # Quantization types (GGML type enum values)
    F32 = 0; F16 = 1; Q4_0 = 2; Q4_1 = 3; Q5_0 = 6; Q5_1 = 7; Q8_0 = 8; Q2_K = 10
    Q3_K = 11; Q4_K = 12; Q5_K = 13; Q6_K = 14

    if args.type == "q4_k_m":
        sch = {'norm': F32, 'embed': F16, 'output': Q6_K, 'attn': Q4_K, 'ffn': Q4_K, 'ffn_edge': Q6_K, 'moe_gate': F32}
    elif args.type == "q8_0":
        sch = {'norm': F32, 'embed': F16, 'output': Q8_0, 'attn': Q8_0, 'ffn': Q8_0, 'ffn_edge': Q8_0, 'moe_gate': F32}
    elif args.type == "q4_0":
        sch = {'norm': F32, 'embed': F16, 'output': Q4_0, 'attn': Q4_0, 'ffn': Q4_0, 'ffn_edge': Q4_0, 'moe_gate': F32}
    elif args.type == "q6_k":
        sch = {'norm': F32, 'embed': F16, 'output': Q6_K, 'attn': Q6_K, 'ffn': Q6_K, 'ffn_edge': Q6_K, 'moe_gate': F32}

    n_layers = 32

    def classify(name):
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

    # ── Read source GGUF header ──
    print(f"Reading: {args.input}", flush=True)
    hdr = GGUFHeader(args.input)
    n_layers = hdr.kv.get('diffuse.block_count', (0, 32))[1]
    print(f"  {hdr.n_tensors} tensors, {n_layers} layers", flush=True)

    # ── Build output via gguf.GGUFWriter ──
    import gguf
    writer = gguf.GGUFWriter(args.output, "diffuse")

    # Copy KV metadata
    for key, (vtype, val) in hdr.kv.items():
        try:
            if vtype == 4: writer.add_uint32(key, int(val))
            elif vtype == 5: writer.add_uint32(key, int(val))
            elif vtype == 6: writer.add_float32(key, float(val))
            elif vtype == 8: writer.add_string(key, str(val))
            elif vtype == 7: writer.add_bool(key, bool(val))
            elif vtype == 9:  # array
                if key == "tokenizer.ggml.tokens": writer.add_token_list(list(val))
                elif key == "tokenizer.ggml.token_type": writer.add_token_types(list(val))
                elif key == "tokenizer.ggml.scores": writer.add_token_scores(list(val))
                elif key == "tokenizer.ggml.merges": writer.add_token_merges(list(val))
        except Exception:
            pass

    print(f"KV copied ({time.time()-t0:.0f}s)", flush=True)

    # ── Add tensor info ──
    plan = []
    for i in range(hdr.n_tensors):
        t = hdr.tensors[i]
        name = t['name']
        src_type = t['type']
        tc = classify(name)
        dst_type = sch[tc]
        if dst_type >= src_type and not (dst_type >= 10):  # don't upcast non-quantized
            dst_type = src_type

        ne = t['ne']
        shape = tuple(reversed(ne))  # GGML stores ne reversed from numpy
        nelem = 1
        for d in ne: nelem *= d

        if dst_type == F32:
            dst_nb = nelem * 4; np_dt = np.float32
        elif dst_type == F16:
            dst_nb = nelem * 2; np_dt = np.float16
        else:
            npr = ne[0]; nr = nelem // npr
            dst_nb = lib.get_quantized_size(dst_type, ctypes.c_int64(nr), ctypes.c_int64(npr))
            np_dt = np.float32

        raw_dt_map = {F32: gguf.GGMLQuantizationType.F32, F16: gguf.GGMLQuantizationType.F16,
                      Q4_K: gguf.GGMLQuantizationType.Q4_K, Q6_K: gguf.GGMLQuantizationType.Q6_K,
                      Q8_0: gguf.GGMLQuantizationType.Q8_0, Q4_0: gguf.GGMLQuantizationType.Q4_0}
        writer.add_tensor_info(name, shape, np_dt, dst_nb, raw_dtype=raw_dt_map.get(dst_type, gguf.GGMLQuantizationType.F16))
        plan.append((name, src_type, dst_type, ne, nelem, i, dst_nb))

    print(f"Writing header ({time.time()-t0:.0f}s)...", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Init quantization tables
    for dt in set(p[2] for p in plan):
        if dt >= 10: lib.init_quant(dt)

    # ── Process tensors ──
    print(f"Processing {len(plan)} tensors ({time.time()-t0:.0f}s)...", flush=True)
    total_src = 0; total_dst = 0; n_q = 0; n_k = 0

    for idx, (name, src_type, dst_type, ne, nelem, ti_idx, dst_nb) in enumerate(plan):
        # Read raw data from file (not mmap!)
        raw_data, raw_ne, raw_type = hdr.read_tensor_data(ti_idx)

        if src_type == dst_type:
            # Keep as-is
            if src_type == F32:
                arr = np.frombuffer(raw_data, dtype=np.float32)
            elif src_type == F16:
                arr = np.frombuffer(raw_data, dtype=np.float16)
            else:
                arr = np.frombuffer(raw_data, dtype=np.uint8)
            writer.write_tensor_data(np.ascontiguousarray(arr))
            total_src += len(raw_data); total_dst += len(raw_data); n_k += 1
            del raw_data, arr
        else:
            # Dequantize
            if src_type == F16:
                f16 = np.frombuffer(raw_data, dtype=np.float16)
                # Chunk the conversion to avoid memory spikes
                CHUNK = 4 * 1024 * 1024  # 4M elements
                f32 = np.empty(nelem, dtype=np.float32)
                for off in range(0, nelem, CHUNK):
                    n = min(CHUNK, nelem - off)
                    f32[off:off+n] = f16[off:off+n].astype(np.float32)
                del f16
            elif src_type == F32:
                f32 = np.frombuffer(raw_data, dtype=np.float32).copy()
            else:
                f32 = None

            del raw_data

            if f32 is None:
                continue

            if dst_type == F32:
                writer.write_tensor_data(f32)
                total_dst += nelem * 4; n_k += 1
            elif dst_type == F16:
                writer.write_tensor_data(f32.astype(np.float16))
                total_dst += nelem * 2; n_k += 1
            else:
                # Quantize via C library
                npr = ne[0]; nr = nelem // npr
                blk = lib.get_block_size(dst_type)
                if npr % blk != 0:
                    writer.write_tensor_data(f32.astype(np.float16))
                    total_dst += nelem * 2; n_k += 1; del f32; continue

                out_buf = ctypes.create_string_buffer(dst_nb)
                actual = lib.quantize_to_type(dst_type,
                    f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out_buf, ctypes.c_int64(nr), ctypes.c_int64(npr))

                out_arr = np.frombuffer(out_buf.raw[:actual], dtype=np.uint8).copy()
                writer.write_tensor_data(out_arr)
                total_src += nelem * 2; total_dst += actual; n_q += 1
                del out_arr, out_buf

            del f32

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(plan)}] ({time.time()-t0:.0f}s)", flush=True)

    hdr.close()
    writer.close()
    total = time.time() - t0
    sz = os.path.getsize(args.output) / 1e9
    print(f"\nDone! {sz:.1f}GB in {total:.0f}s ({total/60:.1f} min)", flush=True)
    print(f"  {n_q} quantized, {n_k} kept", flush=True)

if __name__ == "__main__":
    main()

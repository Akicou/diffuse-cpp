#!/usr/bin/env python3
"""LLaDA2.2 → GGUF — two-pass streaming. Writes one tensor at a time."""

import argparse, json, os, time, struct
import numpy as np
import gguf

class ShardReader:
    def __init__(self, model_dir, index):
        self.shard_info = {}; self.tensor_shard = {}; self._handles = {}
        for tn, sf in index['weight_map'].items():
            self.tensor_shard[tn] = sf
            if sf not in self.shard_info:
                path = os.path.join(model_dir, sf)
                with open(path, 'rb') as f:
                    hl = struct.unpack('<Q', f.read(8))[0]
                    hdr = json.loads(f.read(hl))
                self.shard_info[sf] = (path, hdr, 8 + hl)
    def _h(self, sf):
        if sf not in self._handles: self._handles[sf] = open(self.shard_info[sf][0], 'rb')
        return self._handles[sf]
    def read_f16(self, name):
        sf = self.tensor_shard[name]; _, hdr, ds = self.shard_info[sf]; info = hdr[name]
        f = self._h(sf); f.seek(ds + info['data_offsets'][0])
        total_bytes = info['data_offsets'][1] - info['data_offsets'][0]
        dt = info.get('dtype','BF16')
        if dt == 'F32':
            raw = f.read(total_bytes)
            return np.frombuffer(raw, dtype=np.float32).astype(np.float16).reshape(info['shape'])
        # BF16 → F16 in chunks to avoid large intermediates
        total_elems = total_bytes // 2
        out = np.empty(total_elems, dtype=np.float16)
        CHUNK = 4 * 1024 * 1024  # 4M elements = 8MB bf16 per chunk
        for off in range(0, total_elems, CHUNK):
            n = min(CHUNK, total_elems - off)
            raw = f.read(n * 2)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            out[off:off+n] = (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
        return out.reshape(info['shape'])
    def read_f32(self, name):
        sf = self.tensor_shard[name]; _, hdr, ds = self.shard_info[sf]; info = hdr[name]
        f = self._h(sf); f.seek(ds + info['data_offsets'][0])
        total_bytes = info['data_offsets'][1] - info['data_offsets'][0]
        dt = info.get('dtype','BF16')
        if dt == 'F32':
            raw = f.read(total_bytes)
            return np.frombuffer(raw, dtype=np.float32).reshape(info['shape'])
        total_elems = total_bytes // 2
        out = np.empty(total_elems, dtype=np.float32)
        CHUNK = 4 * 1024 * 1024
        for off in range(0, total_elems, CHUNK):
            n = min(CHUNK, total_elems - off)
            raw = f.read(n * 2)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            out[off:off+n] = (u16.astype(np.uint32) << 16).view(np.float32)
        return out.reshape(info['shape'])
    def get_shape(self, name):
        return self.shard_info[self.tensor_shard[name]][1][name]['shape']
    def close(self):
        for f in self._handles.values(): f.close()
        self._handles.clear()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--type", default="f16", choices=["f16","f32"])
    args = ap.parse_args(); model_dir = args.input; t0 = time.time()

    with open(os.path.join(model_dir, "config.json")) as f: config = json.load(f)
    c = lambda k, d=0: config.get(k, d)
    n_layer=c("num_hidden_layers"); n_embd=c("hidden_size"); n_head=c("num_attention_heads")
    n_head_kv=c("num_key_value_heads"); head_dim=c("head_dim",n_embd//n_head)
    n_ff=c("intermediate_size"); moe_ff=c("moe_intermediate_size")
    n_experts=c("num_experts"); n_ept=c("num_experts_per_tok"); n_shared=c("num_shared_experts")
    first_k=c("first_k_dense_replace"); n_vocab=c("vocab_size"); mask_id=156895
    rope_theta=c("rope_theta",3e6); rms_eps=c("rms_norm_eps",1e-6); rotary_dim=c("rotary_dim",head_dim)
    use_qk=c("use_qk_norm"); routed_sf=c("routed_scaling_factor",1.0); block_size=c("block_size",32)
    tie_emb=c("tie_word_embeddings"); n_group=c("n_group",8); topk_group=c("topk_group",4)
    expert_cap=c("expert_capacity",48)
    pad_id=c("pad_token_id",156892)
    print(f"LLaDA2.2: {n_layer}L {n_experts}E", flush=True)

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f: index = json.load(f)
    reader = ShardReader(model_dir, index)
    out_type = gguf.GGMLQuantizationType.F16 if args.type=="f16" else gguf.GGMLQuantizationType.F32

    # ── Build tensor list: (gguf_name, hf_name, output_dtype) ──
    tensors = []  # list of (gguf_name, hf_name_or_callable, is_f32)

    tensors.append(("token_embd.weight", "model.word_embeddings.weight", False))
    tensors.append(("output_norm.weight", "model.norm.weight", True))
    if not tie_emb: tensors.append(("output.weight", "lm_head.weight", False))

    for i in range(n_layer):
        p = f"model.layers.{i}."; a = p + "attention."
        tensors.append((f"blk.{i}.attn_norm.weight", f"{p}input_layernorm.weight", True))
        tensors.append((f"blk.{i}.post_attn_norm.weight", f"{p}post_attention_layernorm.weight", True))
        tensors.append((f"blk.{i}.attn_qkv.weight", f"{a}query_key_value.weight", False))
        tensors.append((f"blk.{i}.attn_output.weight", f"{a}dense.weight", False))
        if use_qk:
            tensors.append((f"blk.{i}.attn_q_norm.weight", f"{a}query_layernorm.weight", True))
            tensors.append((f"blk.{i}.attn_k_norm.weight", f"{a}key_layernorm.weight", True))
        if i < first_k:
            tensors.append((f"blk.{i}.ffn_gate.weight", f"{p}mlp.gate_proj.weight", False))
            tensors.append((f"blk.{i}.ffn_up.weight", f"{p}mlp.up_proj.weight", False))
            tensors.append((f"blk.{i}.ffn_down.weight", f"{p}mlp.down_proj.weight", False))
        else:
            tensors.append((f"blk.{i}.moe_gate.weight", f"{p}mlp.gate.weight", True))
            if f"{p}mlp.gate.expert_bias" in reader.tensor_shard:
                tensors.append((f"blk.{i}.moe_gate_bias.weight", f"{p}mlp.gate.expert_bias", True))
            # Expert stacks — mark with special prefix
            tensors.append((f"EXPERT_GATE_{i}", f"{p}mlp.experts", False))
            tensors.append((f"EXPERT_UP_{i}", f"{p}mlp.experts", False))
            tensors.append((f"EXPERT_DOWN_{i}", f"{p}mlp.experts", False))
            if n_shared > 0:
                sp = f"{p}mlp.shared_experts."
                tensors.append((f"blk.{i}.moe_shared_gate.weight", f"{sp}gate_proj.weight", False))
                tensors.append((f"blk.{i}.moe_shared_up.weight", f"{sp}up_proj.weight", False))
                tensors.append((f"blk.{i}.moe_shared_down.weight", f"{sp}down_proj.weight", False))

    # ── Pass 1: Add all tensor info (metadata only, no data) ──
    print(f"Pass 1: tensor info ({time.time()-t0:.0f}s)...", flush=True)
    writer = gguf.GGUFWriter(args.output, "diffuse")

    for k,v in [("diffuse.block_count",n_layer),("diffuse.embedding_length",n_embd),
        ("diffuse.attention.head_count",n_head),("diffuse.attention.head_count_kv",n_head_kv),
        ("diffuse.feed_forward_length",n_ff if n_ff else moe_ff),
        ("diffuse.context_length",c("max_position_embeddings",131072)),
        ("diffuse.vocab_size",n_vocab),("diffuse.mask_token_id",mask_id),
        ("diffuse.head_dim",head_dim),("diffuse.rotary_dim",rotary_dim),
        ("diffuse.use_qk_norm",1 if use_qk else 0),
        ("diffuse.expert_count",n_experts),("diffuse.expert_used_count",n_ept),
        ("diffuse.expert_shared_count",n_shared),("diffuse.expert_feed_forward_length",moe_ff),
        ("diffuse.first_k_dense_replace",first_k),("diffuse.norm_topk_prob",1),
        ("diffuse.block_length",block_size),("diffuse.moe_block_size",block_size),
        ("diffuse.n_group",n_group),("diffuse.topk_group",topk_group),
        ("diffuse.expert_capacity",expert_cap),
        ("diffuse.eos_token_id",pad_id),("diffuse.delete_token_id",156930),
        ("diffuse.split_token_id",156931)]:
        writer.add_uint32(k,v)
    for k,v in [("diffuse.rope.freq_base",rope_theta),("diffuse.attention.layer_norm_rms_epsilon",rms_eps),
        ("diffuse.routed_scaling_factor",routed_sf)]:
        writer.add_float32(k,v)
    writer.add_string("diffuse.model_type","llada2_moe")

    # Tokenizer
    print("Tokenizer...", flush=True)
    with open(os.path.join(model_dir, "tokenizer.json")) as f: tok_data = json.load(f)
    md = tok_data.get("model",{}); bv = md.get("vocab",{}); ml = md.get("merges",[]); at = tok_data.get("added_tokens",[])
    fv = {}; [fv.update({t:i}) for t,i in bv.items()]; [fv.update({a["content"]:a["id"]}) for a in at]
    sv = sorted(fv.items(), key=lambda x: x[1]); tl = [t for t,_ in sv]
    sc = {a["content"] for a in at}
    tt = [3 if (t in sc or t.startswith("<|") or t in ("<s>","</s>","<unk>","<pad>","<mask>")) else 1 for t in tl]
    writer.add_string("tokenizer.ggml.model","gpt2"); writer.add_token_list(tl); writer.add_token_types(tt)
    writer.add_token_scores([0.0]*len(tl))
    if ml: writer.add_token_merges(ml); print(f"  {len(ml)} merges", flush=True)
    with open(os.path.join(model_dir, "tokenizer_config.json")) as f: tc = json.load(f)
    for tk,gk in [("bos_token","tokenizer.ggml.bos_token_id"),("eos_token","tokenizer.ggml.eos_token_id"),
                   ("unk_token","tokenizer.ggml.unknown_token_id"),("pad_token","tokenizer.ggml.padding_token_id")]:
        tv = tc.get(tk)
        if isinstance(tv,dict): tv = tv.get("content")
        if tv and tv in fv: writer.add_uint32(gk, fv[tv])
    writer.add_uint32("tokenizer.ggml.mask_token_id", mask_id)
    writer.add_bool("tokenizer.ggml.add_bos_token", bool(tc.get("add_bos_token",False)))
    writer.add_bool("tokenizer.ggml.add_eos_token", bool(tc.get("add_eos_token",False)))
    # Newer HF checkpoints ship the template as a sibling file rather than a
    # tokenizer_config.json key — LLaDA2.2 does exactly that.
    ct = tc.get("chat_template")
    if not ct:
        ct_path = os.path.join(model_dir, "chat_template.jinja")
        if os.path.exists(ct_path):
            with open(ct_path, encoding="utf-8") as f: ct = f.read()
    if ct: writer.add_string("tokenizer.chat_template", ct)
    print(f"  {len(tl)} tokens", flush=True)

    # Add tensor info for all tensors
    for gguf_name, hf_name, is_f32 in tensors:
        if gguf_name.startswith("EXPERT_"):
            # Expert stack: shape is (n_experts, dim1, dim2)
            parts = gguf_name.split("_")
            stack_type = parts[1]  # GATE, UP, DOWN
            if stack_type == "GATE" or stack_type == "UP":
                shape = (n_experts, moe_ff, n_embd)
            else:
                shape = (n_experts, n_embd, moe_ff)
            dt = np.float32 if is_f32 else np.float16
            raw_dt = gguf.GGMLQuantizationType.F32 if is_f32 else out_type
            real_name = f"blk.{parts[2]}.moe_experts_{stack_type.lower()}.weight"
            writer.add_tensor_info(real_name, shape, dt, int(np.prod(shape)) * np.dtype(dt).itemsize, raw_dtype=raw_dt)
        else:
            shape = tuple(reader.get_shape(hf_name))
            dt = np.float32 if is_f32 else np.float16
            raw_dt = gguf.GGMLQuantizationType.F32 if is_f32 else out_type
            writer.add_tensor_info(gguf_name, shape, dt, int(np.prod(shape)) * np.dtype(dt).itemsize, raw_dtype=raw_dt)

    # ── Write header + KV + TI ──
    print(f"Writing header ({time.time()-t0:.0f}s)...", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # ── Pass 2: Write tensor data one at a time ──
    print(f"Pass 2: tensor data ({time.time()-t0:.0f}s)...", flush=True)
    for idx, (gguf_name, hf_name, is_f32) in enumerate(tensors):
        if gguf_name.startswith("EXPERT_"):
            parts = gguf_name.split("_")
            layer_idx = int(parts[2]); stack_type = parts[1].lower()
            ep = f"model.layers.{layer_idx}.mlp.experts."
            if stack_type == "gate" or stack_type == "up":
                shape = (n_experts, moe_ff, n_embd)
            else:
                shape = (n_experts, n_embd, moe_ff)
            arr = np.empty(shape, dtype=np.float16)
            for e in range(n_experts):
                arr[e] = reader.read_f16(f"{ep}{e}.{stack_type}_proj.weight")
            writer.write_tensor_data(arr)
            del arr
        else:
            if is_f32:
                data = reader.read_f32(hf_name)
            else:
                data = reader.read_f16(hf_name)
            writer.write_tensor_data(data)
            del data

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(tensors)} ({time.time()-t0:.0f}s)", flush=True)

    reader.close()
    writer.close()
    total = time.time() - t0
    sz = os.path.getsize(args.output) / 1e9
    print(f"\nDone! {sz:.1f}GB in {total:.0f}s ({total/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()

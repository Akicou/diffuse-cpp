#!/usr/bin/env python3
"""Convert LLaDA2.2-flash MoE model to GGUF format for diffuse-cpp.

LLaDA2.2-flash is a Mixture-of-Experts diffusion language model with:
  - Fused QKV projection
  - QK normalization (RMSNorm on head_dim)
  - Partial rotary embedding (rotary_dim=64, head_dim=128)
  - 256 experts (top-8) + 1 shared expert per MoE layer
  - Sigmoid router scoring with block-level routing
  - First layer is dense (first_k_dense_replace=1)

Usage:
    python convert-llada2.py --input /path/to/LLaDA2.2-flash --output llada2-flash-f16.gguf
    python convert-llada2.py --input /path/to/LLaDA2.2-flash --output llada2-flash-f16.gguf --type f16
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: torch not installed. Run: pip install torch", file=sys.stderr)
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
    sys.exit(1)

try:
    import gguf
except ImportError:
    print("ERROR: gguf not installed. Run: pip install gguf", file=sys.stderr)
    sys.exit(1)


def load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        return json.load(f)


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (stored as uint16) to float32."""
    u32 = raw.view(np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)


def load_tensor(st_file, name):
    """Load a tensor from a safetensors file, converting bf16 to f32."""
    with safe_open(st_file, framework="numpy") as f:
        tensor = f.get_tensor(name)
    if tensor.dtype == np.uint16:
        # Could be bf16
        tensor = bf16_to_f32(tensor)
    return tensor


def find_shard_for_tensor(model_dir, index, tensor_name):
    """Find which shard file contains a tensor."""
    shard_name = index["weight_map"].get(tensor_name)
    if shard_name is None:
        return None
    return os.path.join(model_dir, shard_name)


def main():
    parser = argparse.ArgumentParser(description="Convert LLaDA2.2-flash to GGUF")
    parser.add_argument("--input", required=True, help="Path to LLaDA2.2-flash model directory")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument("--type", default="f16", choices=["f16", "f32"],
                        help="Output precision (default: f16)")
    args = parser.parse_args()

    model_dir = args.input
    config = load_config(model_dir)

    n_layer = config["num_hidden_layers"]
    n_embd = config["hidden_size"]
    n_head = config["num_attention_heads"]
    n_head_kv = config["num_key_value_heads"]
    head_dim = config.get("head_dim", n_embd // n_head)
    n_ff = config.get("intermediate_size", 0)  # dense layer intermediate size
    moe_ff = config.get("moe_intermediate_size", 0)
    n_experts = config.get("num_experts", 0)
    n_experts_per_tok = config.get("num_experts_per_tok", 0)
    n_shared_experts = config.get("num_shared_experts", 0)
    first_k_dense = config.get("first_k_dense_replace", 0)
    n_vocab = config.get("vocab_size", 0)
    mask_token_id = 156895  # <|mask|>
    rope_theta = config.get("rope_theta", 3000000.0)
    rms_eps = config.get("rms_norm_eps", 1e-6)
    rotary_dim = config.get("rotary_dim", head_dim)
    use_qk_norm = config.get("use_qk_norm", False)
    routed_scaling_factor = config.get("routed_scaling_factor", 1.0)
    norm_topk_prob = config.get("norm_topk_prob", True)
    block_size = config.get("block_size", 32)
    expert_capacity = config.get("expert_capacity", 48)
    tie_embeddings = config.get("tie_word_embeddings", False)

    print(f"LLaDA2.2-flash configuration:")
    print(f"  layers: {n_layer} (first {first_k_dense} dense, rest MoE)")
    print(f"  embd: {n_embd}, heads: {n_head}/{n_head_kv}(kv), head_dim: {head_dim}")
    print(f"  dense FFN: {n_ff}, MoE FFN: {moe_ff}")
    print(f"  experts: {n_experts} (top-{n_experts_per_tok}), shared: {n_shared_experts}")
    print(f"  rotary_dim: {rotary_dim}, qk_norm: {use_qk_norm}")
    print(f"  vocab: {n_vocab}, mask_id: {mask_token_id}")

    # Load safetensors index
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            st_index = json.load(f)
    else:
        # Single shard
        st_index = {"weight_map": {}}
        for sf in Path(model_dir).glob("*.safetensors"):
            with safe_open(str(sf), framework="numpy") as f:
                for k in f.keys():
                    st_index["weight_map"][k] = sf.name

    # Shard cache
    shard_cache = {}
    def get_tensor(name):
        shard_name = st_index["weight_map"].get(name)
        if shard_name is None:
            raise KeyError(f"Tensor {name} not found in index")
        if shard_name not in shard_cache:
            shard_path = os.path.join(model_dir, shard_name)
            print(f"  loading shard: {shard_name}")
            shard_cache[shard_name] = str(shard_path)
        return load_tensor(shard_cache[shard_name], name)

    # Output dtype
    if args.type == "f16":
        out_type = gguf.GGMLQuantizationType.F16
    else:
        out_type = gguf.GGMLQuantizationType.F32

    writer = gguf.GGUFWriter(args.output, "diffuse")

    # ── Metadata ──────────────────────────────────────────────────
    writer.add_uint32("diffuse.block_count", n_layer)
    writer.add_uint32("diffuse.embedding_length", n_embd)
    writer.add_uint32("diffuse.attention.head_count", n_head)
    writer.add_uint32("diffuse.attention.head_count_kv", n_head_kv)
    writer.add_uint32("diffuse.feed_forward_length", n_ff if n_ff else moe_ff)
    writer.add_uint32("diffuse.context_length", config.get("max_position_embeddings", 131072))
    writer.add_uint32("diffuse.vocab_size", n_vocab)
    writer.add_uint32("diffuse.mask_token_id", mask_token_id)
    writer.add_float32("diffuse.rope.freq_base", rope_theta)
    writer.add_float32("diffuse.attention.layer_norm_rms_epsilon", rms_eps)
    writer.add_string("diffuse.model_type", "llada2_moe")
    writer.add_uint32("diffuse.head_dim", head_dim)
    writer.add_uint32("diffuse.rotary_dim", rotary_dim)
    writer.add_uint32("diffuse.use_qk_norm", 1 if use_qk_norm else 0)

    # MoE metadata
    writer.add_uint32("diffuse.expert_count", n_experts)
    writer.add_uint32("diffuse.expert_used_count", n_experts_per_tok)
    writer.add_uint32("diffuse.expert_shared_count", n_shared_experts)
    writer.add_uint32("diffuse.expert_feed_forward_length", moe_ff)
    writer.add_uint32("diffuse.first_k_dense_replace", first_k_dense)
    writer.add_float32("diffuse.routed_scaling_factor", routed_scaling_factor)
    writer.add_uint32("diffuse.norm_topk_prob", 1 if norm_topk_prob else 0)
    writer.add_uint32("diffuse.block_length", block_size)
    writer.add_uint32("diffuse.moe_block_size", block_size)
    writer.add_uint32("diffuse.expert_capacity", expert_capacity)

    # Group routing (DeepSeek-V2 style)
    writer.add_uint32("diffuse.n_group", config.get("n_group", 8))
    writer.add_uint32("diffuse.topk_group", config.get("topk_group", 4))

    # EOS token
    writer.add_uint32("diffuse.eos_token_id", config.get("pad_token_id", 156892))

    # Delete/split tokens for generation
    writer.add_uint32("diffuse.delete_token_id", 156930)
    writer.add_uint32("diffuse.split_token_id", 156931)

    # ── Embed tokenizer ────────────────────────────────────────────
    print("Embedding tokenizer...")
    try:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        # Extract vocabulary sorted by ID
        vocab = hf_tok.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        token_list = [t for t, _ in sorted_vocab]
        n_tok = len(token_list)

        # Determine token types
        special_tokens = set()
        if hasattr(hf_tok, 'get_added_vocab'):
            for t in hf_tok.get_added_vocab():
                special_tokens.add(t)

        token_types = []
        for t in token_list:
            if t in special_tokens or t.startswith('<|') or t in ('<s>', '</s>', '<unk>', '<pad>', '<mask>'):
                token_types.append(3)  # CONTROL
            else:
                token_types.append(1)  # NORMAL

        # Tokenizer model name
        tok_model = "gpt2"  # BPE default
        # Check if it's a SentencePiece tokenizer
        if hasattr(hf_tok, 'vocab_files_names'):
            if 'model' in str(hf_tok.vocab_files_names):
                sp_path = os.path.join(model_dir, 'tokenizer.model')
                if os.path.exists(sp_path):
                    tok_model = "llama"

        writer.add_string("tokenizer.ggml.model", tok_model)
        writer.add_token_list(token_list)
        writer.add_token_types(token_types)
        writer.add_token_scores([0.0] * n_tok)

        # BPE merges (if gpt2-style)
        if tok_model == "gpt2":
            merges_path = os.path.join(model_dir, "merges.txt")
            if os.path.exists(merges_path):
                merges = []
                with open(merges_path, encoding="utf-8") as mf:
                    for line in mf:
                        line = line.rstrip('\n')
                        if line and not line.startswith('#'):
                            merges.append(line)
                writer.add_token_merges(merges)
                print(f"  Embedded {len(merges)} BPE merges")
            else:
                # Try extracting from fast tokenizer backend
                try:
                    backend = hf_tok.backend_tokenizer.model
                    if hasattr(backend, 'merges'):
                        merges = [f"{a} {b}" for a, b in backend.merges]
                        writer.add_token_merges(merges)
                        print(f"  Embedded {len(merges)} BPE merges (from backend)")
                except Exception as e:
                    print(f"  WARNING: Could not extract merges: {e}")

        # Special token IDs
        for attr, gguf_key in [
            ('bos_token_id', 'tokenizer.ggml.bos_token_id'),
            ('eos_token_id', 'tokenizer.ggml.eos_token_id'),
            ('unk_token_id', 'tokenizer.ggml.unknown_token_id'),
            ('pad_token_id', 'tokenizer.ggml.padding_token_id'),
        ]:
            val = getattr(hf_tok, attr, None)
            if val is not None:
                writer.add_uint32(gguf_key, int(val))

        # Mask token
        writer.add_uint32("tokenizer.ggml.mask_token_id", mask_token_id)

        # add_bos / add_eos flags
        add_bos = getattr(hf_tok, 'add_bos_token', False)
        add_eos = getattr(hf_tok, 'add_eos_token', False)
        writer.add_bool("tokenizer.ggml.add_bos_token", bool(add_bos))
        writer.add_bool("tokenizer.ggml.add_eos_token", bool(add_eos))

        # Chat template
        if hasattr(hf_tok, 'chat_template') and hf_tok.chat_template:
            writer.add_string("tokenizer.chat_template", hf_tok.chat_template)

        print(f"  Embedded {n_tok} tokens, model={tok_model}")
        del hf_tok  # Free memory

    except ImportError:
        print("  WARNING: transformers not available, skipping tokenizer embedding")
        print("  The model will require external tokenization (Python + transformers)")
    except Exception as e:
        print(f"  WARNING: Failed to embed tokenizer: {e}")
        import traceback
        traceback.print_exc()

    # ── Global tensors ─────────────────────────────────────────────
    print("Writing global tensors...")

    # token embeddings
    tok_embd = get_tensor("model.word_embeddings.weight")
    # numpy (vocab, embd) → GGML ne[0]=embd, ne[1]=vocab
    writer.add_tensor("token_embd.weight", tok_embd.astype(np.float16 if args.type == "f16" else np.float32), raw_dtype=out_type)

    # output norm
    out_norm = get_tensor("model.norm.weight")
    writer.add_tensor("output_norm.weight", out_norm.astype(np.float32))

    # lm_head (not tied)
    if not tie_embeddings:
        lm_head = get_tensor("lm_head.weight")
        writer.add_tensor("output.weight", lm_head.astype(np.float16 if args.type == "f16" else np.float32), raw_dtype=out_type)
    else:
        writer.add_tensor("output.weight", tok_embd.astype(np.float16 if args.type == "f16" else np.float32), raw_dtype=out_type)

    # ── Per-layer tensors ──────────────────────────────────────────
    for i in range(n_layer):
        print(f"  layer {i}/{n_layer}...", end="\r")
        prefix = f"model.layers.{i}."

        # Attention (same for all layers)
        attn_prefix = prefix + "attention."

        writer.add_tensor(f"blk.{i}.attn_norm.weight",
                          get_tensor(prefix + "input_layernorm.weight").astype(np.float32))
        writer.add_tensor(f"blk.{i}.post_attn_norm.weight",
                          get_tensor(prefix + "post_attention_layernorm.weight").astype(np.float32))

        # Fused QKV: stored as (hidden, (n_heads+2*n_kv_heads)*head_dim) in HF
        # numpy shape: ((n_heads+2*n_kv_heads)*head_dim, hidden)
        qkv = get_tensor(attn_prefix + "query_key_value.weight")
        ftype = np.float16 if args.type == "f16" else np.float32
        writer.add_tensor(f"blk.{i}.attn_qkv.weight", qkv.astype(ftype), raw_dtype=out_type)

        # Attention output projection
        wo = get_tensor(attn_prefix + "dense.weight")
        writer.add_tensor(f"blk.{i}.attn_output.weight", wo.astype(ftype), raw_dtype=out_type)

        # QK norm weights (if enabled)
        if use_qk_norm:
            q_norm = get_tensor(attn_prefix + "query_layernorm.weight")
            k_norm = get_tensor(attn_prefix + "key_layernorm.weight")
            writer.add_tensor(f"blk.{i}.attn_q_norm.weight", q_norm.astype(np.float32))
            writer.add_tensor(f"blk.{i}.attn_k_norm.weight", k_norm.astype(np.float32))

        # MLP: dense (first_k_dense layers) or MoE
        if i < first_k_dense:
            # Dense MLP
            gate = get_tensor(prefix + "mlp.gate_proj.weight")
            up = get_tensor(prefix + "mlp.up_proj.weight")
            down = get_tensor(prefix + "mlp.down_proj.weight")
            writer.add_tensor(f"blk.{i}.ffn_gate.weight", gate.astype(ftype), raw_dtype=out_type)
            writer.add_tensor(f"blk.{i}.ffn_up.weight", up.astype(ftype), raw_dtype=out_type)
            writer.add_tensor(f"blk.{i}.ffn_down.weight", down.astype(ftype), raw_dtype=out_type)
        else:
            # MoE: router gate
            gate_w = get_tensor(prefix + "mlp.gate.weight")
            writer.add_tensor(f"blk.{i}.moe_gate.weight", gate_w.astype(np.float32))

            # Expert bias (optional, zeros by default)
            try:
                gate_bias = get_tensor(prefix + "mlp.gate.expert_bias")
                writer.add_tensor(f"blk.{i}.moe_gate_bias.weight", gate_bias.astype(np.float32))
            except KeyError:
                pass  # No expert bias, will default to zeros

            # Stack expert weights into 3D tensors for mul_mat_id
            # gate_proj: 256 × (moe_ff, n_embd) → (moe_ff, n_embd, n_experts)
            # GGML expects ne[0]=n_embd_in, ne[1]=n_embd_out, ne[2]=n_experts
            # For gate: weight is (moe_ff, n_embd) in numpy → GGML mul_mat expects [n_in, n_out]
            # mul_mat: a=[n_in, n_out], b=[n_in, N] → result=[n_out, N]
            # So gate weight numpy (moe_ff, n_embd) needs to be stored as (n_experts, n_embd, moe_ff)
            # to get ne[0]=moe_ff, ne[1]=n_embd... wait

            # Actually in GGML, weight tensor for mul_mat has ne[0]=input_dim, ne[1]=output_dim
            # numpy (output, input) matches: ne[0]=input, ne[1]=output
            # For 3D expert stack: ne[0]=input, ne[1]=output, ne[2]=n_experts
            # numpy shape should be (n_experts, input, output)... no

            # Let's think again. GGML tensor: ne[0] is fastest-changing dim (contiguous).
            # For a weight matrix used in mul_mat(a, b): a has ne[0]=in_dim, ne[1]=out_dim
            # numpy array (out_dim, in_dim) stored row-major → GGML gets ne[0]=in_dim, ne[1]=out_dim ✓

            # For mul_mat_id, as is 3D: ne[0]=in_dim, ne[1]=out_dim, ne[2]=n_experts
            # numpy (n_experts, out_dim, in_dim) stored row-major → ne[0]=in_dim, ne[1]=out_dim, ne[2]=n_experts ✓

            expert_gates = []
            expert_ups = []
            expert_downs = []
            for e in range(n_experts):
                ep = f"{prefix}mlp.experts.{e}."
                expert_gates.append(get_tensor(ep + "gate_proj.weight"))
                expert_ups.append(get_tensor(ep + "up_proj.weight"))
                expert_downs.append(get_tensor(ep + "down_proj.weight"))

            # Stack: list of (moe_ff, n_embd) → (n_experts, moe_ff, n_embd)
            stacked_gate = np.stack(expert_gates, axis=0).astype(ftype)
            stacked_up = np.stack(expert_ups, axis=0).astype(ftype)
            # down_proj: (n_embd, moe_ff)
            stacked_down = np.stack(expert_downs, axis=0).astype(ftype)

            writer.add_tensor(f"blk.{i}.moe_experts_gate.weight", stacked_gate, raw_dtype=out_type)
            writer.add_tensor(f"blk.{i}.moe_experts_up.weight", stacked_up, raw_dtype=out_type)
            writer.add_tensor(f"blk.{i}.moe_experts_down.weight", stacked_down, raw_dtype=out_type)

            # Shared expert
            if n_shared_experts > 0:
                sp = prefix + "mlp.shared_experts."
                shared_gate = get_tensor(sp + "gate_proj.weight")
                shared_up = get_tensor(sp + "up_proj.weight")
                shared_down = get_tensor(sp + "down_proj.weight")
                writer.add_tensor(f"blk.{i}.moe_shared_gate.weight", shared_gate.astype(ftype), raw_dtype=out_type)
                writer.add_tensor(f"blk.{i}.moe_shared_up.weight", shared_up.astype(ftype), raw_dtype=out_type)
                writer.add_tensor(f"blk.{i}.moe_shared_down.weight", shared_down.astype(ftype), raw_dtype=out_type)

    print(f"\nWriting GGUF to {args.output}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done!")


if __name__ == "__main__":
    main()

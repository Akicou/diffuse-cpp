---
license: apache-2.0
language:
  - en
  - zh
tags:
  - diffusion
  - dllm
  - gguf
  - llada
  - moe
base_model: inclusionAI/LLaDA2.2-flash
---

# LLaDA2.2-flash GGUF

Quantized GGUF files for [LLaDA2.2-flash](https://huggingface.co/inclusionAI/LLaDA2.2-flash), a ~103B parameter (13B active) Mixture-of-Experts diffusion language model from Inclusion AI / Ant Group.

## Files

| File | Format | Size | Use Case |
|------|--------|------|----------|
| `LLaDA2.2-flash-F16.gguf` | F16 | ~192 GB | Lossless conversion from BF16, best quality |
| `LLaDA2.2-flash-Q4_K_M.gguf` | Q4_K_M | ~55 GB | 4-bit mixed K-quant, balanced speed/quality |

## ⚠️ Status

**Inference is currently untested.** The GGUF files were converted from the original HuggingFace checkpoint and include the embedded BPE tokenizer. The [diffuse-cpp](https://github.com/Akicou/diffuse-cpp) inference engine implements the correct LLaDA2.2 block-diffusion architecture (block-causal attention, confidence-threshold generation, Levenshtein editing), but end-to-end generation has not yet been verified against the reference implementation.

## Conversion Method

### F16 GGUF

Converted using the custom streaming converter (`tools/convert-llada2-fast.py`) from the [diffuse-cpp](https://github.com/Akicou/diffuse-cpp) project. The converter:

1. Reads the original BF16 safetensors shards directly via raw byte parsing (no PyTorch dependency)
2. Converts BF16 → F16 in 4M-element chunks using numpy bitwise operations (`np.uint16 << 16 → view(f32) → astype(f16)`)
3. Writes the GGUF in two passes: tensor metadata (names, shapes, dtypes) first, then tensor data streamed one at a time — keeping memory usage under 2GB for a 192GB model
4. Embeds the BPE tokenizer (157,153 tokens, 156,635 merges) and all special token IDs directly into the GGUF metadata
5. Preserves all architecture metadata: block_length=32, n_group=8, topk_group=4, delete/split token IDs

### Q4_K_M GGUF

Quantized from the F16 GGUF using the streaming quantizer (`tools/quantize.cpp`):

- **Attention weights** (QKV, output projection): Q4_K
- **FFN / MoE expert weights** (gate/up/down, including 3D expert stacks): Q4_K
- **Output head** (lm_head): Q6_K (higher precision for vocabulary projection)
- **First/last layer FFN**: Q6_K (edge layers get higher precision)
- **Norm weights, MoE router gate, biases**: kept in F32
- **Token embeddings**: kept in F16

The quantizer processes one tensor at a time to avoid loading the full 192GB model into memory. MoE expert tensors (3D stacked: 256 experts × 1024 × 4096) are quantized per-row like standard 2D weights.

## Usage with diffuse-cpp

[diffuse-cpp](https://github.com/Akicou/diffuse-cpp) is a C++ inference engine for diffusion language models built on GGML.

### Build

```bash
git clone --recursive https://github.com/Akicou/diffuse-cpp.git
cd diffuse-cpp

# CPU + GPU (Vulkan for AMD/Intel, CUDA for NVIDIA)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDIFFUSE_VULKAN=ON
cmake --build build -j$(nproc)
```

### Generate Text (CLI)

```bash
./build/diffuse-cli \
    -m LLaDA2.2-flash-Q4_K_M.gguf \
    -p "What is the capital of France?" \
    -n 256 -s 32 -t 12 \
    --threshold 0.95
```

### Run the Server (like llama-server)

```bash
./build/diffuse-server -m LLaDA2.2-flash-Q4_K_M.gguf --port 8080 -ngl 99
```

Then open http://127.0.0.1:8080 for the web UI, or use the OpenAI-compatible API:

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'
```

## Architecture

LLaDA2.2 uses **block diffusion** — a semi-autoregressive generation process:
- Text is generated in **blocks of 32 tokens**
- Within each block, tokens are predicted in **parallel** (bidirectional attention)
- Blocks are processed **left to right** (causal across blocks)
- Each block undergoes iterative **denoising**: tokens are committed when their confidence exceeds the threshold (default 0.95)
- Supports **Levenshtein editing** (KEEP/SUBSTITUTE/DELETE/INSERT) for flexible sequence length changes

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-s` / `--n_steps` | 32 | Denoising steps per block |
| `--threshold` | 0.95 | Confidence threshold for token commit |
| `--temp` | 0.0 | Temperature (0 = argmax/deterministic) |
| `--remasking` | low_confidence | Token selection strategy (low_confidence or random) |
| `-ngl` | 0 | GPU layers to offload |
| `-n` | 256 | Max tokens to generate |

## Model Details

- **Parameters**: ~103B total, 13B active per token
- **Architecture**: MoE with 256 experts (top-8) + 1 shared expert
- **Layers**: 32 (1 dense + 31 MoE)
- **Hidden size**: 4096
- **Attention**: GQA (32 Q heads / 4 KV heads), head_dim=128
- **QK normalization**: RMSNorm on head_dim
- **Partial RoPE**: rotary_dim=64 (of 128)
- **Context**: 128K tokens
- **Vocab**: 157,184
- **Router**: Sigmoid scoring with group-limited top-k (n_group=8, topk_group=4)
- **Source format**: BF16 safetensors → F16 GGUF → Q4_K_M GGUF

## License

Apache 2.0 — same as the original model.

## Citation

```bibtex
@article{bie2026llada22,
  title={LLaDA2.2: Enabling Agentic Diffusion Language Models via Levenshtein Editing},
  author={Bie, Tiwei and others},
  year={2026}
}
```

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

Quantized GGUF files for [LLaDA2.2-flash](https://huggingface.co/inclusionAI/LLaDA2.2-flash), a 150B parameter (13B active) Mixture-of-Experts diffusion language model.

## Files

| File | Format | Size | Use Case |
|------|--------|------|----------|
| `LLaDA2.2-flash-F16.gguf` | F16 | ~300 GB | Lossless, best quality |
| `LLaDA2.2-flash-Q4_K_S.gguf` | Q4_K_S | ~85 GB | 4-bit quantized, balanced speed/quality |

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

### Generate Text

```bash
# CLI with integrated tokenizer
./build/diffuse-cli \
    -m LLaDA2.2-flash-Q4_K_S.gguf \
    -p "What is the capital of France?" \
    -n 256 -s 32 -t 12 \
    --threshold 0.95

# With GPU offload
./build/diffuse-cli \
    -m LLaDA2.2-flash-Q4_K_S.gguf \
    -p "Explain quantum computing" \
    -n 256 -ngl 99 -t 4
```

### Run the Server (like llama-server)

```bash
./build/diffuse-server -m LLaDA2.2-flash-Q4_K_S.gguf --port 8080 -ngl 99
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

The model also supports **Levenshtein editing** (KEEP/SUBSTITUTE/DELETE/INSERT) for flexible sequence length changes during agentic workflows.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-s` / `--n_steps` | 32 | Denoising steps per block |
| `--threshold` | 0.95 | Confidence threshold for token commit |
| `--temp` | 0.0 | Temperature (0 = argmax/deterministic) |
| `--remasking` | low_confidence | Token selection strategy |
| `-ngl` | 0 | GPU layers to offload |
| `-n` | 256 | Max tokens to generate |

## Model Details

- **Parameters**: 150B total, 13B active per token
- **Architecture**: MoE with 256 experts (top-8) + 1 shared expert
- **Layers**: 32 (1 dense + 31 MoE)
- **Hidden size**: 4096
- **Attention**: GQA (32 Q heads / 4 KV heads), head_dim=128
- **QK normalization**: RMSNorm on head_dim
- **Partial RoPE**: rotary_dim=64 (of 128)
- **Context**: 128K tokens
- **Vocab**: 157,184

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

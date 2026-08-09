---
license: apache-2.0
base_model: inclusionAI/LLaDA2.2-flash
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- dllm
- diffusion
- diffusion-language-model
- llada
- moe
- diffuse-cpp
---

<!-- Model card for https://huggingface.co/Akicou/inclusionAI_LLaDA2.2-flash-GGUF
     Keep this file in sync with that repo's README.md. -->

# LLaDA2.2-flash — GGUF

GGUF conversions of [**inclusionAI/LLaDA2.2-flash**](https://huggingface.co/inclusionAI/LLaDA2.2-flash),
a ~103B-parameter (13B active) Mixture-of-Experts **diffusion** language model (dLLM).

Published at
[**Akicou/inclusionAI_LLaDA2.2-flash-GGUF**](https://huggingface.co/Akicou/inclusionAI_LLaDA2.2-flash-GGUF).

> ⚠️ **These files do not work with `llama.cpp`.** LLaDA2 is a *block diffusion*
> model, not an autoregressive one — it denoises a block of masked tokens over
> several steps instead of predicting one token at a time. Use
> [**diffuse-cpp**](https://github.com/Akicou/diffuse-cpp).

---

## Files

| File | Quant | Size | Notes |
|---|---|---|---|
| `LLaDA2.2-flash-Q4_K_S.gguf.part-0000{1,2}-of-00002.gguf` | Q4_K_S | 59.1 GB | Recommended. Fits in ~64 GB RAM. |
| `LLaDA2.2-flash-F16.gguf.part-0000{1..5}-of-00005.gguf` | F16 | 205.9 GB | Full precision, for re-quantizing. |

Hugging Face caps individual files at 50 GB, so each model is uploaded as raw
byte-range parts. **Rejoin them with `cat` before use** (they are *not*
llama.cpp `gguf-split` shards):

```bash
huggingface-cli download Akicou/inclusionAI_LLaDA2.2-flash-GGUF \
  --include "LLaDA2.2-flash-Q4_K_S.gguf.part-*" --local-dir .
cat LLaDA2.2-flash-Q4_K_S.gguf.part-*-of-00002.gguf > LLaDA2.2-flash-Q4_K_S.gguf
```

The zero-padded names sort correctly under a shell glob. Verify the result is
`59086787040` bytes (Q4_K_S) or `205852119520` bytes (F16), then delete the parts.

---

## Usage

```bash
./build/diffuse-cli \
  -m LLaDA2.2-flash-Q4_K_S.gguf \
  -p "Explain why the sky appears blue, in two or three sentences." \
  -n 96 -s 32 -t 32
```

| Flag | Default | Description |
|---|---|---|
| `-n` | 256 | Tokens to generate |
| `-s` | 32 | Denoising steps per block |
| `-t` | 4 | Threads |
| `--temp` | 0.0 | Temperature (0 = argmax) |
| `--threshold` | 0.95 | Confidence needed to commit a token |
| `--remasking` | low_confidence | `low_confidence` or `random` |
| `--no-editing` | off | Disable Levenshtein editing |
| `-ngl` | 0 | GPU layers to offload |

---

## Quality check (Q4_K_S)

CPU-only, greedy (`--temp 0`), `-n 96 -s 32`.

**Input**

```
Explain why the sky appears blue, in two or three sentences.
```

**Output**

```
The sky appears blue due to a phenomenon called Rayleigh scattering. When
sunlight enters Earth's atmosphere, it collides with gas molecules and scatters
in all directions. Blue light, with a shorter wavelength, is scattered more
strongly than other colors, making it sky appear blue to our eyes.
```

59 tokens, stopping on `<|role_end|>`. The physics and structure are right; note
the small grammatical slip ("making it sky appear blue"), the kind of artifact to
expect from a 4-bit quant of a diffusion model.

**Speed:** ~0.1 tok/s on CPU only (no GPU offload). Diffusion decoding runs a full
forward pass per denoising step, so throughput is dominated by `steps × blocks`,
not token count. Lower `-s` to trade quality for speed.

---

## Quantization

Source weights are bf16 safetensors (32 shards). Both passes stream tensors one
at a time, so peak RAM stays near the largest tensor rather than the model:

1. **Convert** — `tools/convert-llada2-fast.py`, bf16 → F16 GGUF. Reads the
   safetensors shards by raw byte parsing (no PyTorch), converts bf16 → f16 in
   4M-element numpy chunks, and writes metadata first then streams tensor data.
   The 256 per-expert matrices in each MoE layer are stacked into single 3-D
   tensors (`[n_expert, out, in]`) so the graph can use `ggml_mul_mat_id`. The
   BPE tokenizer (157,153 tokens / 156,635 merges) and chat template are embedded.
2. **Quantize** — `tools/quantize_streaming.py --type q4_k_s`, calling ggml's own
   quantizers through a small C shim (`tools/quant_lib.c`) via ctypes.

Q4_K_S took 151 min: 254 tensors quantized, 192 copied through unchanged.

### Q4_K_S mix

| Tensor group | Type | Why |
|---|---|---|
| Router (`moe_gate`, `moe_gate_bias`) | **F32** | Routing is winner-take-all; a wrong expert costs more than the bytes saved. |
| All norms (incl. QK-norm) | **F32** | Tiny, and scale errors propagate. |
| `token_embd` | F16 | |
| `output.weight` | Q6_K | Most quantization-sensitive matmul. |
| Attention (fused QKV, output proj) | Q4_K | |
| Experts + shared expert + dense FFN | Q4_K | ~97% of the weights. |

**Q4_K_S vs Q4_K_M here:** within 30 MB of each other (59.09 vs 59.12 GB). The
usual `_M` upgrades target the first/last FFN layers, but only layer 0 is dense —
layers 1–31 are MoE and their expert tensors dominate the file. To get a
meaningfully smaller file, drop `token_embd` from F16 to Q4_K (~0.65 GB).

---

## Model details

| | |
|---|---|
| Parameters | ~103B total, 13B active per token |
| Layers | 32 (layer 0 dense, 1–31 MoE) |
| Hidden size | 4096 |
| Attention | 32 heads / 4 KV heads (GQA), head_dim 128, QK-norm |
| RoPE | partial — 64 of 128 dims, θ = 3e6, NeoX-style |
| Experts | 256 routed (top-8) + 1 shared, expert FFN 1024 |
| Routing | sigmoid + expert bias, block routing (capacity 48 per 32-token block) |
| Diffusion | block length 32, mask token `156895` |
| Context | 128K |
| Vocab | 157,184 |

---

## License & credit

Apache-2.0, inherited from the base model. All credit for the model itself goes
to the **inclusionAI** team — see the
[LLaDA2.X repository](https://github.com/inclusionAI/LLaDA2.X) and the
[technical report](https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf).
This only redistributes converted and quantized weights.

```bibtex
@article{bie2026llada22,
  title={LLaDA2.2: Enabling Agentic Diffusion Language Models via Levenshtein Editing},
  author={Bie, Tiwei and others},
  year={2026}
}
```

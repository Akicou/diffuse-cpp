# diffuse-cpp

High-performance C++ inference engine for **Diffusion Language Models**, built on GGML.

## Highlights

- **LLaDA2.2-flash** support (~103B params, 13B active) with correct block-diffusion architecture
- **Block-causal attention**: bidirectional within blocks, causal across blocks
- **Confidence-threshold generation**: commit tokens when p > 0.95, matching the reference implementation
- **Levenshtein editing**: KEEP / SUBSTITUTE / DELETE / INSERT operations
- **Block MoE routing**: sigmoid router with expert bias and per-block expert capacity
- **Integrated BPE tokenizer**: embedded in GGUF — no Python needed
- **`diffuse-server`**: HTTP API server with web UI (like `llama-server`)
- **GPU acceleration** via Vulkan (cross-vendor), CUDA (NVIDIA), or HIP (AMD ROCm)
- **Multi-architecture**: LLaDA2 MoE (block diffusion), LLaDA 1.0 / Dream (flat diffusion)

## Architecture

### Block Diffusion (LLaDA2.X)

LLaDA2.2 generates text through **block diffusion** — a semi-autoregressive process:

1. The sequence is divided into **blocks of 32 tokens**
2. Blocks are processed **left to right** (causal at block level)
3. Within each block, tokens are generated in **parallel** (bidirectional attention)
4. Each block undergoes iterative **denoising steps**:
   - Forward pass predicts logits for all masked positions
   - Sample token + probability for each mask
   - **Commit tokens above confidence threshold** (p > 0.95)
   - If not enough exceed threshold, commit **top-k by confidence**
   - Repeat until all positions are unmasked

This matches `generate()` in the reference `modeling_llada2_moe.py`. Blocks tile
from absolute position 0, so a prompt that does not end on a block boundary
shares its block with the first generated tokens.

### Levenshtein Editing (LLaDA2.2)

Unlike fixed-length substitution (LLaDA2.0/2.1), LLaDA2.2 supports four edit operations:
- **KEEP**: retain the current token
- **SUBSTITUTE**: replace mask with predicted token
- **DELETE** (token 156930): remove a token, shift sequence left
- **INSERT** (token 156931): create a new mask slot for future denoising

### MoE Routing

LLaDA2.2 uses **block routing** — expert choice is made per block of tokens, not
per token alone:

1. Sigmoid scoring (not softmax), plus a per-expert bias that affects *selection
   only* — the applied weights come from the unbiased scores
2. For each 32-token block, take every expert's **max** score over the block and
   keep the top `expert_capacity` (48 of 256)
3. Each token then picks its top-8 experts **within that allowed set**
4. Weights renormalized over the top-8, × `routed_scaling_factor` (2.5)
5. 1 shared expert, always active, running on the block input

> `n_group` / `topk_group` appear in `config.json` but the reference model never
> reads them — LLaDA2 does not use DeepSeek-style group-limited routing.

### Flat Diffusion (LLaDA 1.0 / Dream)

For LLaDA 1.0 and Dream models, generation uses flat diffusion:
- All tokens generated simultaneously with full bidirectional attention
- Iterative unmasking across the full sequence
- Two remasking strategies: `low_confidence` (highest probability first) or `random`

## Quick Start

### Build

```bash
git clone --recursive https://github.com/Akicou/diffuse-cpp.git
cd diffuse-cpp

# CPU-only
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# With GPU (Vulkan — recommended for AMD Strix Halo and other APUs)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDIFFUSE_VULKAN=ON
cmake --build build -j$(nproc)
```

### Convert a Model

```bash
# HF checkpoint → F16 GGUF (streaming; peak RAM stays near the largest tensor)
python tools/convert-llada2-fast.py \
    --input /path/to/LLaDA2.2-flash \
    --output llada2-flash-f16.gguf --type f16

# F16 GGUF → quantized (q4_k_s, q4_k_m, q6_k, q8_0, q4_0)
cc -O3 -shared -fPIC tools/quant_lib.c -o quant_lib.so -Iggml/include -Lbuild/ggml/src -lggml-base
python tools/quantize_streaming.py \
    --input llada2-flash-f16.gguf \
    --output llada2-flash-q4_k_s.gguf --type q4_k_s --lib ./quant_lib.so
```

The converter embeds the BPE tokenizer (vocab, merges, special tokens) and the
chat template into the GGUF.

### Run the Server

```bash
./build/diffuse-server -m llada2-flash-f16.gguf --port 8080 -t 12 -ngl 99
```

Open **http://127.0.0.1:8080** for the chat UI.

### CLI

```bash
./build/diffuse-cli \
    -m llada2-flash-f16.gguf \
    -p "What is the capital of France?" \
    -n 256 -s 32 -t 12 \
    --threshold 0.95
```

## API

OpenAI-compatible endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/v1/models` | GET | List models |
| `/v1/chat/completions` | POST | Chat completion |
| `/completion` | POST | Text completion |
| `/tokenize` | POST | Tokenize text |
| `/detokenize` | POST | Detokenize tokens |
| `/stats` | GET | Server statistics |

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `DIFFUSE_BUILD_TOOLS` | `ON` | Build CLI tools |
| `DIFFUSE_BUILD_SERVER` | `ON` | Build diffuse-server |
| `DIFFUSE_BUILD_TESTS` | `ON` | Build tests |
| `DIFFUSE_VULKAN` | `OFF` | Vulkan GPU backend |
| `DIFFUSE_CUDA` | `OFF` | CUDA GPU backend |
| `DIFFUSE_HIP` | `OFF` | HIP/ROCm GPU backend |

## Project Structure

```
src/
  diffuse-tokenizer.{h,cpp}    BPE tokenizer (special-token aware, chat templates)
  diffuse-model.{h,cpp}        GGUF loading (dense + MoE)
  diffuse-graph.{h,cpp}        Dense forward + compute infrastructure
  diffuse-moe-graph.cpp        MoE graph: block-causal attention + block routing
  diffuse-sampler.{h,cpp}      Block-sequential + flat generation, Levenshtein editing
  diffuse-backend.h            GGML backend scheduler (CPU/GPU)
  diffuse-json.h               JSON parser/serializer
  diffuse-common.h             Shared model/context structs

tools/
  diffuse-server.cpp           HTTP server + web UI
  diffuse-server-ui.h          Embedded chat interface
  main-cli.cpp                 CLI inference
  dump-logits.cpp              Raw logit dump (cross-validation)
  convert-llada2-fast.py       HF → GGUF converter (embeds tokenizer + template)
  quantize_streaming.py        Streaming quantizer driver
  quant_lib.c                  ggml quantizer shim used by the driver
  quantize.cpp                 Standalone quantizer (diffuse-quantize)

tests/
  test-tokenizer.cpp           Tokenizer unit tests
  test-forward.cpp             Forward-pass smoke test
  test-server.py               Server API validation
```

## License

Apache License 2.0.

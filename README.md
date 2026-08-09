# diffuse-cpp

High-performance C++ inference engine for **Diffusion Language Models**, built on GGML.

## Highlights

- **LLaDA2.2-flash** support (~103B params, 13B active) with correct block-diffusion architecture
- **Block-causal attention**: bidirectional within blocks, causal across blocks
- **Confidence-threshold generation**: commit tokens when p > 0.95, matching the reference implementation
- **Levenshtein editing**: KEEP / SUBSTITUTE / DELETE / INSERT operations
- **Group-limited MoE routing**: DeepSeek-V2 style sigmoid router with expert bias
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

This matches the reference implementation in dFactory's `generate()` method.

### Levenshtein Editing (LLaDA2.2)

Unlike fixed-length substitution (LLaDA2.0/2.1), LLaDA2.2 supports four edit operations:
- **KEEP**: retain the current token
- **SUBSTITUTE**: replace mask with predicted token
- **DELETE** (token 156930): remove a token, shift sequence left
- **INSERT** (token 156931): create a new mask slot for future denoising

### MoE Routing

LLaDA2.2 uses **DeepSeek-V2 style group-limited routing**:
- Sigmoid scoring (not softmax)
- Expert bias addition
- Group selection: 8 groups → select top-4 groups → top-8 experts within selected
- Normalized weights × routed_scaling_factor (2.5)
- 1 shared expert (always active)

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
python tools/convert-llada2.py \
    --input /path/to/LLaDA2.2-flash \
    --output llada2-flash-f16.gguf --type f16
```

The converter embeds the BPE tokenizer (vocab, merges, special tokens) into the GGUF.

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
  diffuse-tokenizer.{h,cpp}    BPE tokenizer
  diffuse-model.{h,cpp}        GGUF loading (dense + MoE)
  diffuse-graph.{h,cpp}        Dense forward + compute infrastructure
  diffuse-moe-graph.cpp        MoE graph with block-causal attention + group routing
  diffuse-sampler.{h,cpp}      Block-sequential + flat generation, Levenshtein editing
  diffuse-backend.h            GGML backend scheduler (CPU/GPU)
  diffuse-json.h               JSON parser/serializer
  diffuse-cache.h              (Legacy, unused for block diffusion)

tools/
  diffuse-server.cpp           HTTP server + web UI
  diffuse-server-ui.h          Embedded chat interface
  main-cli.cpp                 CLI inference
  convert-llada2.py            HF → GGUF converter (embeds tokenizer)

tests/
  test-tokenizer.cpp           Tokenizer unit tests
  test-server.py               Server API validation
```

## License

Apache License 2.0.

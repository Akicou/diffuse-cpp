# diffuse-cpp

High-performance C++ inference engine for **Diffusion Language Models**, built on GGML.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

## Highlights

- **LLaDA2.2-flash** support: 150B param MoE (13B active), QK-norm, partial-RoPE, GQA
- **Integrated BPE tokenizer**: embedded directly in the GGUF file — no Python needed
- **`diffuse-server`**: HTTP API server with web UI (like `llama-server`)
- **GPU acceleration** via Vulkan (cross-vendor), CUDA (NVIDIA), or HIP (AMD ROCm)
- **entropy_exit** adaptive scheduling: 2-4 steps for easy prompts, 16 for hard
- **Inter-step KV cache**: 1.6–1.8x average speedup with no quality degradation

## Quick Start

### Build

```bash
git clone --recursive https://github.com/Akicou/diffuse-cpp.git
cd diffuse-cpp

# CPU-only (works everywhere)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# With AMD/Intel GPU (Vulkan) — recommended for APUs like AMD Strix Halo
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDIFFUSE_VULKAN=ON
cmake --build build -j$(nproc)

# With NVIDIA GPU (CUDA)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDIFFUSE_CUDA=ON
cmake --build build -j$(nproc)
```

### Convert a Model (with embedded tokenizer)

```bash
python tools/convert-llada2.py \
    --input /path/to/LLaDA2.2-flash \
    --output llada2-flash-f16.gguf --type f16

# Quantize to Q4_K_M
./build/diffuse-quantize llada2-flash-f16.gguf llada2-flash-q4km.gguf Q4_K_M
```

The converter automatically embeds the BPE tokenizer (vocab, merges, special tokens)
into the GGUF file — just like llama.cpp. No external Python tokenization needed.

### Run the Server

```bash
# Start the HTTP server with web UI
./build/diffuse-server -m llada2-flash-q4km.gguf --port 8080 -t 12

# With GPU offload
./build/diffuse-server -m llada2-flash-q4km.gguf --port 8080 -ngl 32

# With API key authentication
./build/diffuse-server -m model.gguf --port 8080 --api-key secret123
```

Then open **http://127.0.0.1:8080** in your browser for the chat UI.

### CLI Usage (with integrated tokenizer)

```bash
# Text input — tokenizer is built into the GGUF
./build/diffuse-cli \
    -m llada2-flash-q4km.gguf \
    -p "What is the capital of France?" \
    -n 256 -s 16 -t 12 \
    --remasking entropy_exit
```

## diffuse-server API

The server provides an **OpenAI-compatible** API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI (chat interface) |
| `/health` | GET | Health check (no auth required) |
| `/v1/models` | GET | List models |
| `/v1/chat/completions` | POST | Chat completion (OpenAI format) |
| `/completion` | POST | Raw text completion |
| `/tokenize` | POST | Tokenize text → token IDs |
| `/detokenize` | POST | Detokenize token IDs → text |
| `/stats` | GET | Server statistics |

### Example: Chat Completion

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer secret123" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing briefly."}
        ],
        "max_tokens": 256,
        "n_steps": 16,
        "remasking": "entropy_exit"
    }'
```

### Authentication

When `--api-key` is set, all API endpoints (except `/health` and `/`) require
authentication via:
- `Authorization: Bearer <key>` header
- `X-API-Key: <key>` header

## Testing

```bash
# Tokenizer unit tests (no model required)
./build/test-tokenizer

# Server API tests (requires running server)
./build/diffuse-server -m model.gguf --port 8080 --api-key testkey &
python tests/test-server.py --port 8080 --api-key testkey
```

The server tests validate:
- **Authentication**: unauthenticated requests are rejected with 401
- **Input validation**: missing fields, wrong types, out-of-range values return 400
- **Malformed requests**: bad JSON, empty bodies, wrong content types are rejected
- **API correctness**: valid requests return proper OpenAI-compatible responses
- **CORS**: headers are present for browser access
- **Roundtrip**: tokenize → detokenize preserves content

## How It Works

Diffusion LLMs generate text through iterative refinement:

1. Start with all tokens masked (`<|mask|>`)
2. Forward pass: predict logits for all positions simultaneously
3. Unmask a fraction of tokens (lowest entropy first via `entropy_exit`)
4. Repeat until all tokens are unmasked

Unlike autoregressive models (one token per forward pass), diffusion models
generate all tokens in parallel — reading model weights once per step instead
of once per token.

## Supported Model

| Model | Architecture | Params | Status |
|-------|-------------|--------|--------|
| [LLaDA2.2-flash](https://huggingface.co/inclusionAI/LLaDA2.2-flash) | Custom MoE, GQA (32/4), QK-norm, partial-RoPE | 150B (13B active) | Production |

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `DIFFUSE_BUILD_TOOLS` | `ON` | Build CLI tools (diffuse-cli, diffuse-quantize) |
| `DIFFUSE_BUILD_SERVER` | `ON` | Build diffuse-server (HTTP API + Web UI) |
| `DIFFUSE_BUILD_TESTS` | `ON` | Build test suite |
| `DIFFUSE_VULKAN` | `OFF` | Enable Vulkan GPU backend (cross-vendor) |
| `DIFFUSE_CUDA` | `OFF` | Enable CUDA GPU backend (NVIDIA) |
| `DIFFUSE_HIP` | `OFF` | Enable HIP/ROCm GPU backend (AMD, Linux) |

## GPU Backend Notes

**Vulkan** is the recommended backend for AMD Strix Halo and other APUs:

```bash
# Fedora: install Vulkan dependencies
sudo dnf install vulkan-headers vulkan-loader-devel glslc glslang

# Build with Vulkan
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDIFFUSE_VULKAN=ON
cmake --build build -j$(nproc)

# Run with GPU offload
./build/diffuse-server -m model.gguf --port 8080 -ngl 99
```

Use `-ngl 99` to offload all layers to GPU. If GPU is unavailable, the engine
falls back to CPU automatically.

## Project Structure

```
src/
  diffuse-tokenizer.{h,cpp}   BPE tokenizer (reads from GGUF)
  diffuse-model.{h,cpp}       GGUF model loading (MoE)
  diffuse-graph.{h,cpp}       Compute infrastructure + forward dispatch
  diffuse-moe-graph.cpp       MoE transformer graph builder
  diffuse-sampler.{h,cpp}     Iterative unmasking diffusion loop
  diffuse-cache.h             Inter-step KV cache
  diffuse-backend.h           GGML backend scheduler (CPU/GPU)
  diffuse-json.h              Minimal JSON parser/serializer

tools/
  diffuse-server.cpp          HTTP server + web UI (like llama-server)
  diffuse-server-ui.h         Embedded HTML/CSS/JS chat interface
  main-cli.cpp                Command-line inference
  convert-llada2.py           HF → GGUF converter (embeds tokenizer)
  quantize.cpp                Model quantization

tests/
  test-tokenizer.cpp          Tokenizer unit tests
  test-server.py              Server API validation tests
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

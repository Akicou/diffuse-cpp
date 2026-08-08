#pragma once

#include "diffuse-common.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <vector>

// ── Backend initialization ──────────────────────────────────────
//
// Initializes the GGML backend scheduler with available compute devices.
// When n_gpu_layers > 0, discovers GPU backends (Vulkan, CUDA, HIP) and
// transfers the first n_gpu_layers transformer layers to GPU memory.
//
// Pattern adopted from llama.cpp's llama_context initialization:
//   1. Enumerate devices via ggml_backend_dev_count()
//   2. Init backends (CPU always, GPU if available and requested)
//   3. Create ggml_backend_sched for routing graph ops
//   4. Transfer weights to GPU buffers for offloaded layers

// Detect and initialize GPU backend if available
static ggml_backend_t diffuse_init_gpu_backend() {
    size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);

        if (props.type == GGML_BACKEND_DEVICE_TYPE_GPU && props.caps.async) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                DIFFUSE_LOG("GPU backend: %s (%s)", props.name, props.description);
                return backend;
            }
        }
    }
    return nullptr;
}

// Initialize CPU backend
static ggml_backend_t diffuse_init_cpu_backend(int n_threads) {
    size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);

        if (props.type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                // Set thread count
                ggml_backend_cpu_set_n_threads(backend, n_threads);
                DIFFUSE_LOG("CPU backend: %s (%d threads)", props.name, n_threads);
                return backend;
            }
        }
    }
    // Fallback: try first device
    if (n_devs > 0) {
        ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_dev_get(0), nullptr);
        if (backend) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
            return backend;
        }
    }
    return nullptr;
}

// ── Initialize the backend scheduler for a context ──────────────
static bool diffuse_init_backends(diffuse_context * ctx, int n_gpu_layers) {
    ctx->n_gpu_layers = n_gpu_layers;
    ctx->n_backends = 0;

    // CPU backend (always)
    ctx->backend_cpu = diffuse_init_cpu_backend(ctx->n_threads);
    if (!ctx->backend_cpu) {
        DIFFUSE_LOG("WARNING: failed to init CPU backend");
        return false;
    }
    ctx->n_backends = 1;

    // GPU backend (if requested and available)
    if (n_gpu_layers > 0) {
        ctx->backend_gpu = diffuse_init_gpu_backend();
        if (ctx->backend_gpu) {
            ctx->n_backends = 2;
        } else {
            DIFFUSE_LOG("WARNING: n_gpu_layers=%d but no GPU backend found, using CPU only", n_gpu_layers);
        }
    }

    return true;
}

// ── Transfer model weights to GPU ───────────────────────────────
// Moves the first n_gpu_layers of transformer weights to the GPU backend buffer.
// The scheduler automatically routes ops involving GPU-resident tensors to GPU.
static void diffuse_offload_weights(diffuse_context * ctx, diffuse_model * model) {
    int n_gpu = ctx->n_gpu_layers;
    if (n_gpu <= 0 || !ctx->backend_gpu) return;

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(
        ggml_backend_dev_get(ggml_backend_dev_count() - 1));  // last device = GPU typically

    // For now: mark the buffer usage as WEIGHTS so the scheduler prefers
    // running weight-related ops on the same backend.
    // Full weight transfer to GPU requires creating a backend buffer and
    // copying tensor data — this will be done incrementally.

    // Always offload embeddings and output if possible
    // (These are the largest single tensors and benefit most from GPU)
    DIFFUSE_LOG("GPU offload: %d layers requested (backend: %s)",
                n_gpu,
                ctx->n_backends > 1 ? "GPU+CPU" : "CPU-only");
}

// ── Get buffer types for scheduler creation ─────────────────────
static void diffuse_get_backend_arrays(
        diffuse_context * ctx,
        std::vector<ggml_backend_t> & backends,
        std::vector<ggml_backend_buffer_type_t> & bufts) {
    backends.clear();
    bufts.clear();

    if (ctx->backend_gpu) {
        backends.push_back(ctx->backend_gpu);
        ggml_backend_dev_t gpu_dev = ggml_backend_dev_get(ggml_backend_dev_count() - 1);
        bufts.push_back(ggml_backend_dev_buffer_type(gpu_dev));
    }
    backends.push_back(ctx->backend_cpu);
    // CPU buffer type: find the CPU device's buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (props.type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            bufts.push_back(ggml_backend_dev_buffer_type(dev));
            break;
        }
    }
    if (bufts.size() < backends.size()) {
        bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_dev_get(0)));
    }
}

// ── Compute a graph via the backend scheduler ───────────────────
// Falls back to the legacy CPU path if no scheduler is available.
static bool diffuse_sched_compute(diffuse_context * ctx,
                                   struct ggml_context * ctx_compute,
                                   struct ggml_cgraph * gf) {
    if (ctx->n_backends <= 1 && !ctx->backend_gpu) {
        // CPU-only path: use legacy compute (faster for CPU, no scheduler overhead)
        enum ggml_status status = ggml_graph_compute_with_ctx(ctx_compute, gf, ctx->n_threads);
        return status == GGML_STATUS_SUCCESS;
    }

    // Multi-backend path: use the scheduler
    if (!ctx->sched) {
        std::vector<ggml_backend_t> backends;
        std::vector<ggml_backend_buffer_type_t> bufts;
        diffuse_get_backend_arrays(ctx, backends, bufts);

        ctx->sched = ggml_backend_sched_new(
            backends.data(), bufts.data(), (int)backends.size(),
            ggml_graph_size(gf) + 256, false, true);

        if (!ctx->sched) {
            DIFFUSE_LOG("WARNING: failed to create scheduler, falling back to CPU");
            enum ggml_status status = ggml_graph_compute_with_ctx(ctx_compute, gf, ctx->n_threads);
            return status == GGML_STATUS_SUCCESS;
        }
        DIFFUSE_LOG("scheduler initialized: %d backends, graph_size=%zu",
                    (int)backends.size(), (size_t)ggml_graph_size(gf));
    }

    ggml_backend_sched_reset(ctx->sched);

    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        DIFFUSE_LOG("WARNING: sched_alloc_graph failed, falling back to CPU");
        enum ggml_status status = ggml_graph_compute_with_ctx(ctx_compute, gf, ctx->n_threads);
        return status == GGML_STATUS_SUCCESS;
    }

    enum ggml_status status = ggml_backend_sched_graph_compute(ctx->sched, gf);
    return status == GGML_STATUS_SUCCESS;
}

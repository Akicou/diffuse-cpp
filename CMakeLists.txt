cmake_minimum_required(VERSION 3.14)
project(diffuse-cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(DIFFUSE_BUILD_TESTS   "Build tests"      ON)
option(DIFFUSE_BUILD_TOOLS   "Build CLI tools"   ON)

# ── GPU backend options ─────────────────────────────────────────
# Enable one or more GPU backends. diffuse-cpp auto-detects available
# devices at runtime via ggml_backend_dev_count().
# Vulkan is the recommended cross-vendor backend (NVIDIA, AMD, Intel).
option(DIFFUSE_VULKAN  "Enable Vulkan backend (cross-vendor GPU)" OFF)
option(DIFFUSE_CUDA    "Enable CUDA backend (NVIDIA GPU)"        OFF)
option(DIFFUSE_HIP     "Enable HIP/ROCm backend (AMD GPU)"       OFF)

# ── GGML submodule ──────────────────────────────────────────────
set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP ON CACHE BOOL "" FORCE)

# Propagate GPU backend selections to GGML
if(DIFFUSE_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(DIFFUSE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(DIFFUSE_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

# ── Determine which GGML targets to link ────────────────────────
# CPU is always linked. GPU backends are conditionally linked so the
# scheduler can discover them at runtime.
set(DIFFUSE_GGML_LINKS ggml ggml-cpu)

if(TARGET ggml-vulkan)
    list(APPEND DIFFUSE_GGML_LINKS ggml-vulkan)
endif()
if(TARGET ggml-cuda)
    list(APPEND DIFFUSE_GGML_LINKS ggml-cuda)
endif()
if(TARGET ggml-hip)
    list(APPEND DIFFUSE_GGML_LINKS ggml-hip)
endif()

# ── Core library (diffusion + autoregressive) ─────────────────
add_library(diffuse STATIC
    src/diffuse-model.cpp
    src/diffuse-graph.cpp
    src/diffuse-moe-graph.cpp
    src/diffuse-sampler.cpp
    src/ar-graph.cpp
    src/ar-sampler.cpp
    src/ar-speculative.cpp
)
target_include_directories(diffuse PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(diffuse PUBLIC ${DIFFUSE_GGML_LINKS})

# ── CLI tools ───────────────────────────────────────────────────
if(DIFFUSE_BUILD_TOOLS)
    # Diffusion CLI (LLaDA, Dream, LLaDA2-MoE)
    add_executable(diffuse-cli tools/main-cli.cpp)
    target_link_libraries(diffuse-cli PRIVATE diffuse)

    # Autoregressive CLI (Qwen2.5, Llama, etc.)
    add_executable(diffuse-ar tools/ar-cli.cpp)
    target_link_libraries(diffuse-ar PRIVATE diffuse)

    add_executable(diffuse-bench tools/main-bench.cpp)
    target_link_libraries(diffuse-bench PRIVATE diffuse)

    add_executable(dump-logits tools/dump-logits.cpp)
    target_link_libraries(dump-logits PRIVATE diffuse)

    add_executable(diffuse-quantize tools/quantize.cpp)
    target_link_libraries(diffuse-quantize PRIVATE ggml)
endif()

# ── Tests ───────────────────────────────────────────────────────
if(DIFFUSE_BUILD_TESTS)
    enable_testing()
    add_executable(test-forward tests/test-forward.cpp)
    target_link_libraries(test-forward PRIVATE diffuse)
    add_test(NAME test-forward COMMAND test-forward)
endif()

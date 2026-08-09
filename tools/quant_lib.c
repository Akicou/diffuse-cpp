
// quant_lib.c: GGML quantization as a shared library for Python ctypes
// Build: gcc -shared -fPIC -o quant_lib.so quant_lib.c -I ggml/include -L ggml/src -lggml

#include <ggml.h>
#include <string.h>
#include <stdlib.h>

// Quantize f32 data to target type.
// Returns actual output bytes, or 0 on error.
// out_buf must be large enough (caller allocates via ggml_row_size).
size_t quantize_to_type(int dst_type_int, const float * f32_data, void * out_buf,
                         int64_t nrows, int64_t n_per_row) {
    ggml_type dst_type = (ggml_type)dst_type_int;

    ggml_quantize_init(dst_type);

    size_t expected = ggml_row_size(dst_type, n_per_row) * nrows;

    size_t actual = ggml_quantize_chunk(dst_type, f32_data, out_buf, 0, nrows, n_per_row, nullptr);

    return actual;
}

// Get the output size for quantized data
size_t get_quantized_size(int dst_type_int, int64_t nrows, int64_t n_per_row) {
    ggml_type dst_type = (ggml_type)dst_type_int;
    return ggml_row_size(dst_type, n_per_row) * nrows;
}

// Get block size for a quantization type
int get_block_size(int dst_type_int) {
    return (int)ggml_blck_size((ggml_type)dst_type_int);
}

// Convert f16 → f32
void f16_to_f32(const void * f16_data, float * f32_data, int64_t n) {
    ggml_fp16_to_fp32_row((const ggml_fp16_t *)f16_data, f32_data, n);
}

// Init quantization tables for a type
void init_quant(int dst_type_int) {
    ggml_quantize_init((ggml_type)dst_type_int);
}

// Free quantization tables
void cleanup_quant(void) {
    ggml_quantize_free();
}

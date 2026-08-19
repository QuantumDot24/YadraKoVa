#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// Usamos extern "C" para evitar el Name Mangling de C++ y que el
// KernelRegistry encuentre la función exacta por su nombre.
extern "C" {

constexpr int MAX_DIMS = 8;

struct TensorViewInfo {
    int64_t shape[MAX_DIMS];
    int64_t strides[MAX_DIMS];
    int ndim;
};

__global__ void contiguous_kernel(
    const KERNEL_DTYPE* src,
    KERNEL_DTYPE* dst,
    TensorViewInfo view,
    int64_t numel
) {
    int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (linear_idx < numel) {
        int64_t src_idx = 0;
        int64_t temp_idx = linear_idx;

        // Descomponer índice físico a lógico usando la metadata de la vista
        for (int i = view.ndim - 1; i >= 0; --i) {
            int64_t coord = temp_idx % view.shape[i];
            temp_idx /= view.shape[i];
            src_idx += coord * view.strides[i];
        }

        dst[linear_idx] = src[src_idx];
    }
}

} // extern "C"
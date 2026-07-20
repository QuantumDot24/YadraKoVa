#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <string>

namespace yadrakova::core {

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t err, const char* expr, const char* file, int line,
               const std::string& context = "")
        : std::runtime_error(build(err, expr, file, line, context)), code_(err) {}
    [[nodiscard]] cudaError_t code() const noexcept { return code_; }
private:
    cudaError_t code_;
    static std::string build(cudaError_t err, const char* expr, const char* file, int line,
                              const std::string& context) {
        std::string msg;
        if (!context.empty()) msg += context + ": ";
        msg += std::string("CUDA error: ") + cudaGetErrorString(err) +
               " (" + cudaGetErrorName(err) + ")\n  expr: " + expr +
               "\n  at: " + file + ":" + std::to_string(line);
        return msg;
    }
};

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line,
                        const std::string& context = "") {
    if (err != cudaSuccess) throw CudaError(err, expr, file, line, context);
}

// Para kernels/registry.h, que usa driver API (cubins)
inline void cu_check(CUresult err, const char* expr, const char* file, int line,
                      const std::string& context = "") {
    if (err == CUDA_SUCCESS) return;
    const char *name = nullptr, *desc = nullptr;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &desc);
    std::string msg;
    if (!context.empty()) msg += context + ": ";
    msg += std::string("CUDA driver error: ") + (desc ? desc : "?") +
        " (" + (name ? name : "?") + ")\n  expr: " + expr + "\n  at: " + file + ":" + std::to_string(line);
    throw std::runtime_error(msg);
}

} // namespace yadrakova::core

#define CUDA_CHECK(expr) ::yadrakova::core::cuda_check((expr), #expr, __FILE__, __LINE__)
#define CU_CHECK(expr)   ::yadrakova::core::cu_check((expr), #expr, __FILE__, __LINE__)

// Variantes con contexto -- para cuando el mensaje generico no basta
// (ej: "Graph 'backward_step': cudaGraphLaunch fallo...").
#define CUDA_CHECK_CTX(expr, ctx) ::yadrakova::core::cuda_check((expr), #expr, __FILE__, __LINE__, (ctx))
#define CU_CHECK_CTX(expr, ctx)   ::yadrakova::core::cu_check((expr), #expr, __FILE__, __LINE__, (ctx))
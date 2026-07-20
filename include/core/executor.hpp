#pragma once
#include "core/stream.hpp"
#include "core/cuda_error.hpp"
#include "core/dispatch_registry.hpp"
#include "dtype_utils.hpp"
#include "kernels/registry.hpp"
#include <cuda.h>
#include <string>
#include <vector>

namespace yadrakova::core {

    class Executor {
    public:
        // args ya debe traer TODOS los punteros del kernel en el orden
        // exacto de su firma (incluyendo M, N, K si el kernel los recibe
        // como escalares -- Executor no los inserta, porque no sabe la
        // firma del kernel; eso lo decide quien arma args, ej. Tensor::matmul).
        template <typename T>
        static void execute(const std::string& kernel_name,
                            int64_t M, int64_t N, int64_t K, // solo para DispatchRegistry
                            const std::vector<void*>& args,
                            Stream& stream)
        {
            DType dtype = dtype_traits<T>::value;

            // TODO: leer de un contexto global de dispositivo mas adelante;
            // por ahora, arquitectura nativa fija.
            constexpr kernels::Arch current_arch = kernels::Arch::SM_86;

            CUfunction fn = kernels::KernelRegistry::instance().get_function(kernel_name, current_arch, dtype);
            DispatchDims dims = DispatchRegistry::instance().get_dims(kernel_name, M, N, K);

            CU_CHECK(cuLaunchKernel(fn,
                                     dims.grid.x,  dims.grid.y,  dims.grid.z,
                                     dims.block.x, dims.block.y, dims.block.z,
                                     dims.shared_mem_bytes,
                                     stream.raw(),
                                     const_cast<void**>(args.data()),
                                     nullptr));
        }
    };

} // namespace yadrakova::core
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
        // dims: mapa nombrado (ej. {"M":64,"N":64,"K":64} o {"n":4096}),
        // debe coincidir con los nombres declarados en `dims:` del yaml
        // del kernel. args ya trae todos los punteros/escalares en el
        // orden exacto de la firma real (ver args: del yaml).
        template <typename T>
        static void execute(const std::string& kernel_name,
                            const DimMap& dims,
                            const std::vector<void*>& args,
                            Stream& stream)
        {
            DType dtype = dtype_traits<T>::value;

            constexpr kernels::Arch current_arch = kernels::Arch::SM_86;

            CUfunction fn = kernels::KernelRegistry::instance().get_function(kernel_name, current_arch, dtype);
            DispatchDims d = DispatchRegistry::instance().get_dims(kernel_name, dims);

            CU_CHECK(cuLaunchKernel(fn,
                                     d.grid.x,  d.grid.y,  d.grid.z,
                                     d.block.x, d.block.y, d.block.z,
                                     d.shared_mem_bytes,
                                     stream.raw(),
                                     const_cast<void**>(args.data()),
                                     nullptr));
        }
    };

} // namespace yadrakova::core
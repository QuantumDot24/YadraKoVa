#pragma once
#include "core/stream.hpp"
#include "core/cuda_error.hpp"
#include "core/dispatch_registry.hpp"
#include "dtype_utils.hpp"
#include "kernels/registry.hpp"
#include <cuda.h>
#include <string>
#include <vector>
#include "device.hpp"

namespace yadrakova::core
{
    class Executor
    {
    public:
        template <typename T>
        static void execute(const std::string& kernel_name,
                            const DimMap& dims,
                            const std::vector<void*>& args,
                            Stream& stream)
        {
            DType dtype = dtype_traits<T>::value;

            kernels::Arch current_arch = Device::instance().arch();

            CUfunction fn = kernels::KernelRegistry::instance().get_function(kernel_name, current_arch, dtype);
            DispatchDims d = DispatchRegistry::instance().get_dims(kernel_name, dims);

            CU_CHECK(cuLaunchKernel(fn,
                d.grid.x, d.grid.y, d.grid.z,
                d.block.x, d.block.y, d.block.z,
                d.shared_mem_bytes,
                stream.raw(),
                const_cast<void**>(args.data()),
                nullptr));
        }
    };
} // namespace yadrakova::core

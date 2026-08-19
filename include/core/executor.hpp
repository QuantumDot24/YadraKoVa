#pragma once
#include "core/cuda_error.hpp"
#include "core/stream.hpp"
#include "device.hpp"
#include "dtype_utils.hpp"
#include "kernels/registry.hpp"
#include <cstdint>
#include <cuda.h>
#include <format>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace yadrakova::core
{
    struct Dim3
    {
        unsigned int x = 1, y = 1, z = 1;
    };

    struct DispatchDims
    {
        Dim3 grid;
        Dim3 block;
        unsigned int shared_mem_bytes = 0;
    };

    using DimMap = std::unordered_map<std::string, int64_t>;
    using DispatchFunc = std::function<DispatchDims(const DimMap&)>;

    // Registro global kernel_name -> regla de grid/block. Cada kernel .cu se
    // registra a si mismo en un initializer estatico, igual que KernelRegistry.
    class DispatchRegistry
    {
    public:
        static DispatchRegistry& instance()
        {
            static DispatchRegistry inst;
            return inst;
        }

        void register_dispatch(const std::string& kernel_name, DispatchFunc fn)
        {
            table_[kernel_name] = std::move(fn);
        }

        [[nodiscard]] DispatchDims get_dims(const std::string& kernel_name, const DimMap& dims) const
        {
            auto it = table_.find(kernel_name);
            if (it == table_.end())
                throw std::runtime_error(std::format("DispatchRegistry: no hay regla para '{}'", kernel_name));
            return it->second(dims);
        }

        [[nodiscard]] bool has(const std::string& kernel_name) const { return table_.contains(kernel_name); }

    private:
        DispatchRegistry() = default;
        std::unordered_map<std::string, DispatchFunc> table_;
    };

    // Unico punto de lanzamiento de kernels .cu: resuelve la funcion segun
    // arch+dtype via KernelRegistry, resuelve las dims via DispatchRegistry,
    // y lanza. Tensor llama aca cuando OpsDispatch resuelve a Backend::Custom.
    class Executor
    {
    public:
        template <typename T>
        static void execute(const std::string& kernel_name, const DimMap& dims,
                             const std::vector<void*>& args, Stream& stream)
        {
            DType dtype = dtype_traits<T>::value;
            kernels::Arch arch = Device::instance().arch();

            CUfunction fn = kernels::KernelRegistry::instance().get_function(kernel_name, arch, dtype);
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
#pragma once
#include "core/tensor.h"
#include <cuda.h>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <mutex>

namespace yadrakova::kernels
{
    enum class Arch { SM_86, SM_100 };

    struct KernelBinary
    {
        const unsigned char* data;
        size_t size;
    };

    class KernelRegistry
    {
    public:
        static KernelRegistry& instance()
        {
            static KernelRegistry reg;
            return reg;
        }

        void register_kernel(const std::string& name, Arch arch, core::DType dtype, KernelBinary binary)
        {
            std::lock_guard<std::mutex> lock(mtx_);
            binaries_[key(name, arch, dtype)] = binary;
        }

        CUfunction get_function(const std::string& name, Arch arch, core::DType dtype)
        {
            std::lock_guard<std::mutex> lock(mtx_);
            std::string k = key(name, arch, dtype);

            auto loaded_it = loaded_.find(k);
            if (loaded_it != loaded_.end()) return loaded_it->second.function;

            auto bin_it = binaries_.find(k);
            if (bin_it == binaries_.end())
            {
                throw std::runtime_error("KernelRegistry: no hay cubin registrado para '" + k + "'");
            }

            static bool cu_init_done = false;
            if (!cu_init_done)
            {
                cuInit(0);
                cu_init_done = true;
            }

            Entry entry{};
            CUresult err = cuModuleLoadData(&entry.module, bin_it->second.data);
            if (err != CUDA_SUCCESS)
            {
                throw std::runtime_error("cuModuleLoadData fallo para '" + k + "'");
            }
            err = cuModuleGetFunction(&entry.function, entry.module, (name + "_kernel").c_str());
            if (err != CUDA_SUCCESS)
            {
                throw std::runtime_error("cuModuleGetFunction fallo para '" + k + "'");
            }

            loaded_[k] = entry;
            return entry.function;
        }

    private:
        struct Entry
        {
            CUmodule module = nullptr;
            CUfunction function = nullptr;
        };

        static std::string key(const std::string& name, Arch arch, core::DType dtype)
        {
            return name + "|" + std::to_string((int)arch) + "|" + std::to_string((int)dtype);
        }

        std::mutex mtx_;
        std::unordered_map<std::string, KernelBinary> binaries_;
        std::unordered_map<std::string, Entry> loaded_;
    };
} // namespace yadrakova::kernels

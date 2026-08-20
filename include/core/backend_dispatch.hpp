#pragma once
#include "core/backend.hpp"
#include "core/stream.hpp"
#include "core/executor.hpp"
#include "dtype_utils.hpp"
#include <cuda_runtime.h>
#include <format>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include "generated/ops/ops_caps.hpp"

namespace yadrakova::core
{
    struct BackendHandles
    {
        void* cublas = nullptr;
        void* cudnn = nullptr;
        void* cutensor = nullptr;
        void* curand = nullptr;

        std::function<void(void*)> destroy_cublas;
        std::function<void(void*)> destroy_cudnn;
        std::function<void(void*)> destroy_cutensor;
        std::function<void(void*)> destroy_curand;

        ~BackendHandles()
        {
            if (cublas && destroy_cublas) destroy_cublas(cublas);
            if (cudnn && destroy_cudnn) destroy_cudnn(cudnn);
            if (cutensor && destroy_cutensor) destroy_cutensor(cutensor);
            if (curand && destroy_curand) destroy_curand(curand);
        }
    };

    class BackendContext
    {
    public:
        static BackendContext& instance()
        {
            static BackendContext inst;
            return inst;
        }

        BackendHandles& handles_for(const Stream& stream)
        {
            std::lock_guard lock(mtx_);
            auto [it, _] = table_.try_emplace(stream.raw(), std::make_unique<BackendHandles>());
            return *it->second;
        }

        void reset(const Stream& stream)
        {
            std::lock_guard lock(mtx_);
            table_.erase(stream.raw());
        }

    private:
        BackendContext() = default;
        std::mutex mtx_;
        std::unordered_map<cudaStream_t, std::unique_ptr<BackendHandles>> table_;
    };

    using BackendOpFunc = std::function<void(const std::vector<void*>& args, Stream& stream)>;

    class BackendRegistry
    {
    public:
        static BackendRegistry& instance()
        {
            static BackendRegistry inst;
            return inst;
        }

        void register_op(Op op, Backend backend, DType dtype, BackendOpFunc fn)
        {
            table_[key(op, backend, dtype)] = std::move(fn);
        }

        [[nodiscard]] bool has(Op op, Backend backend, DType dtype) const
        {
            return table_.contains(key(op, backend, dtype));
        }

        void invoke(Op op, Backend backend, DType dtype, const std::vector<void*>& args, Stream& stream) const
        {
            auto it = table_.find(key(op, backend, dtype));
            if (it == table_.end())
            {
                throw std::runtime_error(std::format(
                    "BackendRegistry: no hay implementacion registrada para {} / {} / {}",
                    op_name(op), backend_name(backend), dtype_name(dtype)));
            }
            it->second(args, stream);
        }

    private:
        BackendRegistry() = default;
        static uint32_t key(Op op, Backend backend, DType dtype)
        {
            return (std::to_underlying(op) << 16) | (std::to_underlying(backend) << 8) | std::to_underlying(dtype);
        }
        std::unordered_map<uint32_t, BackendOpFunc> table_;
    };

    // ¡MAGIA! BackendCaps usa las funciones generadas en ops_caps.hpp
    class BackendCaps
    {
    public:
        static DTypeMask supported_dtypes(Op op, Backend backend)
        {
            if (backend == Backend::Custom) return kAllDTypes;
            return get_supported_dtypes(op, backend);
        }
        static bool supports(Op op, Backend backend, DType dt)
        {
            return (supported_dtypes(op, backend) & dtype_bit(dt)) != 0;
        }
        static const std::vector<Backend>& preference_order(Op op)
        {
            return get_preference_order(op);
        }
    };

    struct BackendResolution
    {
        Op op;
        Backend backend;
        DType compute_dtype;
        bool dtype_was_downgraded;
    };

    class OpsDispatch
    {
    public:
        static BackendResolution resolve(Op op, DType requested_dtype,
                                          Backend requested_backend = Backend::Auto,
                                          bool warn_on_downgrade = true)
        {
            if (requested_backend != Backend::Auto)
            {
                if (requested_backend == Backend::Custom ||
                    BackendCaps::supports(op, requested_backend, requested_dtype))
                {
                    return {op, requested_backend, requested_dtype, false};
                }
                return downgrade(op, requested_backend, requested_dtype, warn_on_downgrade);
            }
            for (Backend candidate : BackendCaps::preference_order(op))
            {
                if (candidate == Backend::Custom || BackendCaps::supports(op, candidate, requested_dtype))
                {
                    return {op, candidate, requested_dtype, false};
                }
            }
            return downgrade(op, BackendCaps::preference_order(op).front(), requested_dtype, warn_on_downgrade);
        }

    private:
        static BackendResolution downgrade(Op op, Backend backend, DType requested_dtype, bool warn)
        {
            if (backend != Backend::Custom && BackendCaps::supports(op, backend, kDefaultDType))
            {
                if (warn)
                {
                    std::cerr << std::format("[yadrakova] {}: {} no soporta dtype '{}', usando '{}' en su lugar.\n",
                                              op_name(op), backend_name(backend), dtype_name(requested_dtype),
                                              dtype_name(kDefaultDType));
                }
                return {op, backend, kDefaultDType, true};
            }
            if (warn)
            {
                std::cerr << std::format("[yadrakova] {}: {} no soporta dtype '{}' (ni en bf16), cayendo a kernel custom.\n",
                                          op_name(op), backend_name(backend), dtype_name(requested_dtype));
            }
            return {op, Backend::Custom, requested_dtype, false};
        }
    };

    template <typename T>
    void dispatch_op(Op op, const std::string& kernel_name, const DimMap& dims,
                      const std::vector<void*>& args, Backend backend, Stream& stream)
    {
        auto resolution = OpsDispatch::resolve(op, dtype_traits<T>::value, backend);
        if (resolution.backend == Backend::Custom)
            Executor::execute<T>(kernel_name, dims, args, stream);
        else
            BackendRegistry::instance().invoke(op, resolution.backend, resolution.compute_dtype, args, stream);
    }

} // namespace yadrakova::core
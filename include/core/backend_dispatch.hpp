#pragma once
#include "core/backend.hpp"
#include "core/stream.hpp"
#include "dtype_utils.hpp"
#include <array>
#include <cuda_runtime.h>
#include <format>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace yadrakova::core
{
    // ------------------------------------------------------------------
    // Handles por-stream de cada libreria (cuBLAS/cuDNN/cuTensor/cuRAND).
    // Se guardan como void* porque este header no depende de los SDKs
    // reales -- cada backend_*.cpp hace el reinterpret_cast al tipo
    // concreto cuando crea su handle y registra su propio destructor.
    // ------------------------------------------------------------------
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

    // Un set de handles por Stream (cuBLAS/cuDNN/cuRAND se atan a un stream
    // via *SetStream; cuTensor no, pero se guarda igual por comodidad).
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

        // Se llama al destruir/reciclar un Stream, para no dejar handles
        // colgando apuntando a un cudaStream_t muerto.
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

    // ------------------------------------------------------------------
    // Registro global (Op, Backend, DType) -> implementacion. Cada
    // backend_*.cpp se registra a si mismo en un initializer estatico.
    // ------------------------------------------------------------------
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
                    "BackendRegistry: no hay implementacion registrada para {} / {} / {} "
                    "(falta el register_op correspondiente en el backend_*.cpp)",
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

    // ------------------------------------------------------------------
    // Tabla de capacidades: que dtypes soporta cada (Op, Backend de
    // libreria). Custom no vive aca -- soporta kAllDTypes siempre, lo que
    // realmente compile es responsabilidad de config.yaml/KernelRegistry.
    //
    // Hoy la tabla arranca en cero salvo MatMul/CuBLAS, Softmax/CuDNN y
    // Randn/CuRAND, que son los unicos backends de libreria implementados.
    // Activar uno nuevo es una linea en table(), sin tocar Tensor.
    // ------------------------------------------------------------------
    using DTypeMask = uint8_t;

    inline constexpr DTypeMask dtype_bit(DType dt) { return static_cast<DTypeMask>(1u << std::to_underlying(dt)); }

    inline constexpr DTypeMask kAllDTypes =
        dtype_bit(DType::BF16) | dtype_bit(DType::FP32) | dtype_bit(DType::FP16) | dtype_bit(DType::INT8);

    // Todo el sistema corre en bf16 por defecto: es la garantia universal
    // a la que se puede caer si el backend elegido no soporta el dtype pedido.
    inline constexpr DType kDefaultDType = DType::BF16;

    class BackendCaps
    {
    public:
        static DTypeMask supported_dtypes(Op op, Backend backend)
        {
            if (backend == Backend::Custom) return kAllDTypes;
            return table()[index(op)][backend_index(backend)];
        }

        static bool supports(Op op, Backend backend, DType dt)
        {
            return (supported_dtypes(op, backend) & dtype_bit(dt)) != 0;
        }

        // Orden de preferencia para Backend::Auto. Custom siempre al final:
        // es el fallback garantizado, no la opcion rapida.
        static const std::vector<Backend>& preference_order(Op op)
        {
            static const std::vector matmul_order         = {Backend::CuBLAS, Backend::CuTensor, Backend::Custom};
            static const std::vector conv_order            = {Backend::CuDNN, Backend::Custom};
            static const std::vector norm_activation_order = {Backend::CuDNN, Backend::Custom};
            static const std::vector random_order          = {Backend::CuRAND, Backend::Custom};
            static const std::vector elementwise_order     = {Backend::CuTensor, Backend::Custom};
            static const std::vector custom_only           = {Backend::Custom};

            switch (op)
            {
            case Op::MatMul:    return matmul_order;
            case Op::Conv2D:    return conv_order;
            case Op::Gelu:
            case Op::Softmax:
            case Op::LayerNorm:
            case Op::BatchNorm: return norm_activation_order;
            case Op::Randn:
            case Op::Dropout:   return random_order;
            case Op::Add:
            case Op::Mul:       return elementwise_order;
            }
            return custom_only;
        }

    private:
        static constexpr size_t kNumLibBackends = 4; // CuBLAS, CuDNN, CuTensor, CuRAND, en ese orden fijo

        static size_t index(Op op) { return std::to_underlying(op); }

        static size_t backend_index(Backend b)
        {
            switch (b)
            {
            case Backend::CuBLAS:   return 0;
            case Backend::CuDNN:    return 1;
            case Backend::CuTensor: return 2;
            case Backend::CuRAND:   return 3;
            default:
                throw std::runtime_error(
                    std::format("BackendCaps: '{}' no tiene columna en la tabla de capacidades", backend_name(b)));
            }
        }

        static const std::array<std::array<DTypeMask, kNumLibBackends>, kNumOps>& table()
        {
            static const std::array<std::array<DTypeMask, kNumLibBackends>, kNumOps> t = []
            {
                std::array<std::array<DTypeMask, kNumLibBackends>, kNumOps> caps{};
                caps[std::to_underlying(Op::Add)][2 /*CuTensor*/] = dtype_bit(DType::BF16) | dtype_bit(DType::FP32) | dtype_bit(DType::FP16);
                caps[std::to_underlying(Op::Mul)][2 /*CuTensor*/] = dtype_bit(DType::BF16) | dtype_bit(DType::FP32) | dtype_bit(DType::FP16);
                // cublasGemmEx: bf16/fp32/fp16. INT8 usa el camino IMMA
                // (layout/alineacion distintos) -- se deja fuera a proposito,
                // matmul en int8 cae directo a Custom.
                caps[std::to_underlying(Op::MatMul)][0 /*CuBLAS*/] =
                    dtype_bit(DType::BF16) | dtype_bit(DType::FP32) | dtype_bit(DType::FP16);

                // cudnnSoftmaxForward: bf16/fp32/fp16. Gelu/LayerNorm quedan
                // en 0 -- cuDNN no tiene GELU nativo fuera de la Graph API, y
                // LayerNorm de transformer no mapea limpio a
                // cudnnNormalizationForward* (normaliza sobre eje CNN, no
                // sobre la ultima dimension). Auto cae a Custom para las dos.
                caps[std::to_underlying(Op::Softmax)][1 /*CuDNN*/] =
                    dtype_bit(DType::BF16) | dtype_bit(DType::FP32) | dtype_bit(DType::FP16);

                // curandGenerateNormal solo genera float nativamente; el
                // backend genera en float y castea en GPU aparte, asi que de
                // cara a OpsDispatch los 3 dtypes quedan igual de soportados.
                caps[std::to_underlying(Op::Randn)][3 /*CuRAND*/] =
                    dtype_bit(DType::BF16) | dtype_bit(DType::FP32) | dtype_bit(DType::FP16);

                return caps;
            }();
            return t;
        }
    };

    // ------------------------------------------------------------------
    // Punto unico de decision: "quiero <op> en <requested_dtype>, con
    // <requested_backend> (o Auto)" -> que backend y que dtype se usan de
    // verdad. Tensor llama aca antes de ejecutar cualquier operacion.
    //
    // Backend explicito: si soporta el dtype pedido, se usa tal cual: si no,
    // intenta el mismo backend en bf16, y si tampoco, cae a Custom.
    // Backend::Auto: recorre preference_order(op) y se queda con el primero
    // que soporte el dtype. Custom siempre esta al final y siempre
    // "soporta" cualquier dtype, asi que esto nunca lanza.
    // ------------------------------------------------------------------
    struct BackendResolution
    {
        Op op;
        Backend backend;           // backend real a usar (nunca Auto)
        DType compute_dtype;       // dtype con el que efectivamente se ejecuta
        bool dtype_was_downgraded; // true si compute_dtype != dtype pedido
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

            // Inalcanzable en la practica: Custom siempre esta en
            // preference_order() y siempre matchea arriba. Si esto dispara,
            // una Op nueva se agrego sin fila en BackendCaps::preference_order.
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
            // Ultimo recurso: kernel propio. El dtype pedido SI se respeta
            // aca -- lo que cambio fue el backend, no el dtype.
            if (warn)
            {
                std::cerr << std::format("[yadrakova] {}: {} no soporta dtype '{}' (ni en bf16), cayendo a kernel custom.\n",
                                          op_name(op), backend_name(backend), dtype_name(requested_dtype));
            }
            return {op, Backend::Custom, requested_dtype, false};
        }
    };
} // namespace yadrakova::core
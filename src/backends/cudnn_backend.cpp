#include "core/backend.hpp"
#include "core/backend_dispatch.hpp"
#include "core/stream.hpp"
#include <cudnn.h>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace yadrakova::core
{
    namespace
    {
        void cudnn_check(cudnnStatus_t st, const char* what)
        {
            if (st != CUDNN_STATUS_SUCCESS)
                throw std::runtime_error(std::string("cudnn_backend: ") + what + " fallo: " + cudnnGetErrorString(st));
        }

        // Mismo patron que cublas_handle_for en cublas_backend.cpp: un
        // handle por Stream, creado una sola vez, cacheado en
        // BackendContext. Este es el UNICO lugar del codebase que sabe que
        // BackendHandles::cudnn en realidad es un cudnnHandle_t.
        cudnnHandle_t cudnn_handle_for(Stream& stream)
        {
            BackendHandles& h = BackendContext::instance().handles_for(stream);
            if (!h.cudnn)
            {
                cudnnHandle_t handle = nullptr;
                cudnn_check(cudnnCreate(&handle), "cudnnCreate");
                h.cudnn = handle;
                h.destroy_cudnn = [](void* p) { cudnnDestroy(reinterpret_cast<cudnnHandle_t>(p)); };
            }
            cudnn_check(cudnnSetStream(reinterpret_cast<cudnnHandle_t>(h.cudnn), stream.raw()), "cudnnSetStream");
            return reinterpret_cast<cudnnHandle_t>(h.cudnn);
        }

        cudnnDataType_t to_cudnn_data_type(DType dt)
        {
            switch (dt)
            {
            case DType::BF16: return CUDNN_DATA_BFLOAT16;
            case DType::FP32: return CUDNN_DATA_FLOAT;
            case DType::FP16: return CUDNN_DATA_HALF;
            case DType::INT8:
                throw std::runtime_error("cudnn_backend: INT8 no esta soportado por este backend");
            }
            throw std::runtime_error("cudnn_backend: DType desconocido");
        }

        // Cache de descriptores por (rows, cols, dtype). Medido: crear y
        // destruir un cudnnTensorDescriptor_t en cada llamada le agregaba
        // ~7ms sobre ~8.7ms totales en el benchmark de 4096x4096 -- la
        // gran mayoria del costo NO era el softmax en si, era este setup
        // repetido para un shape que en la practica no cambia entre
        // llamadas (mismo batch/hidden size durante todo un training loop).
        //
        // Los descriptores no estan atados a un Stream (solo describen
        // layout/dtype), asi que el cache es global, no por-stream como
        // BackendContext.
        using DescKey = std::tuple<int, int, cudnnDataType_t>;

        cudnnTensorDescriptor_t desc_for(int rows, int cols, cudnnDataType_t dt)
        {
            static std::mutex mtx;
            static std::map<DescKey, cudnnTensorDescriptor_t> cache;

            std::lock_guard<std::mutex> lock(mtx);
            DescKey key{rows, cols, dt};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second;

            cudnnTensorDescriptor_t desc = nullptr;
            cudnn_check(cudnnCreateTensorDescriptor(&desc), "cudnnCreateTensorDescriptor");
            cudnn_check(
                cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, dt, rows, cols, 1, 1),
                "cudnnSetTensor4dDescriptor");

            cache.emplace(key, desc);
            return desc;
            // Nota: estos descriptores viven hasta que termina el proceso
            // (nunca se llama cudnnDestroyTensorDescriptor). Son ~pocos
            // bytes cada uno y la cantidad de shapes distintos que un
            // modelo real usa es chica y acotada -- no es un leak que vaya
            // a crecer sin limite como si cacheara por elemento o por
            // batch individual.
        }

        // Descriptor NCHW = [rows, cols, 1, 1]. Softmax con
        // CUDNN_SOFTMAX_MODE_INSTANCE normaliza sobre C,H,W combinados por
        // cada N -- con H=W=1 eso es exactamente "softmax sobre cols, por
        // cada fila", que es lo que Tensor::softmax espera (equivalente a
        // torch.softmax(x, dim=-1) para un tensor 2D).
        //
        // args = {&in_ptr, &out_ptr, &rows, &cols} -- mismo layout que
        // Tensor::softmax ya arma para Executor::execute.
        void softmax_via_cudnn(DType dtype, const std::vector<void*>& args, Stream& stream)
        {
            const void* in_ptr = *reinterpret_cast<void* const*>(args[0]);
            void* out_ptr = *reinterpret_cast<void* const*>(args[1]);
            const auto rows = static_cast<int>(*reinterpret_cast<const int64_t*>(args[2]));
            const auto cols = static_cast<int>(*reinterpret_cast<const int64_t*>(args[3]));

            cudnnHandle_t handle = cudnn_handle_for(stream);
            cudnnDataType_t cudnn_dt = to_cudnn_data_type(dtype);
            cudnnTensorDescriptor_t desc = desc_for(rows, cols, cudnn_dt);

            const float alpha = 1.0f, beta = 0.0f;
            cudnn_check(
                cudnnSoftmaxForward(
                    handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                    &alpha, desc, in_ptr, &beta, desc, out_ptr),
                "cudnnSoftmaxForward");
        }

        // Registro estatico -- mismo patron que cublas_registration_instance
        // en cublas_backend.cpp. Necesita --whole-archive para que el
        // linker no lo descarte (nadie llama a este simbolo directamente).
        struct CuDNNRegistration
        {
            CuDNNRegistration()
            {
                auto& reg = BackendRegistry::instance();
                for (DType dt : {DType::BF16, DType::FP32, DType::FP16})
                {
                    reg.register_op(Op::Softmax, Backend::CuDNN, dt,
                        [dt](const std::vector<void*>& args, Stream& stream)
                        {
                            softmax_via_cudnn(dt, args, stream);
                        });
                }
                // Gelu y LayerNorm NO se registran aqui a proposito -- ver
                // el comentario en backend_caps.hpp. Mientras su fila en
                // BackendCaps siga en 0, OpsDispatch nunca va a intentar
                // invocarlos por este backend, asi que no hace falta un
                // "stub que lanza".
            }
        };

        const CuDNNRegistration cudnn_registration_instance;
    } // namespace
} // namespace yadrakova::core
#include "core/backend.hpp"
#include "core/backend_dispatch.hpp"
#include "core/stream.hpp"
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace yadrakova::core
{
    namespace
    {
        // Devuelve el cublasHandle_t para este Stream, creandolo (una sola
        // vez por stream) si todavia no existe. BackendHandles::cublas se
        // guarda como void* -- aqui es el UNICO lugar del codebase que sabe
        // que ese void* en realidad es un cublasHandle_t.
        cublasHandle_t cublas_handle_for(Stream& stream)
        {
            BackendHandles& h = BackendContext::instance().handles_for(stream);
            if (!h.cublas)
            {
                cublasHandle_t handle = nullptr;
                cublasStatus_t st = cublasCreate(&handle);
                if (st != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error(
                        "cublas_backend: cublasCreate fallo (status " + std::to_string(static_cast<int>(st)) + ")");

                h.cublas = handle;
                h.destroy_cublas = [](void* p) { cublasDestroy(reinterpret_cast<cublasHandle_t>(p)); };
            }
            // El handle vive mientras viva el stream, pero el stream con el
            // que se lo asocia se re-confirma en cada llamada -- barato y
            // evita bugs si algun dia el mismo handle se reutiliza distinto.
            cublasSetStream(reinterpret_cast<cublasHandle_t>(h.cublas), stream.raw());
            return reinterpret_cast<cublasHandle_t>(h.cublas);
        }

        cudaDataType_t to_cuda_data_type(DType dt)
        {
            switch (dt)
            {
            case DType::BF16: return CUDA_R_16BF;
            case DType::FP32: return CUDA_R_32F;
            case DType::FP16: return CUDA_R_16F;
            case DType::INT8:
                throw std::runtime_error("cublas_backend: INT8 no esta soportado por este backend");
            }
            throw std::runtime_error("cublas_backend: DType desconocido");
        }

        // args tiene el MISMO layout que Tensor::matmul ya arma para
        // Executor::execute: {&a_ptr, &b_ptr, &c_ptr, &M, &N, &K}. Por eso
        // Tensor no necesita saber si termina en un kernel .cu o en cuBLAS.
        void matmul_via_cublas(DType dtype, const std::vector<void*>& args, Stream& stream)
        {
            const void* a = *reinterpret_cast<void* const*>(args[0]);
            const void* b = *reinterpret_cast<void* const*>(args[1]);
            void* c = *reinterpret_cast<void* const*>(args[2]);
            const auto M = static_cast<int>(*reinterpret_cast<const int64_t*>(args[3]));
            const auto N = static_cast<int>(*reinterpret_cast<const int64_t*>(args[4]));
            const auto K = static_cast<int>(*reinterpret_cast<const int64_t*>(args[5]));

            cublasHandle_t handle = cublas_handle_for(stream);
            cudaDataType_t cuda_dt = to_cuda_data_type(dtype);

            const float alpha = 1.0f, beta = 0.0f;

            // A, B, C estan almacenados row-major (como numpy/pytorch).
            // cuBLAS asume column-major. En vez de transponer datos,
            // calculamos C^T = B^T * A^T -- el resultado en memoria es
            // exactamente C row-major, sin copiar nada. Es el mismo truco
            // que ya usabas en el helper del test.
            cublasStatus_t st = cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                b, cuda_dt, N,
                a, cuda_dt, K,
                &beta,
                c, cuda_dt, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            if (st != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error(
                    "cublas_backend: cublasGemmEx fallo (status " + std::to_string(static_cast<int>(st)) + ")");
        }

        // Registro estatico: corre al cargar la libreria, antes de main().
        // Igual que KernelRegistry, nadie llama a este simbolo directamente
        // -- por eso los ejecutables que enlazan yadrakova_core necesitan
        // --whole-archive (ya lo tenes en el CMakeLists para los tests).
        struct CuBLASRegistration
        {
            CuBLASRegistration()
            {
                auto& reg = BackendRegistry::instance();
                for (DType dt : {DType::BF16, DType::FP32, DType::FP16})
                {
                    reg.register_op(Op::MatMul, Backend::CuBLAS, dt,
                        [dt](const std::vector<void*>& args, Stream& stream)
                        {
                            matmul_via_cublas(dt, args, stream);
                        });
                }
            }
        };

        const CuBLASRegistration cublas_registration_instance;
    } // namespace
} // namespace yadrakova::core
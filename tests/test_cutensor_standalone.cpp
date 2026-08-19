#include <cutensor.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check_cutensor(cutensorStatus_t status, const char* what)
{
    if (status != CUTENSOR_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(what) +
            " failed. cuTENSOR status = " +
            std::to_string(static_cast<int>(status))
        );
    }
}

static void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) +
            " failed: " +
            cudaGetErrorString(status)
        );
    }
}

int main()
{
    try {
        std::cout << "============================================\n";
        std::cout << "   STANDALONE cuTENSOR 2.7 TEST\n";
        std::cout << "   NO YADRA KOVA\n";
        std::cout << "============================================\n\n";

        // ------------------------------------------------------------
        // 1. CUDA information
        // ------------------------------------------------------------

        int runtime_version = 0;
        int driver_version = 0;

        check_cuda(
            cudaRuntimeGetVersion(&runtime_version),
            "cudaRuntimeGetVersion"
        );

        check_cuda(
            cudaDriverGetVersion(&driver_version),
            "cudaDriverGetVersion"
        );

        cudaDeviceProp prop{};

        check_cuda(
            cudaGetDeviceProperties(&prop, 0),
            "cudaGetDeviceProperties"
        );

        std::cout << "CUDA runtime : "
                  << runtime_version / 1000
                  << "."
                  << (runtime_version % 1000) / 10
                  << "\n";

        std::cout << "CUDA driver  : "
                  << driver_version / 1000
                  << "."
                  << (driver_version % 1000) / 10
                  << "\n";

        std::cout << "GPU          : "
                  << prop.name
                  << "\n";

        std::cout << "Compute Cap. : "
                  << prop.major
                  << "."
                  << prop.minor
                  << "\n";

        std::cout << "cuTENSOR     : "
                  << cutensorGetVersion()
                  << "\n\n";


        // ------------------------------------------------------------
        // 2. Create cuTENSOR handle
        // ------------------------------------------------------------

        std::cout << "[1] Creating cuTENSOR handle...\n";

        cutensorHandle_t handle{};

        check_cutensor(
            cutensorCreate(&handle),
            "cutensorCreate"
        );

        std::cout << "    OK\n";


        // ------------------------------------------------------------
        // 3. Allocate tensors
        //
        // A, B, C = 32 x 32 FP32
        // C = A + B
        // ------------------------------------------------------------

        constexpr int64_t rows = 32;
        constexpr int64_t cols = 32;
        constexpr int64_t elements = rows * cols;

        std::vector<float> h_a(elements, 1.0f);
        std::vector<float> h_b(elements, 2.0f);
        std::vector<float> h_c(elements, 0.0f);

        float* d_a = nullptr;
        float* d_b = nullptr;
        float* d_c = nullptr;

        std::cout << "[2] Allocating CUDA memory...\n";

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_a),
                elements * sizeof(float)
            ),
            "cudaMalloc(d_a)"
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_b),
                elements * sizeof(float)
            ),
            "cudaMalloc(d_b)"
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_c),
                elements * sizeof(float)
            ),
            "cudaMalloc(d_c)"
        );

        check_cuda(
            cudaMemcpy(
                d_a,
                h_a.data(),
                elements * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy A"
        );

        check_cuda(
            cudaMemcpy(
                d_b,
                h_b.data(),
                elements * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy B"
        );

        std::cout << "    OK\n";


        // ------------------------------------------------------------
        // 4. Tensor descriptors
        // ------------------------------------------------------------

        std::cout << "[3] Creating tensor descriptors...\n";

        constexpr int32_t num_modes = 2;

        const int64_t extent[2] = {
            rows,
            cols
        };

        const int64_t stride[2] = {
            cols,
            1
        };

        const int32_t modes[2] = {
            0,
            1
        };

        constexpr uint32_t alignment = 16;

        cutensorTensorDescriptor_t desc_a{};
        cutensorTensorDescriptor_t desc_b{};
        cutensorTensorDescriptor_t desc_c{};

        check_cutensor(
            cutensorCreateTensorDescriptor(
                handle,
                &desc_a,
                num_modes,
                extent,
                stride,
                CUTENSOR_R_32F,
                alignment
            ),
            "cutensorCreateTensorDescriptor(A)"
        );

        check_cutensor(
            cutensorCreateTensorDescriptor(
                handle,
                &desc_b,
                num_modes,
                extent,
                stride,
                CUTENSOR_R_32F,
                alignment
            ),
            "cutensorCreateTensorDescriptor(B)"
        );

        check_cutensor(
            cutensorCreateTensorDescriptor(
                handle,
                &desc_c,
                num_modes,
                extent,
                stride,
                CUTENSOR_R_32F,
                alignment
            ),
            "cutensorCreateTensorDescriptor(C)"
        );

        std::cout << "    OK\n";


        // ------------------------------------------------------------
        // 5. Elementwise binary operation
        //
        // C = A + B
        // ------------------------------------------------------------

        std::cout << "[4] Creating elementwise operation descriptor...\n";

        cutensorOperationDescriptor_t operation{};

        check_cutensor(
            cutensorCreateElementwiseBinary(
                handle,

                &operation,

                desc_a,
                modes,
                CUTENSOR_OP_IDENTITY,

                desc_b,
                modes,
                CUTENSOR_OP_IDENTITY,

                desc_c,
                modes,
                CUTENSOR_OP_ADD,

                CUTENSOR_COMPUTE_DESC_32F
            ),
            "cutensorCreateElementwiseBinary"
        );

        std::cout << "    OK\n";


        // ------------------------------------------------------------
        // 6. Plan preference
        // ------------------------------------------------------------

        std::cout << "[5] Creating plan preference...\n";

        cutensorPlanPreference_t preference{};

        check_cutensor(
            cutensorCreatePlanPreference(
                handle,
                &preference,
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE
            ),
            "cutensorCreatePlanPreference"
        );

        std::cout << "    OK\n";


        // ------------------------------------------------------------
        // 7. Workspace estimation
        // ------------------------------------------------------------

        std::cout << "[6] Estimating workspace...\n";

        uint64_t workspace_size = 0;

        check_cutensor(
            cutensorEstimateWorkspaceSize(
                handle,
                operation,
                preference,
                CUTENSOR_WORKSPACE_DEFAULT,
                &workspace_size
            ),
            "cutensorEstimateWorkspaceSize"
        );

        std::cout << "    Workspace = "
                  << workspace_size
                  << " bytes\n";


        // ------------------------------------------------------------
        // 8. Create plan
        // ------------------------------------------------------------

        std::cout << "[7] Creating plan...\n";
        std::cout.flush();

        cutensorPlan_t plan{};

        check_cutensor(
            cutensorCreatePlan(
                handle,
                &plan,
                operation,
                preference,
                workspace_size
            ),
            "cutensorCreatePlan"
        );

        std::cout << "    PLAN CREATED SUCCESSFULLY!\n";


        // ------------------------------------------------------------
        // 9. Execute
        // ------------------------------------------------------------

        std::cout << "[8] Executing A + B...\n";

        float alpha = 1.0f;
        float beta  = 1.0f;

        check_cutensor(
            cutensorElementwiseBinaryExecute(
                handle,
                plan,
                &alpha,
                d_a,
                &beta,
                d_b,
                d_c,
                nullptr
            ),
            "cutensorElementwiseBinaryExecute"
        );

        check_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize"
        );

        std::cout << "    OK\n";


        // ------------------------------------------------------------
        // 10. Verify
        // ------------------------------------------------------------

        check_cuda(
            cudaMemcpy(
                h_c.data(),
                d_c,
                elements * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy C"
        );

        std::cout << "[9] Result:\n";
        std::cout << "    C[0] = "
                  << h_c[0]
                  << "\n";

        if (std::abs(h_c[0] - 3.0f) < 1e-5f) {
            std::cout << "\n============================================\n";
            std::cout << "             TEST PASSED\n";
            std::cout << "============================================\n";
        } else {
            std::cout << "\n============================================\n";
            std::cout << "             TEST FAILED\n";
            std::cout << "============================================\n";
        }


        // ------------------------------------------------------------
        // 11. Cleanup
        // ------------------------------------------------------------

        cutensorDestroyPlan(plan);
        cutensorDestroyPlanPreference(preference);
        cutensorDestroyOperationDescriptor(operation);

        cutensorDestroyTensorDescriptor(desc_a);
        cutensorDestroyTensorDescriptor(desc_b);
        cutensorDestroyTensorDescriptor(desc_c);

        cutensorDestroy(handle);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\nEXCEPTION: "
                  << e.what()
                  << "\n";

        return 1;
    }
}
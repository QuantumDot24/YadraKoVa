#include "core/backend.hpp"
#include "core/backend_dispatch.hpp"
#include "core/stream.hpp"
#include <cutensor.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace yadrakova::core
{
namespace
{
void cutensor_check(cutensorStatus_t st, const char* what)
{
    if (st != CUTENSOR_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cutensor_backend: ") + what + " failed with status " + std::to_string(static_cast<int>(st)));
}

cutensorHandle_t cutensor_handle_for(Stream& stream)
{
    BackendHandles& h = BackendContext::instance().handles_for(stream);
    if (!h.cutensor)
    {
        cutensorHandle_t handle = nullptr;
        cutensor_check(cutensorCreate(&handle), "cutensorCreate");
        h.cutensor = handle;
        h.destroy_cutensor = [](void* p) {
            cutensorDestroy(reinterpret_cast<cutensorHandle_t>(p));
        };
    }
    return reinterpret_cast<cutensorHandle_t>(h.cutensor);
}

cutensorDataType_t to_cutensor_data_type(DType dt)
{
    switch (dt)
    {
    case DType::BF16: return CUTENSOR_R_16BF;
    case DType::FP32: return CUTENSOR_R_32F;
    case DType::FP16: return CUTENSOR_R_16F;
    default:
        throw std::runtime_error("cutensor_backend: Unsupported DType for elementwise");
    }
}

void elementwise_binary_via_cutensor(Op op, DType dtype, const std::vector<void*>& args, Stream& stream)
{
    // Extraer argumentos tal como los envía Tensor<T>::add/mul
    const void* a_ptr = *reinterpret_cast<void* const*>(args[0]);
    const void* b_ptr = *reinterpret_cast<void* const*>(args[1]);
    void* c_ptr = *reinterpret_cast<void* const*>(args[2]);
    const auto n = static_cast<int64_t>(*reinterpret_cast<const int64_t*>(args[3]));

    cutensorHandle_t handle = cutensor_handle_for(stream);
    cutensorDataType_t cutensor_dt = to_cutensor_data_type(dtype);

    // Forzar computación en FP32 internamente para máxima compatibilidad y estabilidad del planificador
    cutensorComputeDescriptor_t compute_desc = CUTENSOR_COMPUTE_DESC_32F;

    // Tratamos el tensor contiguo como 1D. Con el fix de /STACK:16MB en CMake,
    // el planificador de cuTENSOR maneja esto sin stack overflow.
    const int32_t num_modes = 1;
    const int64_t extent[1] = {n};
    const int64_t stride[1] = {1};
    const int32_t mode[1] = {0};
    const uint32_t alignment = 16;

    cutensorTensorDescriptor_t descA, descB, descC;
    cutensor_check(cutensorCreateTensorDescriptor(handle, &descA, num_modes, extent, stride, cutensor_dt, alignment), "descA");
    cutensor_check(cutensorCreateTensorDescriptor(handle, &descB, num_modes, extent, stride, cutensor_dt, alignment), "descB");
    cutensor_check(cutensorCreateTensorDescriptor(handle, &descC, num_modes, extent, stride, cutensor_dt, alignment), "descC");

    cutensorOperator_t opAC = (op == Op::Add) ? CUTENSOR_OP_ADD : CUTENSOR_OP_MUL;

    cutensorOperationDescriptor_t op_desc;
    cutensor_check(cutensorCreateElementwiseBinary(
        handle, &op_desc,
        descA, mode, CUTENSOR_OP_IDENTITY,
        descB, mode, CUTENSOR_OP_IDENTITY,
        descC, mode, opAC,
        compute_desc
    ), "cutensorCreateElementwiseBinary");

    cutensorPlanPreference_t plan_pref;
    cutensor_check(cutensorCreatePlanPreference(handle, &plan_pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE), "cutensorCreatePlanPreference");

    // CLAVE: Estimar workspace con un límite generoso (1MB) para evitar que el planificador
    // entre en bucles de búsqueda que causaban el stack overflow en Windows.
    uint64_t workspace_size = 1024 * 1024;
    cutensor_check(cutensorEstimateWorkspaceSize(handle, op_desc, plan_pref, CUTENSOR_WORKSPACE_DEFAULT, &workspace_size), "cutensorEstimateWorkspaceSize");

    cutensorPlan_t plan;
    cutensor_check(cutensorCreatePlan(handle, &plan, op_desc, plan_pref, workspace_size), "cutensorCreatePlan");

    float alpha = 1.0f;
    float gamma = 1.0f;

    cutensor_check(cutensorElementwiseBinaryExecute(
        handle, plan, &alpha, a_ptr, &gamma, b_ptr, c_ptr, stream.raw()
    ), "cutensorElementwiseBinaryExecute");

    // Limpieza
    cutensorDestroyPlan(plan);
    cutensorDestroyPlanPreference(plan_pref);
    cutensorDestroyOperationDescriptor(op_desc);
    cutensorDestroyTensorDescriptor(descA);
    cutensorDestroyTensorDescriptor(descB);
    cutensorDestroyTensorDescriptor(descC);
}

struct CuTensorRegistration
{
    CuTensorRegistration()
    {
        auto& reg = BackendRegistry::instance();
        for (DType dt : {DType::BF16, DType::FP32, DType::FP16})
        {
            reg.register_op(Op::Add, Backend::CuTensor, dt,
                [dt](const std::vector<void*>& args, Stream& stream) {
                    elementwise_binary_via_cutensor(Op::Add, dt, args, stream);
                });

            reg.register_op(Op::Mul, Backend::CuTensor, dt,
                [dt](const std::vector<void*>& args, Stream& stream) {
                    elementwise_binary_via_cutensor(Op::Mul, dt, args, stream);
                });
        }
    }
};
const CuTensorRegistration cutensor_registration_instance;

} // namespace
} // namespace yadrakova::core
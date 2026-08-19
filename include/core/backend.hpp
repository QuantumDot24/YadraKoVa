#pragma once
#include <cstdint>
#include <stdexcept>

namespace yadrakova::core
{
    // Motor de computo que ejecuta una operacion.
    //
    // Custom = tus kernels .cu via KernelRegistry/Executor (el camino que ya
    // existe hoy). Es el UNICO backend que garantiza soporte total de dtypes
    // (los que hayas compilado en config.yaml) y por eso siempre es el
    // fallback final de cualquier cadena de resolucion.
    //
    // Los demas son las librerias NVIDIA que se van a ir enchufando una por
    // una. Que un backend exista en este enum NO significa que ya este
    // implementado -- ver backend_caps.hpp (la tabla de capacidades) para
    // saber cuales estan realmente activos.
    enum class Backend : uint8_t
    {
        Auto = 0,   // deja que OpsDispatch elija el mejor backend disponible para la op+dtype pedidos
        Custom,     // kernel propio via Executor/KernelRegistry (siempre disponible)
        CuBLAS,     // GEMM / GEMM batched
        CuDNN,      // convoluciones, normalizaciones, activaciones fusionadas
        CuTensor,   // contracciones generales (einsum), permutaciones/transposiciones, reducciones
        CuRAND,     // generacion de numeros aleatorios (randn, dropout mask, init de pesos)
    };

    inline const char* backend_name(Backend b)
    {
        switch (b)
        {
        case Backend::Auto:     return "auto";
        case Backend::Custom:   return "custom";
        case Backend::CuBLAS:   return "cublas";
        case Backend::CuDNN:    return "cudnn";
        case Backend::CuTensor: return "cutensor";
        case Backend::CuRAND:   return "curand";
        }
        throw std::runtime_error("backend_name: Backend desconocido");
    }

    // Identificador estable de operacion, independiente del nombre de kernel
    // .cu (ese sigue siendo un detalle interno del backend Custom, ej.
    // "matmul_wmma"). Esto es lo que usan BackendCaps/BackendRegistry como
    // clave para decidir y despachar.
    //
    // Al agregar una op nueva: agregarla aqui, en op_name(), y darle una fila
    // en BackendCaps::preference_order (backend_caps.hpp).
    enum class Op : uint8_t
    {
        MatMul = 0,
        Gelu,
        Softmax,
        LayerNorm,
        BatchNorm,
        Conv2D,
        Randn,
        Dropout,
        Add,
        Mul,
        Contiguous
    };

    inline constexpr size_t kNumOps = 11;

    inline const char* op_name(Op op)
    {
        switch (op)
        {
        case Op::MatMul:    return "matmul";
        case Op::Gelu:      return "gelu";
        case Op::Softmax:   return "softmax";
        case Op::LayerNorm: return "layer_norm";
        case Op::BatchNorm: return "batch_norm";
        case Op::Conv2D:    return "conv2d";
        case Op::Randn:     return "randn";
        case Op::Dropout:   return "dropout";
        case Op::Add:       return "add";
        case Op::Mul:       return "mul";
            case Op::Contiguous: return "contiguous";
        }
        throw std::runtime_error("op_name: Op desconocido");
    }
} // namespace yadrakova::core
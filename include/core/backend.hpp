#pragma once
#include <cstdint>
#include <stdexcept>

namespace yadrakova::core
{
    enum class Backend : uint8_t
    {
        Auto = 0,
        Custom,
        CuBLAS,
        CuDNN,
        CuTensor,
        CuRAND,
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
} // namespace yadrakova::core

#include "generated/ops/ops_metadata.hpp"
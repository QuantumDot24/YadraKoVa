#pragma once

namespace yadrakova::core {

    enum class OpKind {
        Gemm,
        Gemv,          // <--- Nuevo
        Elementwise,
        Reduction,
        Attention,     // <--- Nuevo
        Conv2D,        // <--- Nuevo
        GraphReplay,
        Memcpy,
        Other
    };

} // namespace yadrakova::core
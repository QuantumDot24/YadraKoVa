#pragma once

namespace yadrakova::core {

    enum class OpKind {
        Gemm,
        Gemv,
        Elementwise,
        Reduction,
        Attention,
        Conv2D,
        GraphReplay,
        Memcpy,
        Other
    };

} // namespace yadrakova::core
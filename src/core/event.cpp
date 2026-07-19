#include "core/event.h"

namespace yadrakova::core {

    float elapsed_ms(const Event& start, const Event& end) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.raw(), end.raw()));
        return ms;
    }

} // namespace yadrakova::core
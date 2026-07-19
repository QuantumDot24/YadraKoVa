#include "core/stream.h"

namespace yadrakova::core {

    Stream& default_stream() {
        static Stream instance(/*non_blocking=*/true);
        return instance;
    }

} // namespace yadrakova::core
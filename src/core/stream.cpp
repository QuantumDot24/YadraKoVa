// stream.cpp
#include "core/stream.hpp"
#include "core/cuda_error.hpp"

namespace yadrakova::core {

    Stream::Stream(bool non_blocking) {
        unsigned int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
        CUDA_CHECK(cudaStreamCreateWithFlags(&handle_, flags));
    }

    Stream::~Stream() {
        if (handle_) cudaStreamDestroy(handle_);
    }

    Stream::Stream(Stream&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Stream& Stream::operator=(Stream&& other) noexcept {
        if (this != &other) {
            if (handle_) cudaStreamDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void Stream::synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(handle_));
    }

    bool Stream::is_done() const {
        cudaError_t err = cudaStreamQuery(handle_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        CUDA_CHECK(err);
        return false;
    }

    Stream& default_stream() {
        static Stream instance(/*non_blocking=*/true);
        return instance;
    }

} // namespace yadrakova::core
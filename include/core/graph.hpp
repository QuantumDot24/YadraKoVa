#pragma once
#include "core/stream.hpp"
#include "core/memory.hpp"
#include <string>

namespace yadrakova::core {

    class Graph {
    public:
        explicit Graph(std::string name);
        ~Graph();

        Graph(const Graph&) = delete;
        Graph& operator=(const Graph&) = delete;

        void begin_capture(Stream& stream, MemoryPool& pool = default_pool(), bool strict = true);
        void end_capture(Stream& stream);
        void launch(Stream& stream);
        void update_from(Graph& new_topology);
        void abort_capture(Stream& stream) noexcept;

        const std::string& name() const { return name_; }
        bool is_instantiated() const { return graph_exec_ != nullptr; }

    private:
        void instantiate();

        std::string name_;
        cudaGraph_t graph_ = nullptr;
        cudaGraphExec_t graph_exec_ = nullptr;
        bool capturing_ = false;
        bool strict_ = true;
        MemoryPool* pool_ = nullptr;
        size_t bytes_reserved_before_ = 0;
    };

} // namespace yadrakova::core
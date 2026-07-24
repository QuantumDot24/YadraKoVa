#pragma once
#include "core/graph.hpp"
#include "core/stream.hpp"
#include "telemetry/telemetry.hpp"
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>

namespace yadrakova::core
{
    class GraphManager
    {
    public:
        explicit GraphManager(MemoryPool& pool = default_pool());

        Graph& get_or_create(const std::string& name);

        Graph& get(const std::string& name);

        bool has(const std::string& name) const;

        Stream& stream_for(const std::string& name);

        void launch(const std::string& name, Stream& stream);

        void launch(const std::string& name);


        Graph& capture(const std::string& name, bool strict, const std::function<void()>& fn);

        void synchronize_all();

        size_t count() const { return graphs_.size(); }
        size_t stream_count() const { return owned_streams_.size(); }

        MemoryPool& pool() { return pool_; }
        Telemetry& telemetry() { return telemetry_; }
        const Telemetry& telemetry() const { return telemetry_; }
        template <typename F>
         auto time(std::string label, OpKind kind, Stream& stream, F&& fn,
                   bool from_graph_replay = false,
                   uint64_t bytes_moved = 0, double gflops = 0.0)
             -> decltype(fn())
        {
            return telemetry_.time(std::move(label), kind, stream, std::forward<F>(fn),
                                    from_graph_replay, bytes_moved, gflops);
        }

        template <typename F>
        void benchmark(std::string group_label, size_t ops_per_iteration,
                       int replays, F&& fn)
        {
            telemetry_.benchmark(std::move(group_label), ops_per_iteration, replays,
                                  std::forward<F>(fn));
        }
    private:
        MemoryPool& pool_;
        std::unordered_map<std::string, std::unique_ptr<Graph>> graphs_;
        std::unordered_map<std::string, std::unique_ptr<Stream>> owned_streams_;
        Telemetry telemetry_;
    };
} // namespace yadrakova::core

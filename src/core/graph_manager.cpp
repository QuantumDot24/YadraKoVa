#include "core/graph_manager.hpp"

#include <ranges>
#include <stdexcept>

namespace yadrakova::core
{
    GraphManager::GraphManager(MemoryPool& pool) : pool_(pool)
    {
    }

    Graph& GraphManager::get_or_create(const std::string& name)
    {
        auto it = graphs_.find(name);
        if (it != graphs_.end()) return *it->second;

        auto [inserted_it, ok] = graphs_.emplace(name, std::make_unique<Graph>(name));
        return *inserted_it->second;
    }

    Graph& GraphManager::get(const std::string& name)
    {
        auto it = graphs_.find(name);
        if (it == graphs_.end())
        {
            throw std::runtime_error("GraphManager: no Graph named '" + name + "' exists.");
        }
        return *it->second;
    }

    bool GraphManager::has(const std::string& name) const
    {
        return graphs_.find(name) != graphs_.end();
    }

    Stream& GraphManager::stream_for(const std::string& name)
    {
        auto it = owned_streams_.find(name);
        if (it == owned_streams_.end())
        {
            it = owned_streams_.emplace(name, std::make_unique<Stream>()).first;
        }
        return *it->second;
    }

    void GraphManager::launch(const std::string& name, Stream& stream)
    {
        get(name).launch(stream);
    }

    void GraphManager::launch(const std::string& name)
    {
        launch(name, stream_for(name));
    }

    Graph& GraphManager::capture(const std::string& name, bool strict, const std::function<void()>& fn)
    {
        Graph& g = get_or_create(name);
        Stream& s = stream_for(name);
        g.begin_capture(s, pool_, strict);
        try
        {
            fn();
            g.end_capture(s);
        }
        catch (...)
        {
            g.abort_capture(s);
            throw;
        }
        return g;
    }

    void GraphManager::synchronize_all()
    {
        for (const auto& stream : owned_streams_ | std::views::values)
        {
            stream->synchronize();
        }
    }
} // namespace yadrakova::core

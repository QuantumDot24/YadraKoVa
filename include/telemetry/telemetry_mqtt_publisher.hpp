#pragma once
#include <string>
#include <vector>

namespace yadrakova::core
{
    class TelemetryMqttPublisher
    {
    public:
        TelemetryMqttPublisher(std::string server_address, std::string client_id);
        bool publish_lines(const std::vector<std::string>& json_lines,
                           const std::string& topic) const;

    private:
        std::string server_address_;
        std::string client_id_;
    };
} // namespace yadrakova::core

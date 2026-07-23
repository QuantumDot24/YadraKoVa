// telemetry_mqtt_publisher.cpp
#include "telemetry/telemetry_mqtt_publisher.hpp"
#include <mqtt/client.h>
#include <iostream>

namespace yadrakova::core {

    TelemetryMqttPublisher::TelemetryMqttPublisher(std::string server_address,
                                                     std::string client_id)
        : server_address_(std::move(server_address)),
          client_id_(std::move(client_id)) {}

    bool TelemetryMqttPublisher::publish_lines(const std::vector<std::string>& json_lines,
                                                 const std::string& topic) {
        if (json_lines.empty()) {
            return true; // nada que mandar, no es un error
        }

        mqtt::client client(server_address_, client_id_);

        mqtt::connect_options conn_opts;
        conn_opts.set_clean_session(true);
        conn_opts.set_keep_alive_interval(20);

        try {
            client.connect(conn_opts);

            for (const auto& line : json_lines) {
                mqtt::message_ptr msg = mqtt::make_message(topic, line);
                msg->set_qos(0); // fire-and-forget, ver comentario en el .hpp
                client.publish(msg);
            }

            client.disconnect();
        } catch (const mqtt::exception& exc) {
            // No relanzamos: perder telemetria no deberia tumbar el
            // benchmark. El caller ya tiene las lineas en memoria si
            // quiere hacer algo mas con ellas (loggear a disco, etc.).
            std::cerr << "TelemetryMqttPublisher: error MQTT -- " << exc.what() << "\n";
            return false;
        }

        return true;
    }

} // namespace yadrakova::core
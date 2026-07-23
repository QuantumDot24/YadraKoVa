#pragma once
#include <string>
#include <vector>

namespace yadrakova::core {

    // Publisher MQTT desacoplado de Telemetry a proposito -- Telemetry no
    // sabe que existe MQTT, este publisher solo sabe mandar strings a un
    // topic. Asi el dia que quieras otro transporte (websocket, archivo,
    // lo que sea) no tocas Telemetry para nada.
    //
    // Uso tipico, despues del benchmark completo (nunca dentro del loop
    // de replay -- eso rompe el timing que estas midiendo):
    //
    //   TelemetryMqttPublisher publisher("tcp://localhost:1883", "yadrakova_benchmark");
    //   publisher.publish_lines(manager.telemetry().export_all_as_json_lines(),
    //                            "yadrakova/telemetry");
    //
    // QoS 0 (fire-and-forget) a proposito: para telemetria de benchmark no
    // necesitamos garantia de entrega, y es el modo mas rapido/liviano.
    class TelemetryMqttPublisher {
    public:
        // server_address: ej "tcp://localhost:1883"
        // client_id: identificador unico de este cliente ante el broker.
        //            Si corres varias instancias a la vez, dales client_id
        //            distintos o el broker las va a ir desconectando entre si.
        TelemetryMqttPublisher(std::string server_address, std::string client_id);

        // Conecta, publica cada linea como un mensaje independiente en
        // `topic`, y desconecta. Si algo falla (broker caido, etc.), lo
        // reporta por stderr y retorna false -- no tira excepcion, porque
        // perder telemetria no deberia tumbar tu benchmark.
        bool publish_lines(const std::vector<std::string>& json_lines,
                            const std::string& topic);

    private:
        std::string server_address_;
        std::string client_id_;
    };

} // namespace yadrakova::core
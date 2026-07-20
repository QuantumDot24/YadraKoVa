#pragma once
#include "core/graph.hpp"
#include "core/stream.hpp"
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>

namespace yadrakova::core {

// Punto unico de orquestacion para multiples Graph con nombre --
// pensado exactamente para el caso GAN: "G_forward", "G_backward",
// "D_forward", "D_backward" viven aqui, y el loop de training (en
// train/, no en core/) decide el orden/alternancia lanzandolos por
// nombre. El core no sabe ni le importa que es una GAN -- solo
// gestiona el ciclo de vida de N grafos nombrados.
//
// Multi-stream: ademas de poder lanzar un grafo pasandole tu propio
// Stream (como antes), el manager puede asignarle a cada grafo nombrado
// un Stream propio, creado lazy la primera vez que se pide. Eso permite
// correr, por ejemplo, "training" y "telemetry" en paralelo sin que el
// caller tenga que gestionar los Streams a mano.
class GraphManager {
public:
    explicit GraphManager(MemoryPool& pool = default_pool());

    // Crea (o retorna, si ya existe) el Graph con este nombre.
    Graph& get_or_create(const std::string& name);

    Graph& get(const std::string& name);

    bool has(const std::string& name) const;

    // Devuelve el Stream dedicado de este nombre, creandolo si no existe.
    // El nombre aqui no tiene que coincidir con un Graph -- puedes pedir
    // stream_for("telemetry") aunque el Graph se llame distinto, aunque
    // en el caso comun ambos nombres van a coincidir.
    Stream& stream_for(const std::string& name);

    // Atajo: lanza un grafo ya instanciado por nombre, en el Stream
    // que el caller pase explicitamente (comportamiento original).
    void launch(const std::string& name, Stream& stream);

    // Overload nuevo: lanza el grafo `name` en SU PROPIO Stream dedicado
    // (creado lazy via stream_for). Asi "G_forward" y "telemetry" pueden
    // correr en paralelo sin que el caller gestione streams a mano.
    void launch(const std::string& name);

    // Encapsula el patron begin_capture/fn()/end_capture usando el
    // stream dedicado de `name` (via stream_for). `fn` debe encolar
    // exactamente las operaciones que quieres capturar -- kernels,
    // memcpys async, etc. -- sobre ese mismo stream dedicado.
    //
    // Ejemplo de uso desde train/:
    //   manager.capture("G_forward", /*strict=*/true, [&] {
    //       generator.forward(z, out, manager.stream_for("G_forward"));
    //   });
    //   ...
    //   manager.launch("G_forward"); // reusa el mismo stream dedicado
    Graph& capture(const std::string& name, bool strict, const std::function<void()>& fn);

    // Bloquea hasta que todos los streams dedicados (los creados via
    // stream_for/launch(name)/capture(name,...)) terminen. NO toca
    // streams externos que el caller haya pasado a launch(name, stream)
    // manualmente.
    void synchronize_all();

    size_t count() const { return graphs_.size(); }
    size_t stream_count() const { return owned_streams_.size(); }

    MemoryPool& pool() { return pool_; }

private:
    MemoryPool& pool_;
    std::unordered_map<std::string, std::unique_ptr<Graph>> graphs_;
    std::unordered_map<std::string, std::unique_ptr<Stream>> owned_streams_;
};

} // namespace yadrakova::core
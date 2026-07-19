core/
├── memory.h/.cpp        → MemoryPool con free-list + guard de lifetime
│                           (shared_ptr<Impl> compartido, pool sobrevive
│                           aunque el objeto MemoryPool muera primero)
├── tensor.h/.cpp         → Tensor<T> templado, strides arbitrarios,
│                           views sin copia (transpose/permute/slice/reshape)
├── stream.h/.cpp         → wrapper de cudaStream_t
├── event.h/.cpp          → timing real de GPU (elapsed_ms, time_kernel_ms)
├── graph.h/.cpp          → captura/instancia/launch + guard anti-malloc-
│                           en-captura + update_from() para topologia estable
└── graph_manager.h/.cpp  → multiples grafos nombrados (listo para GAN
u orquestacion multi-red desde train/)

kernels/
├── registry.h/.cpp       → carga lazy de cubins via driver API, dispatch
│                           por (nombre, arch, dtype)
└── matmul.cu             → primer kernel real, validado correcto en
fp32 contra CPU (1.43e-06 de error)

scripts/
└── compile_kernel.py     → codegen automatico: detecta .cu nuevos,
compila multi-dtype x multi-arch, cachea
por hash, embebe cubin + auto-registro

CMakeLists.txt            → glob automatico de src/tests, target de
codegen como dependencia de la libreria
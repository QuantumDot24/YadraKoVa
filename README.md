![Diagrama de arquitectura](architecture.png)

# Cómo agregar una nueva arquitectura de GPU a YadraKoVa

Esta guía documenta el proceso completo para que YadraKoVa soporte una GPU con
una `compute capability` que todavía no conoce (ej. pasar de una RTX 3060
sm_86 a una RTX 5090 sm_120, o a un datacenter Blackwell sm_100).

## Antes de empezar: identifica el compute capability real

No asumas por el nombre comercial de la tarjeta. Verifica el compute
capability exacto en la [tabla oficial de NVIDIA](https://developer.nvidia.com/cuda-gpus).

Ejemplos de referencia (verificar siempre, esto puede cambiar con nuevos lanzamientos):
- RTX 3060 → sm_86 (Ampere)
- RTX 5090 → sm_120 (Blackwell consumer)
- B100 / B200 (datacenter) → sm_100 (Blackwell datacenter)

Blackwell consumer y Blackwell datacenter **no comparten compute capability**,
aunque compartan nombre de arquitectura. Trátalas como entradas separadas.

## Cuándo necesitas seguir esta guía completa vs. cuándo no

- **Comprar una segunda GPU con la misma arch que ya soportas** (ej. ya tienes
  una sm_86 y compras otra sm_86): no necesitas tocar nada. `Device` la
  detecta sola en runtime.
- **Comprar una GPU con compute capability nuevo**: sigue todos los pasos de
  abajo.

## Pasos

### 1. `include/kernels/registry.hpp` — agregar el valor al enum

```cpp
enum class Arch { SM_86, SM_100, SM_120 };  // agrega el nuevo valor
```

Este enum es el vocabulario compartido entre el codegen de build-time
(`compile_kernel.py`) y la selección de kernel en runtime (`KernelRegistry`,
`Device`). Vive aquí porque `KernelRegistry` es quien lo usa como parte de su
clave de búsqueda.

### 2. `src/core/device.cpp` — mapear compute capability → `Arch`

En `Device::map_arch`, agrega el caso correspondiente:

```cpp
kernels::Arch Device::map_arch(int major, int minor) {
    int cc = major * 10 + minor;
    switch (cc) {
        case 86:  return kernels::Arch::SM_86;
        case 100: return kernels::Arch::SM_100;
        case 120: return kernels::Arch::SM_120;  // <- nuevo
        default:
            throw std::runtime_error(/* ... */);
    }
}
```

`Device` detecta la GPU real en runtime vía `cudaGetDeviceProperties` y
traduce major/minor a este enum. Si te saltas este paso, el programa truena
al arrancar con un mensaje explicando exactamente qué falta (por diseño, no
falla en silencio).

### 3. `scripts/compile_kernel.py` — agregar a los diccionarios de mapeo

```python
ARCH_TO_SM = {"sm_86": "86", "sm_100": "100", "sm_120": "120"}
ARCH_TO_ENUM = {"sm_86": "Arch::SM_86", "sm_100": "Arch::SM_100", "sm_120": "Arch::SM_120"}
```

Estos diccionarios traducen el string del `config.yaml` (`sm_120`) al flag
real de `nvcc` (`-arch=sm_120`) y al literal C++ (`Arch::SM_120`) que se
escribe en los `.cpp` de registro autogenerados.

### 4. `config.yaml` — activar la arquitectura en el build

```yaml
cuda:
  architectures: [sm_86, sm_120]
  arch_flags:
    sm_86:
      extra_flags: []
    sm_120:
      extra_flags: []   # flags especificas de la arch, si aplica
```

Este es el único paso que "enciende" la compilación para la arch nueva. Los
pasos 1-3 son vocabulario que se toca una sola vez por arquitectura nunca
antes vista; este paso 4 es el que controla qué se compila en cada build.

### 5. Recompilar

```bash
python scripts/compile_kernel.py
```

Esto regenera automáticamente los `_embedded.cuh` y `_registration.cpp` de
**todos** los kernels existentes (`gelu`, `matmul_wmma`, etc.) para incluir
también los cubins de la arch nueva. No hay que tocar el código fuente `.cu`
de cada kernel individual para que compile en la arch nueva.

### 6. Nada más

`Device` detecta automáticamente en runtime qué GPU hay presente y pide a
`KernelRegistry` el cubin correspondiente. `Executor` no necesita ningún
cambio — nunca tuvo hardcodeada la arch, siempre le pregunta a `Device`.

## Nota sobre aprovechar features nuevas de la arquitectura

Compilar para una arch nueva (pasos 1-5) garantiza que el kernel **corre**
correctamente ahí — `nvcc` traduce el código CUDA existente al ISA de la
nueva arquitectura. Pero **no garantiza que estés aprovechando** hardware
nuevo exclusivo de esa arch (ej. tipos de dato fp4/fp6 en Blackwell
datacenter, cambios en cómo se invocan los fragments de Tensor Cores, nuevas
instrucciones WGMMA/TMA, etc.).

Aprovechar eso requiere escribir una variante del kernel específica para esa
arch (ej. `matmul_wmma_sm120.cu` con su propio path de código), lo cual es
trabajo de kernel-engineering, no de infraestructura. Cuando llegue ese
momento, es un proyecto aparte de "hacer que compile en la arch nueva".

## Checklist rápido

- [ ] Confirmar compute capability real (major.minor) de la GPU nueva
- [ ] `registry.hpp`: agregar valor al enum `Arch`
- [ ] `device.cpp`: agregar case en `Device::map_arch`
- [ ] `compile_kernel.py`: agregar entradas a `ARCH_TO_SM` y `ARCH_TO_ENUM`
- [ ] `config.yaml`: agregar el string de arch a `cuda.architectures`
- [ ] Correr `python scripts/compile_kernel.py`
- [ ] Correr los tests existentes en la GPU nueva para confirmar
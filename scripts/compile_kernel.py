#!/usr/bin/env python3
"""
YadraKoVa kernel compiler + embedder.

Escanea src/kernels/*.cu, compila cada uno a .cubin (SASS nativo, cero
overhead de carga -- sin JIT) para cada combinacion arch x dtype
definida en config.yaml, cachea por hash de contenido+flags+version de
nvcc+headers compartidos para evitar recompilar innecesariamente.

Politica de cache: SOLO se conserva la version mas reciente de cada
combinacion kernel+dtype+arch. Al detectar un cache miss (el kernel
cambio), se borra automaticamente cualquier .cubin viejo de esa misma
combinacion antes de compilar el nuevo -- pensado para iteracion
constante de tuning de performance sin acumular basura para siempre.

Genera DOS archivos por kernel, separando datos de efectos secundarios:
  - include/kernels/embedded/{kernel}_embedded.cuh
        Solo datos (arrays de bytes). Header puro, sin side effects.
  - generated/kernels/{kernel}_registration.cpp
        Solo el efecto secundario: registro estatico en KernelRegistry.
        Vive FUERA de include/, en el build dir, generado y descartable.
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import yaml  # pip install pyyaml

DTYPE_TO_CPP = {
    "bf16": "__nv_bfloat16",
    "fp32": "float",
    "fp16": "__half",
    "int8": "int8_t",
}

DTYPE_TO_ENUM = {
    "bf16": "core::DType::BF16",
    "fp32": "core::DType::FP32",
    "fp16": "core::DType::FP16",
    "int8": "core::DType::INT8",
}

ARCH_TO_SM = {"sm_86": "86", "sm_100": "100"}
ARCH_TO_ENUM = {"sm_86": "Arch::SM_86", "sm_100": "Arch::SM_100"}


def sha256_of(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


def nvcc_version() -> str:
    return subprocess.check_output(["nvcc", "--version"], text=True).strip()


def hash_shared_headers(shared_headers_dir: Path) -> str:
    """Hash combinado de TODOS los .cuh compartidos en include/kernels/
    (common.cuh, etc. -- no los generados en embedded/, esos son
    output, no input). Conservador a proposito: si CUALQUIER header
    compartido cambia, invalida el cache de TODOS los kernels. Mas
    seguro que un parser de #include que puede fallar silenciosamente."""
    h = hashlib.sha256()
    if not shared_headers_dir.exists():
        return h.hexdigest()
    for header in sorted(shared_headers_dir.glob("*.cuh")):
        h.update(header.read_bytes())
    return h.hexdigest()


def prune_old_versions(source: Path, dtype: str, arch: str,
                       cache_dir: Path, keep_path: Path):
    """Borra cualquier .cubin viejo de esta MISMA combinacion
    kernel+dtype+arch (distinto hash = version anterior del mismo
    archivo). Se llama SOLO en cache miss, justo antes de compilar
    la version nueva -- garantiza que .kernel_cache/ nunca acumula
    mas de una version por combinacion, sin importar cuantas veces
    itere el kernel durante tuning."""
    if not cache_dir.exists():
        return
    prefix = f"{source.stem}_{dtype}_{arch}_"
    for old_cubin in cache_dir.glob(f"{prefix}*.cubin"):
        if old_cubin != keep_path:
            print(f"  [cache prune] version vieja: {old_cubin.name}")
            old_cubin.unlink()


def prune_orphaned_kernels(cache_dir: Path, valid_kernel_names: set):
    """Borra .cubin de kernels cuyo .cu ya no existe (renombrado o
    eliminado). Se llama UNA vez al final, sobre todo el cache."""
    if not cache_dir.exists():
        return
    removed = 0
    for cubin in cache_dir.glob("*.cubin"):
        kernel_name = cubin.stem.split("_")[0]
        if kernel_name not in valid_kernel_names:
            cubin.unlink()
            removed += 1
    if removed:
        print(f"[cache] {removed} .cubin huerfanos eliminados (kernel ya no existe)")


def compile_one(source: Path, dtype: str, arch: str, cache_dir: Path,
                extra_flags: list, nvcc_ver: str, shared_headers_hash: str,
                include_dirs: list) -> Path:
    source_text = source.read_text(encoding="utf-8")
    key = sha256_of(source_text, dtype, arch, nvcc_ver,
                    " ".join(extra_flags), shared_headers_hash)
    cubin_path = cache_dir / f"{source.stem}_{dtype}_{arch}_{key}.cubin"

    if cubin_path.exists():
        print(f"  [cache hit]  {source.stem} dtype={dtype} arch={arch}")
        return cubin_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    prune_old_versions(source, dtype, arch, cache_dir, cubin_path)

    sm = ARCH_TO_SM[arch]
    cmd = ["nvcc", "--cubin", f"-arch=sm_{sm}", "-O3", "-std=c++17",
           f"-DKERNEL_DTYPE={DTYPE_TO_CPP[dtype]}"]
    for inc_dir in include_dirs:
        cmd += ["-I", str(inc_dir)]
    cmd += [*extra_flags, "-o", str(cubin_path), str(source)]

    print(f"  [nvcc]       {source.stem} dtype={dtype} arch={arch} (cache miss)")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR nvcc] {source.name} dtype={dtype} arch={arch}", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        raise

    return cubin_path


def embed_cubin_bytes(cubin_path: Path, var_name: str) -> str:
    data = cubin_path.read_bytes()
    lines = [f"inline constexpr unsigned char {var_name}[] = {{"]
    for i in range(0, len(data), 20):
        chunk = data[i:i + 20]
        lines.append("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append(f"inline constexpr size_t {var_name}_size = sizeof({var_name});")
    return "\n".join(lines)


def write_cuh(kernel_name: str, archs: list, dtypes: list,
              cubins: dict, cuh_path: Path):
    """Escribe el .cuh -- SOLO datos, cero efectos secundarios."""
    lines = [
        "// AUTOGENERADO por scripts/compile_kernel.py -- NO EDITAR A MANO.",
        f"// Fuente: src/kernels/{kernel_name}.cu",
        "// Solo datos (cubins embebidos). Sin efectos secundarios --",
        "// seguro de incluir desde cualquier translation unit.",
        "#pragma once",
        "#include <cstddef>",
        "",
        "namespace yadrakova::embedded {",
    ]
    for arch in archs:
        lines.append(f"namespace {arch} {{")
        for dtype in dtypes:
            var = f"{kernel_name}_{dtype}"
            lines.append(embed_cubin_bytes(cubins[(arch, dtype)], var))
        lines.append(f"}} // namespace {arch}")
    lines.append("} // namespace yadrakova::embedded")

    cuh_path.parent.mkdir(parents=True, exist_ok=True)
    cuh_path.write_text("\n".join(lines), encoding="utf-8")


def write_registration_cpp(kernel_name: str, archs: list, dtypes: list,
                           cuh_path: Path, registration_cpp_path: Path,
                           registry_header: Path):
    """Escribe el .cpp de registro -- SOLO el efecto secundario.
    Vive fuera de include/, en el build dir."""
    cuh_include = Path(os.path.relpath(cuh_path, registration_cpp_path.parent)).as_posix()
    registry_include = Path(os.path.relpath(registry_header, registration_cpp_path.parent)).as_posix()

    lines = [
        "// AUTOGENERADO por scripts/compile_kernel.py -- NO EDITAR A MANO.",
        "// Efecto secundario: registro estatico en KernelRegistry.",
        "// Vive fuera de include/ a proposito -- un .cpp con side",
        "// effects no pertenece al arbol de headers.",
        f'#include "{cuh_include}"',
        f'#include "{registry_include}"',
        "",
        f"namespace yadrakova::kernels::registration_{kernel_name} {{",
        f"struct {kernel_name.capitalize()}Registrar {{",
        f"    {kernel_name.capitalize()}Registrar() {{",
        "        auto& reg = KernelRegistry::instance();",
    ]
    for arch in archs:
        for dtype in dtypes:
            var = f"{kernel_name}_{dtype}"
            lines.append(
                f'        reg.register_kernel("{kernel_name}", {ARCH_TO_ENUM[arch]}, '
                f'{DTYPE_TO_ENUM[dtype]}, {{embedded::{arch}::{var}, embedded::{arch}::{var}_size}});'
            )
    lines += [
        "    }",
        "};",
        f"{kernel_name.capitalize()}Registrar instance;",
        "} // namespace",
    ]

    registration_cpp_path.parent.mkdir(parents=True, exist_ok=True)
    registration_cpp_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="Override del directorio de salida para los .cpp de "
                             "registro (usado por CMake para apuntar al build dir)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    archs = config["cuda"]["architectures"]
    dtypes = config["kernels"]["dtypes"]
    src_dir = Path(config["kernels"]["source_dir"])
    cuh_out_dir = Path(config["kernels"]["output_dir"])
    cpp_out_dir = Path(args.output_dir) if args.output_dir else Path("generated/kernels")
    registry_header = Path(config["kernels"].get("registry_header", "include/kernels/registry.h")).resolve()
    cache_dir = Path(config["cache"]["dir"])

    # Directorio de headers compartidos (common.cuh, etc.) -- mismo
    # nivel que registry.h, usado tanto para -I de nvcc como para
    # el hash de invalidacion.
    shared_headers_dir = Path("include/kernels")
    include_dirs = [Path("include")]

    try:
        nvcc_ver = nvcc_version()
    except FileNotFoundError:
        print("ERROR: nvcc no encontrado en PATH.", file=sys.stderr)
        sys.exit(1)

    sources = sorted(src_dir.glob("*.cu"))
    if not sources:
        print(f"No hay .cu en {src_dir}")
        return

    shared_headers_hash = hash_shared_headers(shared_headers_dir)

    for source in sources:
        kernel_name = source.stem
        print(f"[kernel] {kernel_name}")
        cubins = {}
        for arch in archs:
            extra_flags = config["cuda"]["arch_flags"].get(arch, {}).get("extra_flags", [])
            for dtype in dtypes:
                cubins[(arch, dtype)] = compile_one(
                    source, dtype, arch, cache_dir, extra_flags, nvcc_ver,
                    shared_headers_hash, include_dirs)

        cuh_path = cuh_out_dir / f"{kernel_name}_embedded.cuh"
        cpp_path = cpp_out_dir / f"{kernel_name}_registration.cpp"

        write_cuh(kernel_name, archs, dtypes, cubins, cuh_path)
        write_registration_cpp(kernel_name, archs, dtypes, cuh_path, cpp_path, registry_header)

        print(f"  -> {cuh_path}")
        print(f"  -> {cpp_path}")

    # Prune final: kernels renombrados/eliminados desde la ultima corrida.
    valid_names = {s.stem for s in sources}
    prune_orphaned_kernels(cache_dir, valid_names)


if __name__ == "__main__":
    main()
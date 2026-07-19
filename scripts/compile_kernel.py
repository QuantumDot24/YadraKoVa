#!/usr/bin/env python3
"""
YadraKoVa kernel compiler + embedder.

Escanea src/kernels/*.cu, compila cada uno a .cubin (SASS nativo, cero
overhead de carga -- sin JIT) para cada combinacion arch x dtype
definida en config.yaml, cachea por hash de contenido+flags+version de
nvcc para evitar recompilar, y genera un .cuh embebido por kernel en
include/kernels/embedded/, incluyendo el codigo de auto-registro en
el KernelRegistry.
"""
import argparse
import hashlib
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

ARCH_TO_SM = {"sm_86": "86", "sm_100": "100"}


def sha256_of(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


def nvcc_version() -> str:
    return subprocess.check_output(["nvcc", "--version"], text=True).strip()


def compile_one(source: Path, dtype: str, arch: str, cache_dir: Path,
                extra_flags: list, nvcc_ver: str) -> Path:
    """Compila a .cubin usando cache por hash. Si el hash ya existe en
    cache_dir, NO se invoca nvcc -- se reusa el .cubin de disco."""
    source_text = source.read_text(encoding="utf-8")
    key = sha256_of(source_text, dtype, arch, nvcc_ver, " ".join(extra_flags))
    cubin_path = cache_dir / f"{source.stem}_{dtype}_{arch}_{key}.cubin"

    if cubin_path.exists():
        print(f"  [cache hit]  {source.stem} dtype={dtype} arch={arch}")
        return cubin_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    sm = ARCH_TO_SM[arch]
    cmd = [
        "nvcc", "--cubin", f"-arch=sm_{sm}", "-O3",
        f"-DKERNEL_DTYPE={DTYPE_TO_CPP[dtype]}",
        *extra_flags,
        "-o", str(cubin_path), str(source),
    ]
    print(f"  [nvcc]       {source.stem} dtype={dtype} arch={arch} (cache miss)")
    subprocess.run(cmd, check=True)
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


def generate_header(kernel_name: str, archs: list, dtypes: list,
                    cubins: dict, output_path: Path):
    dtype_enum = {"bf16": "DType::BF16", "fp32": "DType::FP32",
                  "fp16": "DType::FP16", "int8": "DType::INT8"}
    arch_enum = {"sm_86": "Arch::SM_86", "sm_100": "Arch::SM_100"}

    lines = [
        "// AUTOGENERADO por scripts/compile_kernel.py -- NO EDITAR A MANO.",
        f"// Fuente: src/kernels/{kernel_name}.cu",
        "#pragma once",
        "#include <cstddef>",
        '#include "kernels/registry.h"',
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

    # Auto-registro: al incluir este header en cualquier .cpp, un
    # objeto estatico se construye antes de main() y registra los
    # 4 dtypes x N archs en el KernelRegistry global. Cero pasos
    # manuales -- igual que "registrar shaders" en tu sistema Vulkan.
    lines.append("")
    lines.append(f"namespace yadrakova::kernels::registration_{kernel_name} {{")
    lines.append(f"struct {kernel_name.capitalize()}Registrar {{")
    lines.append(f"    {kernel_name.capitalize()}Registrar() {{")
    lines.append("        auto& reg = KernelRegistry::instance();")
    for arch in archs:
        for dtype in dtypes:
            var = f"{kernel_name}_{dtype}"
            lines.append(
                f'        reg.register_kernel("{kernel_name}", {arch_enum[arch]}, '
                f'{dtype_enum[dtype]}, {{embedded::{arch}::{var}, embedded::{arch}::{var}_size}});'
            )
    lines.append("    }")
    lines.append("};")
    lines.append(f"inline {kernel_name.capitalize()}Registrar instance;")
    lines.append("} // namespace")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    archs = config["cuda"]["architectures"]
    dtypes = config["kernels"]["dtypes"]
    src_dir = Path(config["kernels"]["source_dir"])
    out_dir = Path(config["kernels"]["output_dir"])
    cache_dir = Path(config["cache"]["dir"])

    try:
        nvcc_ver = nvcc_version()
    except FileNotFoundError:
        print("ERROR: nvcc no encontrado en PATH.", file=sys.stderr)
        sys.exit(1)

    sources = sorted(src_dir.glob("*.cu"))
    if not sources:
        print(f"No hay .cu en {src_dir}")
        return

    for source in sources:
        kernel_name = source.stem
        print(f"[kernel] {kernel_name}")
        cubins = {}
        for arch in archs:
            extra_flags = config["cuda"]["arch_flags"].get(arch, {}).get("extra_flags", [])
            for dtype in dtypes:
                cubins[(arch, dtype)] = compile_one(
                    source, dtype, arch, cache_dir, extra_flags, nvcc_ver)
        out_header = out_dir / f"{kernel_name}_embedded.cuh"
        generate_header(kernel_name, archs, dtypes, cubins, out_header)
        print(f"  -> {out_header}")


if __name__ == "__main__":
    main()
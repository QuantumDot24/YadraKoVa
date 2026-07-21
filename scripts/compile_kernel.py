#!/usr/bin/env python3
"""
YadraKoVa kernel compiler + embedder + dispatch codegen.

Escanea src/kernels/*.cu, compila cada uno a .cubin (SASS nativo, cero
overhead de carga -- sin JIT) para cada combinacion arch x dtype
definida en config.yaml, cachea por hash de contenido+flags+version de
nvcc+headers compartidos para evitar recompilar innecesariamente.

Politica de cache: SOLO se conserva la version mas reciente de cada
combinacion kernel+dtype+arch. Al detectar un cache miss (el kernel
cambio), se borra automaticamente cualquier .cubin viejo de esa misma
combinacion antes de compilar el nuevo -- pensado para iteracion
constante de tuning de performance sin acumular basura para siempre.

Genera TRES archivos por kernel, separando datos de efectos secundarios:
  - include/kernels/embedded/{kernel}_embedded.cuh
        Solo datos (arrays de bytes). Header puro, sin side effects.
  - generated/kernels/{kernel}_registration.cpp
        Efecto secundario: registro estatico en KernelRegistry
        (nombre -> CUfunction). Vive FUERA de include/, en el build
        dir, generado y descartable.
  - generated/kernels/{kernel}_dispatch.generated.cpp   [NUEVO]
        Efecto secundario: registro estatico en DispatchRegistry
        (nombre -> regla de calculo de grid/block), derivado de
        src/kernels/{kernel}.yaml. Si el .yaml no existe, se omite
        con un aviso -- permite migrar kernels uno por uno.

El .yaml de dispatch vive junto al .cu (mismo stem, misma carpeta) y
NO participa del hash de cache de compilacion de cubins: cambiar solo
la regla de dispatch no deberia forzar una recompilacion de CUDA.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import yaml  # pip install pyyaml
from pathlib import Path

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


# --------------------------------------------------------------------------
# NUEVO: codegen de dispatch (grid/block) a partir de src/kernels/{k}.yaml
# --------------------------------------------------------------------------

def _ceildiv_expr(numer: str, denom: int) -> str:
    """ceil(numer/denom) en aritmetica entera, sin floats. numer es un
    nombre de variable (M, N o K) que existe como parametro int64_t de
    la funcion de dispatch generada; denom es un literal (block size),
    conocido en tiempo de generacion."""
    return f"(({numer} + {denom} - 1) / {denom})"


# Cada helper declara a que dimensiones de grid (x/y/z) mapea cada uno
# de sus argumentos posicionales. tile2d(filas, columnas) es el caso
# comun de "un thread por elemento de una matriz 2D de salida".
GRID_HELPERS = {
    "tile2d": ("y", "x"),
    "rows": ("y",),
    "cols": ("x",),
    "elementwise": ("x",),
}


def parse_grid_spec(grid_field, block: list) -> dict:
    """Devuelve {'x': cpp_expr, 'y': cpp_expr, 'z': cpp_expr}, expresado
    en terminos de M/N/K (los parametros de la funcion de dispatch) con
    el block size ya resuelto como literal entero."""
    dims = {"x": "1", "y": "1", "z": "1"}
    block_by_dim = {"x": block[0], "y": block[1], "z": block[2] if len(block) > 2 else 1}

    if isinstance(grid_field, str):
        grid_field = grid_field.strip()
        name, _, inner = grid_field.partition("(")
        name = name.strip()
        if name not in GRID_HELPERS or not inner.endswith(")"):
            raise ValueError(
                f"grid: '{grid_field}' no reconocido. Usa uno de "
                f"{list(GRID_HELPERS)}(...) o la forma explicita {{x: .., y: ..}}"
            )
        arg_names = [a.strip() for a in inner[:-1].split(",") if a.strip()]
        target_dims = GRID_HELPERS[name]
        if len(arg_names) != len(target_dims):
            raise ValueError(
                f"grid: '{grid_field}' espera {len(target_dims)} argumento(s), "
                f"recibio {len(arg_names)}"
            )
        for var_name, dim in zip(arg_names, target_dims):
            if var_name not in ("M", "N", "K"):
                raise ValueError(
                    f"grid: '{var_name}' no es M, N ni K -- revisa {grid_field}"
                )
            dims[dim] = _ceildiv_expr(var_name, block_by_dim[dim])
        return dims

    if isinstance(grid_field, dict):
        # Forma explicita, fallback para casos que no encajan en un
        # helper: cada componente es una expresion C++ escrita a mano,
        # libre de usar M, N, K.
        for dim in ("x", "y", "z"):
            if dim in grid_field:
                dims[dim] = str(grid_field[dim])
        return dims

    raise ValueError(f"grid: tipo no soportado ({type(grid_field).__name__}); usa string o dict")


def find_dispatch_yaml(source: Path) -> Path | None:
    yaml_path = source.with_suffix(".yaml")
    return yaml_path if yaml_path.exists() else None


import re

SIG_RE = re.compile(
    r'__global__\s+void\s+\w+_kernel\s*\((.*?)\)\s*\{',
    re.DOTALL
)


def parse_kernel_signature(source_text: str) -> list[tuple[str, str]]:
    """Devuelve [(tipo_cpp, nombre), ...] en orden, desde la firma real."""
    m = SIG_RE.search(source_text)
    if not m:
        raise ValueError("No se encontro __global__ ...(...) { en el kernel")
    params = [p.strip() for p in m.group(1).split(",") if p.strip()]
    result = []
    for p in params:
        parts = p.replace("*", " * ").split()
        name = parts[-1]
        type_str = " ".join(parts[:-1]).replace(" * ", "*").strip()
        result.append((type_str, name))
    return result


def validate_args_match(kernel_name: str, source_text: str, yaml_args: list):
    real_sig = parse_kernel_signature(source_text)
    if len(real_sig) != len(yaml_args):
        raise ValueError(
            f"'{kernel_name}': yaml declara {len(yaml_args)} args, "
            f"la firma real tiene {len(real_sig)}"
        )
    for (real_type, real_name), decl in zip(real_sig, yaml_args):
        if real_name != decl["name"]:
            raise ValueError(
                f"'{kernel_name}': orden/nombre no coincide -- yaml dice "
                f"'{decl['name']}', firma real dice '{real_name}'"
            )
        if decl["kind"] in ("in", "out") and "*" not in real_type:
            raise ValueError(
                f"'{kernel_name}.{real_name}': yaml dice kind={decl['kind']} "
                f"(deberia ser puntero) pero la firma real es '{real_type}'"
            )
        expected_scalar = {"int64": "int64_t", "int32": "int"}.get(decl.get("dtype", ""))
        if decl["kind"] == "scalar" and expected_scalar and expected_scalar not in real_type:
            raise ValueError(
                f"'{kernel_name}.{real_name}': yaml declara dtype={decl['dtype']} "
                f"pero la firma real es '{real_type}'"
            )


def parse_grid_spec(grid_field, block: list, dim_names: list) -> dict:
    """Igual que antes, pero valida contra `dims:` del yaml en vez de
    contra M/N/K fijos -- cualquier kernel puede nombrar sus propias
    dimensiones."""
    dims = {"x": "1", "y": "1", "z": "1"}
    block_by_dim = {"x": block[0], "y": block[1], "z": block[2] if len(block) > 2 else 1}

    if isinstance(grid_field, str):
        grid_field = grid_field.strip()
        name, _, inner = grid_field.partition("(")
        name = name.strip()
        if name not in GRID_HELPERS or not inner.endswith(")"):
            raise ValueError(
                f"grid: '{grid_field}' no reconocido. Usa uno de "
                f"{list(GRID_HELPERS)}(...) o la forma explicita {{x: .., y: ..}}"
            )
        arg_names = [a.strip() for a in inner[:-1].split(",") if a.strip()]
        target_dims = GRID_HELPERS[name]
        if len(arg_names) != len(target_dims):
            raise ValueError(
                f"grid: '{grid_field}' espera {len(target_dims)} argumento(s), "
                f"recibio {len(arg_names)}"
            )
        for var_name, dim in zip(arg_names, target_dims):
            if var_name not in dim_names:
                raise ValueError(
                    f"grid: '{var_name}' no esta declarado en 'dims: {dim_names}' -- revisa {grid_field}"
                )
            dims[dim] = _ceildiv_expr(var_name, block_by_dim[dim])
        return dims

    if isinstance(grid_field, dict):
        for dim in ("x", "y", "z"):
            if dim in grid_field:
                dims[dim] = str(grid_field[dim])
        return dims

    raise ValueError(f"grid: tipo no soportado ({type(grid_field).__name__}); usa string o dict")


def write_dispatch_cpp(kernel_name: str, kernel_yaml: dict, source_text: str,
                       dispatch_cpp_path: Path, dispatch_registry_header: Path):
    """Escribe el .cpp de registro de dispatch. Ahora valida yaml vs
    firma real ANTES de generar, y arma la lambda sobre DimMap en vez
    de M,N,K posicionales fijos."""
    if kernel_yaml.get("kernel", kernel_name) != kernel_name:
        print(f"  [WARN] 'kernel:' en el yaml no coincide con el nombre de archivo "
              f"({kernel_yaml.get('kernel')!r} vs {kernel_name!r})", file=sys.stderr)

    dim_names = kernel_yaml.get("dims")
    if not dim_names:
        raise ValueError(f"'{kernel_name}.yaml' no define 'dims:' (ej. [M, N, K] o [n])")

    yaml_args = kernel_yaml.get("args")
    if not yaml_args:
        raise ValueError(f"'{kernel_name}.yaml' no define 'args:'")

    validate_args_match(kernel_name, source_text, yaml_args)

    block = list(kernel_yaml.get("block", [1, 1, 1]))
    while len(block) < 3:
        block.append(1)
    total_threads = block[0] * block[1] * block[2]
    if total_threads > 1024:
        raise ValueError(
            f"'{kernel_name}.yaml': block {block} = {total_threads} threads/bloque, "
            f"excede el maximo de 1024 de CUDA"
        )

    grid_field = kernel_yaml.get("grid")
    if grid_field is None:
        raise ValueError(f"'{kernel_name}.yaml' no define 'grid'")
    grid = parse_grid_spec(grid_field, block, dim_names)

    shared_mem = int(kernel_yaml.get("shared_mem_bytes", 0))

    registry_include = Path(os.path.relpath(dispatch_registry_header, dispatch_cpp_path.parent)).as_posix()
    class_name = f"{kernel_name.capitalize()}DispatchRegistrar"
    fn_name = f"compute_{kernel_name}_dispatch"

    dim_extract_lines = [f'    int64_t {name} = dims.at("{name}");' for name in dim_names]

    lines = [
        "// AUTOGENERADO por scripts/compile_kernel.py -- NO EDITAR A MANO.",
        f"// Regla de dispatch para '{kernel_name}', derivada de src/kernels/{kernel_name}.yaml",
        f'#include "{registry_include}"',
        "",
        f"namespace yadrakova::kernels::dispatch_{kernel_name} {{",
        "",
        f"core::DispatchDims {fn_name}(const core::DimMap& dims) {{",
        "    core::DispatchDims out;",
        *dim_extract_lines,
        f"    out.block = core::Dim3{{{block[0]}u, {block[1]}u, {block[2]}u}};",
        "    out.grid = core::Dim3{",
        f"        static_cast<unsigned int>({grid['x']}),",
        f"        static_cast<unsigned int>({grid['y']}),",
        f"        static_cast<unsigned int>({grid['z']})",
        "    };",
        f"    out.shared_mem_bytes = {shared_mem}u;",
        "    return out;",
        "}",
        "",
        f"struct {class_name} {{",
        f"    {class_name}() {{",
        f'        core::DispatchRegistry::instance().register_dispatch("{kernel_name}", &{fn_name});',
        "    }",
        "};",
        f"{class_name} instance;",
        "",
        "} // namespace",
    ]

    dispatch_cpp_path.parent.mkdir(parents=True, exist_ok=True)
    dispatch_cpp_path.write_text("\n".join(lines), encoding="utf-8")

import re

def extract_kernel_name(cubin_stem: str, known_dtypes: set) -> str:
    # Buscar el patrón "_{dtype}_" y tomar todo lo anterior como kernel_name
    for dtype in known_dtypes:
        marker = f"_{dtype}_"
        if marker in cubin_stem:
            return cubin_stem[:cubin_stem.index(marker)]
    # fallback: split por "_" y confiar en que no haya dtype en el nombre (poco fiable)
    return cubin_stem.split("_")[0]

def prune_orphaned_kernels(cache_dir: Path, valid_kernel_names: set, known_dtypes: set):
    if not cache_dir.exists():
        return
    removed = 0
    for cubin in cache_dir.glob("*.cubin"):
        kernel_name = extract_kernel_name(cubin.stem, known_dtypes)
        if kernel_name not in valid_kernel_names:
            cubin.unlink()
            removed += 1
    if removed:
        print(f"[cache] {removed} .cubin huerfanos eliminados (kernel ya no existe)")

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
    registry_header = Path(config["kernels"].get("registry_header", "include/kernels/registry.hpp")).resolve()
    dispatch_registry_header = Path(
        config["kernels"].get("dispatch_registry_header", "include/core/dispatch_registry.hpp")
    ).resolve()
    cache_dir = Path(config["cache"]["dir"])

    # Directorio de headers compartidos (common.cuh, etc.) -- mismo
    # nivel que registry.hpp, usado tanto para -I de nvcc como para
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

        yaml_path = find_dispatch_yaml(source)
        kernel_yaml = None
        if yaml_path is not None:
            with open(yaml_path, encoding="utf-8") as f:
                kernel_yaml = yaml.safe_load(f)

        kernel_dtypes = kernel_yaml.get("dtypes", dtypes) if kernel_yaml else dtypes
        if kernel_dtypes != dtypes:
            print(f"  [dtypes override] {kernel_name}: {kernel_dtypes} ...")

        cubins = {}
        for arch in archs:
            extra_flags = config["cuda"]["arch_flags"].get(arch, {}).get("extra_flags", [])
            for dtype in kernel_dtypes:
                cubins[(arch, dtype)] = compile_one(
                    source, dtype, arch, cache_dir, extra_flags, nvcc_ver,
                    shared_headers_hash, include_dirs)

        cuh_path = cuh_out_dir / f"{kernel_name}_embedded.cuh"
        cpp_path = cpp_out_dir / f"{kernel_name}_registration.cpp"

        write_cuh(kernel_name, archs, kernel_dtypes, cubins, cuh_path)
        write_registration_cpp(kernel_name, archs, kernel_dtypes, cuh_path, cpp_path, registry_header)

        print(f"  -> {cuh_path}")
        print(f"  -> {cpp_path}")

        if kernel_yaml is None:
            print(f"  [dispatch] sin {kernel_name}.yaml -- se omite ...")
        else:
            source_text = source.read_text(encoding="utf-8")
            dispatch_cpp_path = cpp_out_dir / f"{kernel_name}_dispatch.generated.cpp"
            write_dispatch_cpp(kernel_name, kernel_yaml, source_text, dispatch_cpp_path, dispatch_registry_header)
            print(f"  -> {dispatch_cpp_path}")

    # Prune final: kernels renombrados/eliminados desde la ultima corrida.
    valid_names = {s.stem for s in sources}
    prune_orphaned_kernels(cache_dir, valid_names, set(dtypes))  # dtypes es la lista de config


if __name__ == "__main__":
    main()

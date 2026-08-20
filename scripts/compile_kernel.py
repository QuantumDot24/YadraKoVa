#!/usr/bin/env python3
"""
YadraKoVa kernel compiler + embedder + dispatch codegen + ops codegen.

Genera automáticamente:
  - include/kernels/embedded/{kernel}_embedded.cuh  (datos: cubins embebidos)
  - generated/kernels/{kernel}_registration.cpp     (registro en KernelRegistry)
  - generated/kernels/{kernel}_dispatch.generated.cpp (registro en DispatchRegistry)
  - generated/ops/{kernel}_ops.hpp                  (función libre + wrapper Tensor)
  - generated/ops/ops_metadata.hpp                  (enum Op + op_name + kNumOps)
  - generated/ops/ops_caps.hpp                      (BackendCaps: preference_order + supported_dtypes)
  - generated/ops/all_ops.hpp                       (include maestro de todos los *_ops.hpp)
  - generated/ops/tensor_declarations.hpp           (declaraciones para incluir DENTRO de Tensor<T>)
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import yaml
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


# =========================================================================
# Utilidades de cache y compilación
# =========================================================================
def sha256_of(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


def nvcc_version() -> str:
    return subprocess.check_output(["nvcc", "--version"], text=True).strip()


def hash_shared_headers(shared_headers_dir: Path) -> str:
    h = hashlib.sha256()
    if not shared_headers_dir.exists():
        return h.hexdigest()
    for header in sorted(shared_headers_dir.glob("*.cuh")):
        h.update(header.read_bytes())
    return h.hexdigest()


def prune_old_versions(source: Path, dtype: str, arch: str,
                       cache_dir: Path, keep_path: Path):
    if not cache_dir.exists():
        return
    prefix = f"{source.stem}_{dtype}_{arch}_"
    for old_cubin in cache_dir.glob(f"{prefix}*.cubin"):
        if old_cubin != keep_path:
            print(f"  [cache prune] old version: {old_cubin.name}")
            old_cubin.unlink()


def extract_kernel_name(cubin_stem: str, known_dtypes: set) -> str:
    for dtype in known_dtypes:
        marker = f"_{dtype}_"
        if marker in cubin_stem:
            return cubin_stem[:cubin_stem.index(marker)]
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
        print(f"[cache] {removed} orphaned .cubin removed (kernel no longer exists)")


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
        if e.stdout: print(e.stdout, file=sys.stderr)
        if e.stderr: print(e.stderr, file=sys.stderr)
        raise

    return cubin_path


# =========================================================================
# Generación de .cuh (datos) y .cpp de registro
# =========================================================================
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
    lines = [
        "// AUTO-GENERATED by scripts/compile_kernel.py -- DO NOT EDIT BY HAND.",
        f"// Source: src/kernels/{kernel_name}.cu",
        "// Data only (embedded cubins). No side effects.",
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
    cuh_include = Path(os.path.relpath(cuh_path, registration_cpp_path.parent)).as_posix()
    registry_include = Path(os.path.relpath(registry_header, registration_cpp_path.parent)).as_posix()

    lines = [
        "// AUTO-GENERATED by scripts/compile_kernel.py -- DO NOT EDIT BY HAND.",
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


# =========================================================================
# Parseo de firma del kernel y YAML
# =========================================================================
SIG_RE = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{',
    re.DOTALL
)

# Fallback si no tiene extern "C"
SIG_RE_FALLBACK = re.compile(
    r'__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{',
    re.DOTALL
)


def parse_kernel_signature(source_text: str) -> tuple[str, list[tuple[str, str]]]:
    """Returns (kernel_function_name, [(cpp_type, name), ...])."""
    m = SIG_RE.search(source_text) or SIG_RE_FALLBACK.search(source_text)
    if not m:
        raise ValueError("Could not find __global__ void ...(...) { in the kernel")
    fn_name = m.group(1)
    params = [p.strip() for p in m.group(2).split(",") if p.strip()]
    result = []
    for p in params:
        parts = p.replace("*", " * ").split()
        name = parts[-1]
        type_str = " ".join(parts[:-1]).replace(" * ", "*").strip()
        result.append((type_str, name))
    return fn_name, result


SCALAR_DIM_TYPES = {"int64_t", "int32_t", "int", "unsigned", "unsigned int", "size_t", "uint32_t", "uint64_t"}

def infer_dim_names(sig: list[tuple[str, str]]) -> list[str]:
    """Un escalar-dimensión es un entero pasado por valor (no struct, no puntero)."""
    return [name for typ, name in sig if "*" not in typ and typ.strip() in SCALAR_DIM_TYPES]


def find_dispatch_yaml(source: Path) -> Path | None:
    yaml_path = source.with_suffix(".yaml")
    return yaml_path if yaml_path.exists() else None


# =========================================================================
# Grid helpers
# =========================================================================
def _ceildiv_expr(numer: str, denom: int) -> str:
    return f"(({numer} + {denom} - 1) / {denom})"


GRID_HELPERS = {
    "tile2d": ("y", "x"),
    "rows": ("y",),
    "cols": ("x",),
    "elementwise": ("x",),
}


def parse_grid_spec(grid_field, block: list, dim_names: list) -> dict:
    dims = {"x": "1", "y": "1", "z": "1"}
    block_by_dim = {"x": block[0], "y": block[1], "z": block[2] if len(block) > 2 else 1}

    if isinstance(grid_field, str):
        grid_field = grid_field.strip()
        name, _, inner = grid_field.partition("(")
        name = name.strip()
        if name not in GRID_HELPERS or not inner.endswith(")"):
            raise ValueError(
                f"grid: '{grid_field}' not recognized. Use one of "
                f"{list(GRID_HELPERS)}(...) or the explicit form {{x: .., y: ..}}"
            )
        arg_names = [a.strip() for a in inner[:-1].split(",") if a.strip()]
        target_dims = GRID_HELPERS[name]
        if len(arg_names) != len(target_dims):
            raise ValueError(
                f"grid: '{grid_field}' expects {len(target_dims)} argument(s), "
                f"got {len(arg_names)}"
            )
        for var_name, dim in zip(arg_names, target_dims):
            if var_name not in dim_names:
                raise ValueError(
                    f"grid: '{var_name}' is not a scalar in the kernel signature."
                )
            dims[dim] = _ceildiv_expr(var_name, block_by_dim[dim])
        return dims

    if isinstance(grid_field, dict):
        for dim in ("x", "y", "z"):
            if dim in grid_field:
                dims[dim] = str(grid_field[dim])
        return dims

    raise ValueError(f"grid: unsupported type ({type(grid_field).__name__})")


# =========================================================================
# Generación de dispatch.generated.cpp
# =========================================================================
def write_dispatch_cpp(kernel_name: str, kernel_yaml: dict, source_text: str,
                       dispatch_cpp_path: Path, dispatch_registry_header: Path):
    # Inferir dimensiones del signature del kernel (YAML minimalista)
    _, sig = parse_kernel_signature(source_text)
    dim_names = infer_dim_names(sig)

    block = list(kernel_yaml.get("block", [1, 1, 1]))
    while len(block) < 3:
        block.append(1)
    total_threads = block[0] * block[1] * block[2]
    if total_threads > 1024:
        raise ValueError(
            f"'{kernel_name}.yaml': block {block} = {total_threads} threads/block, "
            f"exceeds CUDA maximum of 1024"
        )

    grid_field = kernel_yaml.get("grid")
    if grid_field is None:
        raise ValueError(f"'{kernel_name}.yaml' does not define 'grid'")
    grid = parse_grid_spec(grid_field, block, dim_names)

    shared_mem = int(kernel_yaml.get("shared_mem_bytes", 0))

    registry_include = Path(os.path.relpath(dispatch_registry_header, dispatch_cpp_path.parent)).as_posix()
    class_name = f"{kernel_name.capitalize()}DispatchRegistrar"
    fn_name = f"compute_{kernel_name}_dispatch"

    dim_extract_lines = [f'    int64_t {name} = dims.at("{name}");' for name in dim_names]

    lines = [
        "// AUTO-GENERATED by scripts/compile_kernel.py -- DO NOT EDIT BY HAND.",
        f"// Dispatch rule for '{kernel_name}', inferred from kernel signature.",
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


# =========================================================================
# Generación de ops de alto nivel (API de Tensor)
# =========================================================================
def generate_op_hpp(kernel_name: str, op_meta: dict, sig: list, output_dir: Path):
    """Genera la función libre y el wrapper de Tensor::method para una op."""
    op_name = op_meta.get("name", kernel_name.capitalize())
    inputs = op_meta.get("inputs", [])
    output = op_meta.get("output", "out")
    checks = op_meta.get("checks", [])
    scalar_inits = op_meta.get("scalar_init", {})

    # Identificar escalares (no punteros) del signature
    scalar_args = [name for typ, name in sig if "*" not in typ]

    # Construir checks
    check_code = "\n".join([
        f'    if (!({check})) throw std::runtime_error("{op_name.lower()}: validation failed ({check})");'
        for check in checks
    ])

    # DimMap entries
    dim_map_entries = [f'{{"{name}", {name}}}' for name in scalar_args]

    # Args para el kernel (punteros y escalares)
    args_entries = []
    for typ, name in sig:
        if name in inputs or name == output:
            args_entries.append(f'&{name.lower()}_ptr')
        else:
            args_entries.append(f'&{name}')

    # Punteros de entrada/salida
    input_ptrs = [f'    const T* {name.lower()}_ptr = {name}.data();' for name in inputs]
    output_ptr = f'    T* {output.lower()}_ptr = {output.lower()}.data();'
    out_shape = f"{inputs[0]}.shape()" if inputs else "in.shape()"

    # Parámetros de la función libre
    input_params = ", ".join([f"const Tensor<T>& {inp}" for inp in inputs])
    func_params = f"{input_params}, Backend backend, Stream& stream" if input_params else "Backend backend, Stream& stream"

    lines = [
        "// AUTO-GENERATED by scripts/compile_kernel.py -- DO NOT EDIT BY HAND.",
        "#pragma once",
        '#include "core/tensor.hpp"',
        '#include "core/backend_dispatch.hpp"',
        "",
        "namespace yadrakova::core {",
        "",
        f"// Función libre para {op_name}",
        f"template <typename T>",
        f"Tensor<T> {op_name.lower()}({func_params}) {{",
    ]

    if check_code:
        lines.append(check_code)

    # Declaraciones de escalares con inicialización
    scalar_decls = []
    for typ, name in sig:
        if name in scalar_args:
            if name in scalar_inits:
                scalar_decls.append(f"    {typ} {name} = {scalar_inits[name]};")
            else:
                scalar_decls.append(f"    {typ} {name} = 0; // WARNING: Add '{name}' to scalar_init in YAML")

    lines.extend([
                     "",
                     f"    Tensor<T> {output.lower()}({out_shape});",
                 ] + input_ptrs + [
                     f"    {output_ptr}",
                 ] + scalar_decls + [
                     "",
                     f"    std::vector<void*> args = {{ {', '.join(args_entries)} }};",
                     f"    dispatch_op<T>(Op::{op_name}, \"{kernel_name}\", DimMap{{{', '.join(dim_map_entries)}}}, args, backend, stream);",
                     f"    return {output.lower()};",
                     "}",
                     "",
                     f"// Wrapper del método de Tensor",
                     f"template <typename T>",
                 ])

    if len(inputs) > 1:
        method_params = ", ".join([f"const Tensor<T>& {inp}" for inp in inputs[1:]] + ["Backend backend", "Stream& stream"])
        method_args = ", ".join([inp for inp in inputs[1:]] + ["backend", "stream"])
        lines.append(f"Tensor<T> Tensor<T>::{op_name.lower()}({method_params}) const {{")
        lines.append(f"    return yadrakova::core::{op_name.lower()}(*this, {method_args});")
    else:
        lines.append(f"Tensor<T> Tensor<T>::{op_name.lower()}(Backend backend, Stream& stream) const {{")
        lines.append(f"    return yadrakova::core::{op_name.lower()}(*this, backend, stream);")

    lines.extend(["}", "", "} // namespace yadrakova::core"])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{kernel_name}_ops.hpp").write_text("\n".join(lines), encoding="utf-8")


def generate_global_op_files(all_ops_meta: list, output_dir: Path):
    """Genera los archivos globales: enum Op, capacidades, includes maestros."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ops que hacen dispatch manual (escritas a mano en tensor.hpp, no siguen
    # el patrón genérico de generate_op_hpp) pero igual necesitan un valor
    # en el enum Op porque se llaman via dispatch_op<T>(Op::X, ...).
    # Estas NO generan wrapper de alto nivel ni entran en tensor_declarations.hpp.
    MANUAL_OPS = [{"name": "Contiguous"}]

    # all_ops_meta = ops autogeneradas (con sección 'op:' en su .yaml, ej.
    # Add/Mul/MatMul/Softmax). enum_ops = unión, solo para el enum y op_name.
    enum_ops = MANUAL_OPS + all_ops_meta

    # 1. ops_metadata.hpp (enum Op + kNumOps + op_name)
    meta_lines = [
        "// AUTO-GENERATED by scripts/compile_kernel.py -- DO NOT EDIT BY HAND.",
        "#pragma once",
        "#include <cstdint>",
        "#include <string>",
        "#include <stdexcept>",
        "",
        "namespace yadrakova::core {",
        "",
        "enum class Op : uint8_t {"
    ]
    enum_entries = [f"    {op['name']} = {i}" for i, op in enumerate(enum_ops)]
    meta_lines.extend([",\n".join(enum_entries), "};", ""])
    meta_lines.append(f"inline constexpr size_t kNumOps = {len(enum_ops)};")
    meta_lines.append("")
    meta_lines.append("inline const char* op_name(Op op) {")
    meta_lines.append("    switch (op) {")
    for op in enum_ops:
        meta_lines.append(f"        case Op::{op['name']}: return \"{op['name'].lower()}\";")
    meta_lines.extend(["    }", '    throw std::runtime_error("op_name: Op desconocido");', "}", "", "} // namespace yadrakova::core"])
    (output_dir / "ops_metadata.hpp").write_text("\n".join(meta_lines), encoding="utf-8")

    # 2. ops_caps.hpp (preference_order + supported_dtypes)
    caps_lines = [
        "// AUTO-GENERATED by scripts/compile_kernel.py -- DO NOT EDIT BY HAND.",
        "#pragma once",
        '#include "core/backend.hpp"',   # trae Op, Backend
        '#include "dtype_utils.hpp"',    # trae DType, DTypeMask, dtype_bit, kAllDTypes
        "#include <vector>",
        "",
        "namespace yadrakova::core {",
        "",
        "inline const std::vector<Backend>& get_preference_order(Op op) {"
    ]
    # Ops manuales (ej. Contiguous) siempre son Custom-only: no tienen
    # sección 'op:' ni lista de 'backends' en su yaml, así que no entran
    # al loop de abajo (que usa all_ops_meta) -- caen directo al default/fallback.
    for op in all_ops_meta:
        backends = op.get("backends", ["Custom"])
        backend_list = ", ".join([f"Backend::{b}" for b in backends])
        caps_lines.append(f"    static const std::vector<Backend> {op['name'].lower()}_order = {{{backend_list}}};")
    caps_lines.append("    switch (op) {")
    for op in all_ops_meta:
        caps_lines.append(f"        case Op::{op['name']}: return {op['name'].lower()}_order;")
    caps_lines.extend(["    default: break;", "    }", "    static const std::vector<Backend> custom_only = {Backend::Custom};", "    return custom_only;", "}"])
    caps_lines.append("")
    caps_lines.append("inline DTypeMask get_supported_dtypes(Op op, Backend backend) {")
    caps_lines.append("    if (backend == Backend::Custom) return kAllDTypes;")
    caps_lines.append("    switch (op) {")
    for op in all_ops_meta:
        caps_lines.append(f"        case Op::{op['name']}:")
        caps_lines.append("            if (backend == Backend::CuBLAS || backend == Backend::CuTensor || backend == Backend::CuDNN || backend == Backend::CuRAND)")
        caps_lines.append("                return dtype_bit(DType::BF16) | dtype_bit(DType::FP32) | dtype_bit(DType::FP16);")
        caps_lines.append("            break;")
    caps_lines.extend(["    default: break;", "    }", "    return 0;", "}", "", "} // namespace yadrakova::core"])
    (output_dir / "ops_caps.hpp").write_text("\n".join(caps_lines), encoding="utf-8")

    # 3. all_ops.hpp (include maestro)
    all_ops_lines = ["// AUTO-GENERATED", "#pragma once"]
    for op in all_ops_meta:
        all_ops_lines.append(f'#include "{op["kernel_name"]}_ops.hpp"')
    (output_dir / "all_ops.hpp").write_text("\n".join(all_ops_lines), encoding="utf-8")

    # 4. tensor_declarations.hpp (para incluir DENTRO de la clase Tensor)
    decl_lines = ["// AUTO-GENERATED: Incluir DENTRO de la clase Tensor"]
    for op in all_ops_meta:
        inputs = op.get("inputs", [])
        if len(inputs) > 1:
            args = ", ".join([f"const Tensor<T>& {inp}" for inp in inputs[1:]] + ["Backend backend = Backend::Auto", "Stream& stream = default_stream()"])
            decl_lines.append(f"    Tensor<T> {op['name'].lower()}({args}) const;")
        else:
            decl_lines.append(f"    Tensor<T> {op['name'].lower()}(Backend backend = Backend::Auto, Stream& stream = default_stream()) const;")
    (output_dir / "tensor_declarations.hpp").write_text("\n".join(decl_lines), encoding="utf-8")


# =========================================================================
# main()
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="Override for the output directory of the registration .cpp files")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    archs = config["cuda"]["architectures"]
    dtypes = config["kernels"]["dtypes"]
    src_dir = Path(config["kernels"]["source_dir"])
    cuh_out_dir = Path(config["kernels"]["output_dir"])
    cpp_out_dir = Path(args.output_dir) if args.output_dir else Path("generated/kernels")

    registry_header = Path(config["kernels"].get("registry_header", "include/kernels/registry.hpp")).resolve()
    dispatch_registry_header = Path(config["kernels"].get("dispatch_registry_header", "include/core/executor.hpp")).resolve()
    cache_dir = Path(config["cache"]["dir"])
    shared_headers_dir = Path("include/kernels")
    include_dirs = [Path("include")]

    try:
        nvcc_ver = nvcc_version()
    except FileNotFoundError:
        print("ERROR: nvcc not found in PATH.", file=sys.stderr)
        sys.exit(1)

    sources = sorted(src_dir.glob("*.cu"))
    if not sources:
        print(f"No .cu files in {src_dir}")
        return

    shared_headers_hash = hash_shared_headers(shared_headers_dir)

    # Directorio de ops generadas
    ops_output_dir = cpp_out_dir.parent / "ops"
    ops_output_dir.mkdir(parents=True, exist_ok=True)

    # Lista para recolectar metadatos de todas las ops
    all_ops_meta = []

    for source in sources:
        kernel_name = source.stem
        print(f"[kernel] {kernel_name}")

        yaml_path = find_dispatch_yaml(source)
        kernel_yaml = None
        if yaml_path is not None:
            with open(yaml_path, encoding="utf-8") as f:
                kernel_yaml = yaml.safe_load(f)

        kernel_dtypes = kernel_yaml.get("dtypes", dtypes) if kernel_yaml else dtypes

        # Compilar cubins
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
            print(f"  [dispatch] no {kernel_name}.yaml -- skipping ...")
        else:
            source_text = source.read_text(encoding="utf-8")
            dispatch_cpp_path = cpp_out_dir / f"{kernel_name}_dispatch.generated.cpp"
            write_dispatch_cpp(kernel_name, kernel_yaml, source_text, dispatch_cpp_path, dispatch_registry_header)
            print(f"  -> {dispatch_cpp_path}")

            # Si el YAML tiene sección 'op:', generar la API de alto nivel
            if "op" in kernel_yaml:
                op_meta = kernel_yaml["op"].copy()
                op_meta["kernel_name"] = kernel_name
                all_ops_meta.append(op_meta)

                _, sig = parse_kernel_signature(source_text)
                generate_op_hpp(kernel_name, op_meta, sig, ops_output_dir)
                print(f"  -> {ops_output_dir}/{kernel_name}_ops.hpp")

    # Generar archivos globales de Ops al final
    if all_ops_meta:
        generate_global_op_files(all_ops_meta, ops_output_dir)
        print(f"[ops] Generated global metadata, caps, and declarations in {ops_output_dir}")

    # Prune final
    valid_names = {s.stem for s in sources}
    prune_orphaned_kernels(cache_dir, valid_names, set(dtypes))


if __name__ == "__main__":
    main()
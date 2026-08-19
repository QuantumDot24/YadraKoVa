#!/usr/bin/env python3
"""
YadraKoVa kernel compiler + embedder + dispatch codegen.

Scans src/kernels/*.cu, compiles each to .cubin (native SASS, zero
load overhead -- no JIT) for every arch x dtype combination defined in
config.yaml, caches by content+flags+nvcc version+shared headers hash
to avoid unnecessary recompilation.

Cache policy: ONLY the most recent version of each kernel+dtype+arch
combination is kept. On a cache miss (the kernel changed), any old
.cubin for that same combination is automatically deleted before
compiling the new one -- designed for constant performance tuning
iteration without accumulating garbage forever.

Generates THREE files per kernel, separating data from side effects:
  - include/kernels/embedded/{kernel}_embedded.cuh
        Data only (byte arrays). Pure header, no side effects.
  - generated/kernels/{kernel}_registration.cpp
        Side effect: static registration in KernelRegistry
        (name -> CUfunction). Lives OUTSIDE include/, in the build
        dir, generated and discardable.
  - generated/kernels/{kernel}_dispatch.generated.cpp   [NEW]
        Side effect: static registration in DispatchRegistry
        (name -> grid/block computation rule), derived from
        src/kernels/{kernel}.yaml. If the .yaml doesn't exist, it's
        skipped with a warning -- allows migrating kernels one by one.

The dispatch .yaml lives next to the .cu (same stem, same folder) and
does NOT participate in the cubin compilation cache hash: changing only
the dispatch rule should not force a CUDA recompilation.
"""
import argparse
import hashlib
import os
import re
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
    """Combined hash of ALL shared .cuh headers in include/kernels/
    (common.cuh, etc. -- not the generated ones in embedded/, those are
    output, not input). Intentionally conservative: if ANY shared header
    changes, invalidates the cache of ALL kernels. Safer than an #include
    parser that can fail silently."""
    h = hashlib.sha256()
    if not shared_headers_dir.exists():
        return h.hexdigest()
    for header in sorted(shared_headers_dir.glob("*.cuh")):
        h.update(header.read_bytes())
    return h.hexdigest()


def prune_old_versions(source: Path, dtype: str, arch: str,
                       cache_dir: Path, keep_path: Path):
    """Deletes any old .cubin for this SAME kernel+dtype+arch combination
    (different hash = previous version of the same file). Called ONLY on
    cache miss, just before compiling the new version -- guarantees that
    .kernel_cache/ never accumulates more than one version per
    combination, regardless of how many times the kernel is iterated
    during tuning."""
    if not cache_dir.exists():
        return
    prefix = f"{source.stem}_{dtype}_{arch}_"
    for old_cubin in cache_dir.glob(f"{prefix}*.cubin"):
        if old_cubin != keep_path:
            print(f"  [cache prune] old version: {old_cubin.name}")
            old_cubin.unlink()


def extract_kernel_name(cubin_stem: str, known_dtypes: set) -> str:
    """Extracts kernel name from a cubin stem that may contain dtype.
    Looks for pattern '_<dtype>_' and takes everything before it."""
    for dtype in known_dtypes:
        marker = f"_{dtype}_"
        if marker in cubin_stem:
            return cubin_stem[:cubin_stem.index(marker)]
    # fallback: split by '_' and hope there's no dtype in the name (unreliable)
    return cubin_stem.split("_")[0]


def prune_orphaned_kernels(cache_dir: Path, valid_kernel_names: set, known_dtypes: set):
    """Deletes .cubin files whose kernel .cu no longer exists (renamed or
    deleted). Called ONCE at the end, over the entire cache."""
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
    """Writes the .cuh -- data ONLY, zero side effects."""
    lines = [
        "// AUTO-GENERATED by scripts/compile_kernel.py -- DO NOT EDIT BY HAND.",
        f"// Source: src/kernels/{kernel_name}.cu",
        "// Data only (embedded cubins). No side effects --",
        "// safe to include from any translation unit.",
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
    """Writes the registration .cpp -- ONLY the side effect.
    Lives outside include/, in the build dir."""
    cuh_include = Path(os.path.relpath(cuh_path, registration_cpp_path.parent)).as_posix()
    registry_include = Path(os.path.relpath(registry_header, registration_cpp_path.parent)).as_posix()

    lines = [
        "// AUTO-GENERATED by scripts/compile_kernel.py -- DO NOT EDIT BY HAND.",
        "// Side effect: static registration in KernelRegistry.",
        "// Lives outside include/ on purpose -- a .cpp with side",
        "// effects does not belong in the header tree.",
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
# NEW: dispatch codegen (grid/block) from src/kernels/{k}.yaml
# --------------------------------------------------------------------------

def _ceildiv_expr(numer: str, denom: int) -> str:
    """ceil(numer/denom) in integer arithmetic, no floats. numer is a
    variable name (M, N, K, or any dim name) that exists as an int64_t
    parameter of the generated dispatch function; denom is a literal
    (block size), known at generation time."""
    return f"(({numer} + {denom} - 1) / {denom})"


# Each helper declares which grid dimensions (x/y/z) each of its
# positional arguments maps to. tile2d(rows, columns) is the common
# case of "one thread per element of a 2D output matrix".
GRID_HELPERS = {
    "tile2d": ("y", "x"),
    "rows": ("y",),
    "cols": ("x",),
    "elementwise": ("x",),
}


def parse_grid_spec(grid_field, block: list, dim_names: list) -> dict:
    """Returns {'x': cpp_expr, 'y': cpp_expr, 'z': cpp_expr}, expressed
    in terms of the dimension names declared in the YAML (dim_names),
    with the block size already resolved as an integer literal."""
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
                    f"grid: '{var_name}' is not declared in 'dims: {dim_names}' -- check {grid_field}"
                )
            dims[dim] = _ceildiv_expr(var_name, block_by_dim[dim])
        return dims

    if isinstance(grid_field, dict):
        # Explicit form, fallback for cases that don't fit a helper:
        # each component is a hand-written C++ expression, free to use
        # the dimension names from dims.
        for dim in ("x", "y", "z"):
            if dim in grid_field:
                dims[dim] = str(grid_field[dim])
        return dims

    raise ValueError(f"grid: unsupported type ({type(grid_field).__name__}); use string or dict")


def find_dispatch_yaml(source: Path) -> Path | None:
    yaml_path = source.with_suffix(".yaml")
    return yaml_path if yaml_path.exists() else None


SIG_RE = re.compile(
    r'__global__\s+void\s+\w+_kernel\s*\((.*?)\)\s*\{',
    re.DOTALL
)


def parse_kernel_signature(source_text: str) -> list[tuple[str, str]]:
    """Returns [(cpp_type, name), ...] in order, from the actual signature."""
    m = SIG_RE.search(source_text)
    if not m:
        raise ValueError("Could not find __global__ ...(...) { in the kernel")
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
            f"'{kernel_name}': YAML declares {len(yaml_args)} args, "
            f"actual signature has {len(real_sig)}"
        )
    for (real_type, real_name), decl in zip(real_sig, yaml_args):
        if real_name != decl["name"]:
            raise ValueError(
                f"'{kernel_name}': order/name mismatch -- YAML says "
                f"'{decl['name']}', actual signature says '{real_name}'"
            )
        if decl["kind"] in ("in", "out") and "*" not in real_type:
            raise ValueError(
                f"'{kernel_name}.{real_name}': YAML says kind={decl['kind']} "
                f"(should be a pointer) but actual signature is '{real_type}'"
            )
        expected_scalar = {"int64": "int64_t", "int32": "int"}.get(decl.get("dtype", ""))
        if decl["kind"] == "scalar" and expected_scalar and expected_scalar not in real_type:
            raise ValueError(
                f"'{kernel_name}.{real_name}': YAML declares dtype={decl['dtype']} "
                f"but actual signature is '{real_type}'"
            )


def write_dispatch_cpp(kernel_name: str, kernel_yaml: dict, source_text: str,
                       dispatch_cpp_path: Path, dispatch_registry_header: Path):
    """Writes the dispatch registration .cpp. Now validates YAML vs
    actual signature BEFORE generating, and builds the lambda over
    DimMap instead of fixed positional M,N,K."""
    if kernel_yaml.get("kernel", kernel_name) != kernel_name:
        print(f"  [WARN] 'kernel:' in YAML does not match the file name "
              f"({kernel_yaml.get('kernel')!r} vs {kernel_name!r})", file=sys.stderr)

    dim_names = kernel_yaml.get("dims")
    if not dim_names:
        raise ValueError(f"'{kernel_name}.yaml' does not define 'dims:' (e.g., [M, N, K] or [n])")

    yaml_args = kernel_yaml.get("args")
    if not yaml_args:
        raise ValueError(f"'{kernel_name}.yaml' does not define 'args:'")

    validate_args_match(kernel_name, source_text, yaml_args)

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
        f"// Dispatch rule for '{kernel_name}', derived from src/kernels/{kernel_name}.yaml",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="Override for the output directory of the registration "
                             ".cpp files (used by CMake to point to the build dir)")
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
        config["kernels"].get("dispatch_registry_header", "include/core/executor.hpp")
    ).resolve()
    cache_dir = Path(config["cache"]["dir"])

    # Shared headers directory (common.cuh, etc.) -- same level as
    # registry.hpp, used both for nvcc -I and for the invalidation hash.
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
            print(f"  [dispatch] no {kernel_name}.yaml -- skipping ...")
        else:
            source_text = source.read_text(encoding="utf-8")
            dispatch_cpp_path = cpp_out_dir / f"{kernel_name}_dispatch.generated.cpp"
            write_dispatch_cpp(kernel_name, kernel_yaml, source_text, dispatch_cpp_path, dispatch_registry_header)
            print(f"  -> {dispatch_cpp_path}")

    # Final prune: kernels renamed/deleted since the last run.
    valid_names = {s.stem for s in sources}
    prune_orphaned_kernels(cache_dir, valid_names, set(dtypes))


if __name__ == "__main__":
    main()
"""Build a loadable shared library from a Triton kernel.

Origin of the code in this module:
- `_compile_kernel` largely follows Triton's AOT compile flow in
  `python/triton/tools/compile.py`.
- the generated launcher C source comes from Triton's
  `third_party/nvidia/tools/cuda/compile.c` or
  `third_party/amd/tools/hip/compile.c` templates.
- `_backend_c_type_for_signature` uses Triton's backend-specific type mapping from
  `third_party/nvidia/backend/driver.py` or `third_party/amd/backend/driver.py`.

Local code in this module:
- `build_shared` links the final `.so` from Triton's generated C launcher.
- `_copy_generated_source` adds `#include <stdlib.h>` because Triton's generated
  `compile.c` templates call `exit(...)` but do not include that header themselves.
"""

from __future__ import annotations

import binascii
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.tools import compile as triton_compile_tool


# Local wrapper around Triton's AOT artifacts: compile the kernel, materialize
# Triton's generated C launcher source, link the final shared library, then
# delete temporary sources.
def build_shared(
    kernel,
    *,
    signature: dict[str, str],
    constexpr: dict[str, object],
    grid: tuple[int, int, int] = (1, 1, 1),
    target: str = "hip:gfx90a:64",
    num_warps: int = 1,
    num_stages: int = 1,
    build_dir: str = ".",
    out: str = "librdma_triton_decoder.so",
    compiler: str = "",
    cflags: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """Compile a Triton kernel and package it into a shared library."""
    if len(grid) != 3:
        raise ValueError(f"grid must have exactly 3 dimensions, got {grid!r}")
    if target.count(":") != 2:
        raise ValueError(f"target must look like 'backend:arch:warp', got {target!r}")

    build_path = Path(build_dir).resolve()
    build_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend, generated_symbol, generated_c = _compile_kernel(
        kernel,
        signature=signature,
        constexpr=constexpr,
        grid=grid,
        target=target,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    with tempfile.TemporaryDirectory(dir=build_path, prefix=".triton_shared_") as td:
        scratch_path = Path(td)
        device_c = scratch_path / "device_kernel_aot.c"

        _copy_generated_source(generated_c, device_c)

        if backend == "cuda":
            compiler = compiler or os.environ.get("NVCC", "nvcc")
        elif backend == "hip":
            compiler = compiler or os.environ.get("HIPCC", "hipcc")
        else:
            raise ValueError(
                f"unsupported backend for shared-library compilation: {backend}"
            )

        cmd = [
            compiler,
            "-shared",
            "-O2",
            "-o",
            str(out_path),
            str(device_c),
        ]
        if backend == "cuda":
            cmd[1:1] = ["-Xcompiler", "-fPIC"]
            cmd.append("-lcuda")
        elif backend == "hip":
            cmd[1:1] = ["-fPIC"]
        else:
            raise ValueError(
                f"unsupported backend for shared-library compilation: {backend}"
            )
        cmd.extend(cflags)
        subprocess.run(cmd, check=True)

    return out_path, generated_symbol


# Mostly adapted from Triton's `triton.tools.compile` flow. This function stops
# after rendering Triton's generated launcher source and returns that artifact
# to the local shared-library wrapper above.
def _compile_kernel(
    kernel,
    *,
    signature: dict[str, str],
    constexpr: dict[str, object],
    grid: tuple[int, int, int],
    target: str,
    num_warps: int,
    num_stages: int,
) -> tuple[str, str, Path]:
    """AOT-compile a Triton kernel and return the generated launcher source."""
    backend, _, _ = target.split(":", 2)
    runtime_arg_names = [name for name in kernel.arg_names if name not in constexpr]
    signature = {
        arg_name: "constexpr"
        if arg_name in constexpr
        else signature[arg_name].split(":", 1)[0].strip()
        for arg_name in kernel.arg_names
    }
    constants = dict(constexpr)

    # Adapted from Triton's triton.tools.compile: binder setup + ASTSource construction.
    kernel.create_binder()
    src = kernel.ASTSource(
        fn=kernel, constexprs=constants, signature=signature, attrs={}
    )

    # Adapted from Triton's triton.tools.compile: target/backend/options setup and compile call.
    target_obj = _make_target(target)
    backend_impl = triton.compiler.make_backend(target_obj)
    options = backend_impl.parse_options(
        {"num_warps": num_warps, "num_stages": num_stages}
    )
    ccinfo = triton.compile(src, target=target_obj, options=options.__dict__)

    # Copied verbatim from Triton's triton.tools.compile: this MVP only supports zero-scratch kernels.
    if getattr(ccinfo.metadata, "global_scratch_size", 0) > 0:
        raise RuntimeError(
            "AOT compiling kernels with global scratch requirements is not yet implemented"
        )
    if ccinfo.metadata.profile_scratch_size > 0:
        raise RuntimeError(
            "AOT compiling kernels with profile scratch requirements is not yet implemented"
        )

    func_name = kernel.__name__
    asm = ccinfo.asm[backend_impl.binary_ext]
    hex_ = str(binascii.hexlify(asm))[2:-1]
    runtime_signature = ", ".join(
        f"{_backend_c_type_for_signature(backend, signature[name])} {name}"
        for name in runtime_arg_names
    )

    # Adapted from Triton's triton.tools.compile: params fed into Triton's compile.* templates.
    params = {
        "kernel_name": func_name,
        "triton_kernel_name": kernel.__name__,
        "bin_size": len(asm),
        "bin_data": ", ".join(f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])),
        "signature": runtime_signature,
        "full_signature": runtime_signature,
        "arg_pointers": ", ".join(
            [f"&{arg}" for arg in runtime_arg_names]
            + ["&global_scratch", "&profile_scratch"]
        ),
        "num_args": len(runtime_arg_names) + 2,
        "kernel_docstring": "",
        "shared": ccinfo.metadata.shared,
        "num_warps": num_warps,
        "algo_info": func_name,
        "gridX": str(grid[0]),
        "gridY": str(grid[1]),
        "gridZ": str(grid[2]),
        "_placeholder": "",
        "warp_size": target_obj.warp_size,
        "backend_name": target_obj.backend,
    }

    # Adapted from Triton's triton.tools.compile: render triton/tools/extra/<backend>/compile.*.
    with tempfile.TemporaryDirectory(prefix="triton_shared_aot_") as td:
        tmpdir = Path(td)
        out_base = tmpdir / kernel.__name__
        generated_c = None
        template_dir = (
            Path(triton_compile_tool.__file__).parent / "extra" / target_obj.backend
        )
        for template_path in template_dir.glob("compile.*"):
            output_file = out_base.with_suffix(template_path.suffix)
            output_file.write_text(template_path.read_text().format(**params))
            if template_path.suffix == ".c":
                generated_c = output_file

        if generated_c is None:
            raise RuntimeError(
                f"expected Triton compile templates to generate .c in {template_dir}"
            )

        c_fd, c_path = tempfile.mkstemp(prefix="triton_shared_", suffix=".c")
        os.close(c_fd)
        kept_c = Path(c_path)
        shutil.copyfile(generated_c, kept_c)

    return backend, func_name, kept_c


def _make_target(target: str) -> GPUTarget:
    """Parse a backend:arch:warp string into a Triton GPUTarget."""
    backend, arch, warp_size = target.split(":", 2)
    arch_value: int | str = int(arch) if backend == "cuda" and arch.isdigit() else arch
    return GPUTarget(backend=backend, arch=arch_value, warp_size=int(warp_size))


def _copy_generated_source(source_path: Path, dest_path: Path) -> None:
    """Copy Triton's generated C source and patch in ``stdlib.h`` if needed."""
    source_text = source_path.read_text()
    if "#include <stdlib.h>" not in source_text:
        source_text = "#include <stdlib.h>\n" + source_text
    dest_path.write_text(source_text)


def _backend_c_type_for_signature(backend: str, signature: str) -> str:
    """Map a Triton signature type using Triton's backend driver helpers."""
    signature = signature.split(":", 1)[0].strip()
    if backend == "cuda":
        from triton.backends.nvidia.driver import ty_to_cpp as ty_to_c_decl
    elif backend == "hip":
        from triton.backends.amd.driver import ty_to_cpp as ty_to_c_decl
    else:
        raise ValueError(f"unsupported backend for type mapping: {backend}")
    return ty_to_c_decl(signature)

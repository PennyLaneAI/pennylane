# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are derived from Triton:
# - _compile_kernel: python/triton/tools/compile.py
#
# Triton is licensed under the MIT License:
#
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Build a loadable shared library from a Triton kernel."""

from __future__ import annotations

import binascii
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

try:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.tools import compile as triton_compile_tool
except ImportError as exc:
    raise ImportError("Triton decoders require installed `triton` Python package.") from exc


class _HashableConstexprTuple(tuple):
    """Tuple wrapper with a recursive Triton cache key.

    This is needed because Triton caches compiled kernels using ``ASTSource.hash()``, which
    only consults ``cache_key`` on the top-level constexpr object, so nested ``JITFunction``
    values inside a plain tuple fall back to ``str(...)`` and can collide.
    """

    @property
    def cache_key(self) -> str:
        """Return a recursive cache key for nested constexpr tuples."""
        return str(tuple(_constexpr_cache_key_part(value) for value in self))


def _constexpr_cache_key_part(value: object) -> object:
    """Build a stable cache-key fragment for nested constexpr values."""
    if isinstance(value, tuple):  # pragma: no cover
        return tuple(_constexpr_cache_key_part(item) for item in value)
    return value.cache_key if hasattr(value, "cache_key") else value


def _wrap_constexpr(value: object) -> object:
    """Wrap tuple constexprs so Triton hashes nested JIT functions by cache key."""
    if isinstance(value, tuple) and not hasattr(value, "cache_key"):
        return _HashableConstexprTuple(_wrap_constexpr(item) for item in value)
    return value


def _validate_platform(platform: str) -> tuple[str, str, int]:
    """Validate a Triton platform string and return its parsed components."""
    if platform.count(":") != 2:
        raise ValueError(f"platform must look like 'backend:arch:warp', got {platform!r}")

    backend, arch, warp_size_text = platform.split(":", 2)
    if backend not in {"cuda", "hip"}:
        raise ValueError(f"platform backend must be 'cuda' or 'hip', got {backend!r}")
    if not arch:
        raise ValueError("platform arch must be non-empty")

    try:
        warp_size = int(warp_size_text)
    except ValueError as exc:
        raise ValueError(f"platform warp size must be an integer, got {warp_size_text!r}") from exc

    expected_warp_size = 32 if backend == "cuda" else 64
    if warp_size != expected_warp_size:
        raise ValueError(
            f"platform warp size must be {expected_warp_size} for {backend}, got {warp_size}"
        )

    return backend, arch, warp_size


def _make_catalyst_wrapper_source(backend: str, raw_symbol: str) -> str:
    """Build a Catalyst-compatible launcher wrapper around a raw Triton symbol."""
    wrapper_symbol = f"{raw_symbol}_catalyst"
    # HIP is the portable source language, but Triton's generated host launcher
    # ABI is still backend-specific: CUDA emits CUDA Driver API entrypoints
    # (CUstream/CUdeviceptr/CUresult), while HIP emits hip* types.
    if backend == "cuda":
        stream_type = "CUstream"
        deviceptr_type = "CUdeviceptr"
    elif backend == "hip":  # pragma: no cover
        stream_type = "hipStream_t"
        deviceptr_type = "hipDeviceptr_t"
    else:
        raise ValueError(f"unsupported backend for Catalyst wrapper: {backend}")

    return textwrap.dedent(f"""

        typedef struct {{
            void *ring;
            uint32_t ring_slots;
            void *handoff;
            void *stop;
            uint64_t total;
            void *stream;
        }} CoprocLaunchDescCompat;

        int {wrapper_symbol}(const CoprocLaunchDescCompat *desc, void *ctx) {{
            (void)ctx;
            if (!desc || !desc->ring || !desc->handoff || !desc->stream) {{
                return 1;
            }}
            if (desc->ring_slots == 0 || (desc->ring_slots & (desc->ring_slots - 1)) != 0) {{
                return 1;
            }}

            int rc = {raw_symbol}(
                ({stream_type})desc->stream,
                ({deviceptr_type})(uintptr_t)desc->ring,
                ({deviceptr_type})(uintptr_t)desc->handoff,
                ({deviceptr_type})(uintptr_t)desc->stop,
                desc->ring_slots,
                desc->total
            );
            return rc == 0 ? 0 : 1;
        }}
        """)


def _build_so(  # pylint: disable=too-many-arguments
    kernel,
    *,
    signature: dict[str, str],
    constexpr: dict[str, object],
    grid: tuple[int, int, int],
    platform: str,
    num_warps: int,
    num_stages: int,
    out: str,
    compiler: str,
    cflags: tuple[str, ...],
) -> tuple[Path, str]:
    """Compile a Triton kernel and package it into a shared library.

    Args:
        kernel (Callable): Triton kernel to compile.

    Keyword Args:
        signature (dict[str, str]): Runtime signature for the kernel arguments.
        constexpr (dict[str, object]): Compile-time kernel arguments.
        grid (tuple[int, int, int]): Triton kernel launch grid dimensions.
        platform (str): Triton platform string of the form
            ``"backend:arch:warp_size"``.
        num_warps (int): Triton kernel launch warp count.
        num_stages (int): Triton pipeline stage count.
        out (str): Destination path for the generated shared library.
        compiler (str): Optional compiler executable override.
        cflags (tuple[str, ...]): Extra flags passed to the compiler.

    Returns:
        tuple[Path, str]: Path to the shared library and the exported
            Catalyst-compatible launcher symbol.

    Raises:
        ValueError: If ``grid`` or ``platform`` is malformed.
        subprocess.CalledProcessError: If the backend compiler fails.
    """
    if len(grid) != 3:
        raise ValueError(f"grid must have exactly 3 dimensions, got {grid!r}")
    backend, _, _ = _validate_platform(platform)

    out_path = Path(out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_path = out_path.parent

    backend, generated_symbol, generated_c = _compile_kernel(
        kernel,
        signature=signature,
        constexpr=constexpr,
        grid=grid,
        platform=platform,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    with tempfile.TemporaryDirectory(dir=build_path, prefix=".triton_shared_") as temp_dir:
        scratch_path = Path(temp_dir)
        device_c = scratch_path / "device_kernel_aot.c"

        # Triton's generated ``compile.c`` calls ``exit(...)`` but doesn't
        # include ``<stdlib.h>``, so patch it here.
        source_text = generated_c.read_text(encoding="utf-8")
        if "#include <stdlib.h>" not in source_text:
            source_text = "#include <stdlib.h>\n" + source_text
        source_text += _make_catalyst_wrapper_source(backend, generated_symbol)
        device_c.write_text(source_text, encoding="utf-8")

        # TODO: in principle we only need a host compiler, however we need to includes and libs
        # use nvcc and hipcc for now
        hip_include_dir = None
        if backend == "cuda":
            compiler = compiler or os.environ.get("NVCC", "nvcc")
        elif backend == "hip":  # pragma: no cover
            compiler = compiler or os.environ.get("HIPCC", "hipcc")
            candidate = Path(triton.__file__).resolve().parent / "backends" / "amd" / "include"
            if (candidate / "hip" / "hip_runtime.h").exists():
                hip_include_dir = candidate

        cmd = [
            compiler,
            "-fPIC",
            "-shared",
            "-O3",
            "-o",
            str(out_path),
            str(device_c),
        ]
        if Path(compiler).name == "nvcc":
            cmd.insert(1, "-Xcompiler")
        if backend == "cuda":
            cmd.append("-lcuda")
        elif hip_include_dir is not None:  # pragma: no cover
            cmd.append(f"-I{hip_include_dir}")
        cmd.extend(cflags)
        subprocess.run(cmd, check=True)

    return out_path, f"{generated_symbol}_catalyst"


def _compile_kernel(  # pylint: disable=too-many-arguments
    kernel,
    *,
    signature: dict[str, str],
    constexpr: dict[str, object],
    grid: tuple[int, int, int],
    platform: str,
    num_warps: int,
    num_stages: int,
) -> tuple[str, str, Path]:
    """AOT-compile a Triton kernel and keep the generated launcher source.

    Args:
        kernel (Callable): Triton kernel to compile.

    Keyword Args:
        signature (dict[str, str]): Runtime signature for the kernel arguments.
        constexpr (dict[str, object]): Compile-time kernel arguments.
        grid (tuple[int, int, int]): Triton kernel launch grid dimensions.
        platform (str): Triton platform string of the form
            ``"backend:arch:warp_size"``.
        num_warps (int): Triton kernel launch warp count.
        num_stages (int): Triton pipeline stage count.

    Returns:
        tuple[str, str, Path]: Backend name, exported launcher symbol, and path
            to the generated C launcher source.

    Raises:
        RuntimeError: If Triton reports unsupported scratch-space requirements
            or does not emit the expected launcher source.
    """
    backend, arch, warp_size = _validate_platform(platform)
    runtime_arg_names = [name for name in kernel.arg_names if name not in constexpr]
    signature = {
        arg_name: (
            "constexpr" if arg_name in constexpr else signature[arg_name].split(":", 1)[0].strip()
        )
        for arg_name in kernel.arg_names
    }
    # Triton's ASTSource.hash() only checks cache_key on top-level constexpr objects.
    # Wrap tuples so nested JIT functions contribute their own cache keys.
    constants = {name: _wrap_constexpr(value) for name, value in constexpr.items()}

    # Adapted from Triton's python/triton/tools/compile.py:compile_kernel
    kernel.create_binder()
    ast_source = kernel.ASTSource(fn=kernel, constexprs=constants, signature=signature, attrs={})

    target_arch = int(arch) if backend == "cuda" and arch.isdigit() else arch
    target_obj = GPUTarget(backend, target_arch, warp_size)
    backend_impl = triton.compiler.make_backend(target_obj)
    options = backend_impl.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    compile_result = triton.compile(ast_source, target=target_obj, options=options.__dict__)

    if getattr(compile_result.metadata, "global_scratch_size", 0) > 0:
        raise RuntimeError(
            "AOT compiling kernels with global scratch requirements is not yet implemented"
        )
    if compile_result.metadata.profile_scratch_size > 0:
        raise RuntimeError(
            "AOT compiling kernels with profile scratch requirements is not yet implemented"
        )

    func_name = f"{kernel.__name__}_{ast_source.hash()[:12]}"
    kernel_binary = compile_result.asm[backend_impl.binary_ext]
    binary_hex = str(binascii.hexlify(kernel_binary))[2:-1]
    if backend == "cuda":
        from triton.backends.nvidia.driver import (  # pylint: disable=import-outside-toplevel,no-name-in-module
            ty_to_cpp,
        )
    elif backend == "hip":  # pragma: no cover
        from triton.backends.amd.driver import (  # pylint: disable=import-outside-toplevel,no-name-in-module
            ty_to_cpp,
        )
    else:  # pragma: no cover
        # Unreachable: _validate_platform already rejects backends outside {"cuda", "hip"}.
        raise ValueError(f"unsupported backend for type mapping: {backend}")
    runtime_signature = ", ".join(
        f"{ty_to_cpp(signature[name].split(':', 1)[0].strip())} {name}"
        for name in runtime_arg_names
    )

    params = {
        "kernel_name": func_name,
        "triton_kernel_name": kernel.__name__,
        "bin_size": len(kernel_binary),
        "bin_data": ", ".join(
            f"0x{high_nibble}{low_nibble}"
            for high_nibble, low_nibble in zip(binary_hex[::2], binary_hex[1::2])
        ),
        "signature": runtime_signature,
        "full_signature": runtime_signature,
        "arg_pointers": ", ".join(
            [f"&{arg}" for arg in runtime_arg_names] + ["&global_scratch", "&profile_scratch"]
        ),
        "num_args": len(runtime_arg_names) + 2,
        "kernel_docstring": "",
        "shared": compile_result.metadata.shared,
        "num_warps": num_warps,
        "algo_info": func_name,
        "gridX": str(grid[0]),
        "gridY": str(grid[1]),
        "gridZ": str(grid[2]),
        "_placeholder": "",
        "warp_size": target_obj.warp_size,
        "backend_name": target_obj.backend,
    }

    with tempfile.TemporaryDirectory(prefix="triton_shared_aot_") as temp_dir:
        tmpdir = Path(temp_dir)
        out_base = tmpdir / func_name
        generated_c = None
        template_dir = Path(triton_compile_tool.__file__).parent / "extra" / target_obj.backend
        for template_path in template_dir.glob("compile.*"):
            output_file = out_base.with_suffix(template_path.suffix)
            output_file.write_text(
                template_path.read_text(encoding="utf-8").format(**params),
                encoding="utf-8",
            )
            if template_path.suffix == ".c":
                generated_c = output_file

        if generated_c is None:
            raise RuntimeError(
                f"expected Triton compile templates to generate .c in {template_dir}"
            )

        temp_file_descriptor, temp_source_path = tempfile.mkstemp(
            prefix="triton_shared_", suffix=".c"
        )
        os.close(temp_file_descriptor)
        kept_c = Path(temp_source_path)
        shutil.copyfile(generated_c, kept_c)

    return backend, func_name, kept_c

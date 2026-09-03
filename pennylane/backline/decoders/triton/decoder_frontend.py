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

"""Frontend helpers for building Triton decoder shared libraries for backline devices."""

from __future__ import annotations

import shutil
import tempfile
import types
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

try:
    import triton
    import triton.language as tl

    from .algorithms import _decode_one
    from .persistent_kernel import _persistent_decoder_kernel
    from .triton_so_builder import _build_so, _validate_platform
except ImportError as exc:
    raise ImportError("Triton decoders require installed `triton` Python package.") from exc


def _clone_with_name(fn: object, name: str) -> object:
    """Clone a Python function object while overriding its name metadata."""
    cloned = types.FunctionType(fn.__code__, fn.__globals__, name, fn.__defaults__, fn.__closure__)
    cloned.__kwdefaults__ = getattr(fn, "__kwdefaults__", None)
    cloned.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    cloned.__doc__ = getattr(fn, "__doc__", None)
    cloned.__module__ = getattr(fn, "__module__", None)
    cloned.__qualname__ = name
    cloned.__dict__.update(getattr(fn, "__dict__", {}))
    return cloned


def _triton_jit_with_unique_names(decoder_fns: tuple[object, ...]) -> tuple[object, ...]:
    """Return Triton JIT kernels with unique names from un-jitted Triton functions."""
    normalized = []
    for i, decoder_fn in enumerate(decoder_fns):
        if not isinstance(decoder_fn, types.FunctionType):
            raise TypeError(
                "decoder_fns must contain Python function objects; "
                "already-jitted Triton functions fail this check, so pass undecorated "
                "functions to triton_decoder"
            )
        unique_name = f"{decoder_fn.__name__}_{i}"
        normalized.append(triton.jit(_clone_with_name(decoder_fn, unique_name)))
    return tuple(normalized)


def _build_triton_decoder(  # pylint: disable=too-many-arguments
    decoder_fns: tuple[object, ...],
    *,
    platform: str,
    grid: tuple[int, int, int] = (1, 1, 1),
    num_warps: int = 1,
    num_stages: int = 1,
    compiler: str = "",
    cflags: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """Build a shared library based on the decoder functions provided.

    Args:
        decoder_fns (tuple[object, ...]): Tuple of un-jitted Triton decoder functions. Each
            entry is jitted with a unique generated name, and ``decoder_id`` selects the
            appropriate decoder function at runtime.

    Keyword Args:
        platform (str): Required Triton platform string of the form
            ``"backend:arch:warp_size"``. For example, ``"hip:gfx942:64"`` or
            ``"cuda:80:32"``.
        grid (tuple[int, int, int]): Triton kernel launch grid dimensions.
        num_warps (int): Triton kernel launch warp count.
        num_stages (int): Triton pipeline stage count.
        compiler (str): Optional compiler executable override.
        cflags (tuple[str, ...]): Extra compiler flags.

    Returns:
        tuple[Path, str]: Path to the compiled shared library in a temporary location and the
            Catalyst-compatible exported launcher symbol. The caller is responsible for the
            returned shared library's lifetime and cleanup.

    Raises:
        TypeError: If ``decoder_fns`` contains entries other than un-jitted Triton functions.
        ValueError: If the build options are invalid.

    **Example**

    >>> import triton.language as tl
    >>> from pennylane.backline.decoders.triton.decoder_frontend import _build_triton_decoder
    >>> def steane_lookup(syndrome):
    ...     one = tl.cast(1, tl.uint64)
    ...     zero = tl.cast(0, tl.uint64)
    ...     return tl.where(syndrome != 0, one << (syndrome - 1), zero)
    >>> so_path, symbol_name = _build_triton_decoder(  # doctest: +SKIP
    ...     (steane_lookup, steane_lookup),
    ...     platform="hip:gfx942:64",
    ... )
    """
    _validate_build_options(
        decoder_fns=decoder_fns,
        num_warps=num_warps,
        num_stages=num_stages,
        platform=platform,
    )
    decoder_fns = _triton_jit_with_unique_names(decoder_fns)
    tmpdir = Path(tempfile.mkdtemp(prefix="pl_triton_decoder_"))
    out = tmpdir / "librdma_triton_decoder.so"
    try:
        return _build_so(
            _persistent_decoder_kernel,
            signature={
                "ring_u64_ptr": "*u64",
                "handoff_u64_ptr": "*u64",
                "stop_u32_ptr": "*u32",
                "ring_slots": "u32",
                "total": "u64",
            },
            constexpr={
                "decoder_fns": tuple(decoder_fns),
                # The polling loads need both a compiler barrier and a cache bypass,
                # and the two backends injects that differently:
                #   HIP  - ".cv" lowers to an LLVM volatile load (glc / sc0 sc1);
                #          volatile=True alone is dropped and the load
                #          gains !amdgpu.noclobber, so the poll never sees host writes.
                #   CUDA - volatile=True alone gives ld.volatile.global, which the PTX
                #          ISA already performs with the .cv cache operator. Triton
                #          <=3.7 silently dropped a redundant ".cv" here; >=3.8 emits
                #          both and ptxas rejects the combination.
                "cache_mod": ".cv" if platform.strip().split(":", 1)[0] == "hip" else "",
            },
            grid=grid,
            platform=platform.strip(),
            out=str(out.resolve()),
            num_warps=num_warps,
            num_stages=num_stages,
            compiler=compiler,
            cflags=tuple(cflags),
        )
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def _build_css_bp_decoder(  # pylint: disable=too-many-arguments
    Hx: ArrayLike,
    Hz: ArrayLike,
    *,
    postprocess: str,
    num_iters: int,
    prob: float,
    platform: str,
    grid: tuple[int, int, int] = (1, 1, 1),
    num_warps: int = 1,
    num_stages: int = 1,
    compiler: str = "",
    cflags: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """Build a shared library for a triton-based CSS belief-propagation decoder.

    This helper specializes one Triton decoder for ``Hx`` and one for ``Hz``, then packages both
    decoders behind a single dispatcher selected by ``decoder_id`` at runtime.

    .. note::
        The decoder uses packed ``u64`` bitmasks for syndromes and corrections,
        currently limiting to maximum 64 checks/qubits.

    Args:
        Hx (ArrayLike): X parity-check matrix.
        Hz (ArrayLike): Z parity-check matrix.

    Keyword Args:
        postprocess (str): Postprocessing step applied after belief propagation. Use
            ``"hard"`` for hard-decision output or ``"osd"`` for ordered-statistics decoding.
        num_iters (int): Number of belief-propagation iterations.
        prob (float): Uniform prior error probability across qubits.
        platform (str): Required Triton platform string of the form
            ``"backend:arch:warp_size"``. For example, ``"hip:gfx942:64"`` or
            ``"cuda:80:32"``.
        grid (tuple[int, int, int]): Triton kernel launch grid dimensions.
        num_warps (int): Triton kernel launch warp count.
        num_stages (int): Triton pipeline stage count.
        compiler (str): Optional compiler executable override.
        cflags (tuple[str, ...]): Extra compiler flags.

    Returns:
        tuple[Path, str]: Path to the compiled shared library in a temporary location and the
            Catalyst-compatible exported launcher symbol. The caller is responsible for the
            returned shared library's lifetime and cleanup.

    Raises:
        ValueError: If the decoder options or parity-check matrices are invalid.

    **Example**

    >>> import numpy as np
    >>> Hz = Hx = np.array([
    ...     [1, 0, 1, 0, 1, 0, 1],
    ...     [0, 1, 1, 0, 0, 1, 1],
    ...     [0, 0, 0, 1, 1, 1, 1],
    ... ])
    >>> so_path, symbol_name = _build_css_bp_decoder(  # doctest: +SKIP
    ...     Hx,
    ...     Hz,
    ...     postprocess="osd",
    ...     num_iters=10,
    ...     prob=0.1,
    ...     platform="hip:gfx942:64",
    ... )
    """
    _validate_css_options(postprocess=postprocess, num_iters=num_iters, prob=prob)
    hx = _to_numpy(Hx)
    hz = _to_numpy(Hz)
    decoder_fns = (
        _make_css_decoder(hx, postprocess=postprocess, num_iters=num_iters, prob=prob),
        _make_css_decoder(hz, postprocess=postprocess, num_iters=num_iters, prob=prob),
    )
    return _build_triton_decoder(
        decoder_fns,
        platform=platform,
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        compiler=compiler,
        cflags=cflags,
    )


def _to_numpy(H: ArrayLike) -> np.ndarray:
    """Validate and convert a parity-check matrix to a NumPy array.

    Args:
        H (ArrayLike): Candidate parity-check matrix.

    Returns:
        np.ndarray: Binary parity-check matrix.

    Raises:
        ValueError: If ``H`` is empty, non-binary, not two-dimensional, or
            exceeds the current 64-parity-checks or 64-qubits packing limit.
    """
    matrix = np.asarray(H)
    if matrix.ndim != 2:
        raise ValueError(f"H must be a 2D array, got shape {matrix.shape!r}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"H must be non-empty, got shape {matrix.shape!r}")

    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("H must contain only binary entries 0/1")
    if matrix.shape[0] > 64:
        raise ValueError(f"H has {matrix.shape[0]} checks, but Triton decoder supports at most 64")
    if matrix.shape[1] > 64:
        raise ValueError(f"H has {matrix.shape[1]} qubits, but Triton decoder supports at most 64")
    return matrix


def _validate_build_options(
    *, decoder_fns: tuple[object, ...], num_warps: int, num_stages: int, platform: str
) -> None:
    """Validate generic Triton decoder build options.

    Args:
        decoder_fns (tuple[object, ...]): Un-jitted Triton decoder functions to dispatch.
        num_warps (int): Triton kernel launch warp count.
        num_stages (int): Triton pipeline stage count.
        platform (str): Triton platform string of the form
            ``"backend:arch:warp_size"``.

    Raises:
        ValueError: If any option falls outside the supported range or format.
    """
    if not decoder_fns:
        raise ValueError(
            "decoder_fns must be a non-empty tuple of un-jitted Triton decoder functions"
        )
    if num_warps <= 0:
        raise ValueError("num_warps must be > 0")
    if num_stages <= 0:
        raise ValueError("num_stages must be > 0")
    if platform.strip().count(":") != 2:
        raise ValueError(
            "platform must be a raw Triton platform string like "
            f"'hip:gfx942:64' or 'cuda:80:32', got {platform!r}"
        )
    _validate_platform(platform.strip())


def _validate_css_options(*, postprocess: str, num_iters: int, prob: float) -> None:
    """Validate CSS decoder options.

    Args:
        postprocess (str): Postprocessing rule applied after belief propagation.
        num_iters (int): Number of belief-propagation iterations.
        prob (float): Uniform prior error probability.

    Raises:
        ValueError: If any option falls outside the supported range.
    """
    if postprocess not in {"hard", "osd"}:
        raise ValueError("postprocess must be 'hard' or 'osd'")
    if num_iters <= 0:
        raise ValueError("num_iters must be > 0")
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")


def _make_css_decoder(
    matrix: np.ndarray, *, postprocess: str, num_iters: int, prob: float
) -> object:
    """Specialize one un-jitted Triton decoder function for a fixed parity-check matrix.

    Args:
        matrix (np.ndarray): Binary parity-check matrix.
        postprocess (str): Postprocessing rule applied after belief propagation.
        num_iters (int): Number of belief-propagation iterations.
        prob (float): Uniform prior error probability.

    Returns:
        object: Un-jitted Triton function that maps one packed syndrome to one packed correction
            mask.
    """
    matrix = tl.constexpr(tuple(tuple(int(value) for value in row) for row in matrix.tolist()))
    postprocess = tl.constexpr(postprocess)
    prob = tl.constexpr(prob)
    num_iters = tl.constexpr(num_iters)

    def decode_impl(syndrome):  # pragma: no cover
        return _decode_one(
            syndrome,
            matrix,
            postprocess=postprocess,
            prob=prob,
            num_iters=num_iters,
        )

    return decode_impl

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

"""Frontend helpers for building Triton decoder shared libraries.

This module provides the public entry points for packaging Triton decoder
kernels into shared libraries that can be loaded by backline devices.
"""

# pylint: disable=no-name-in-module,no-member,too-many-arguments

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

try:
    import triton
    import triton.language as tl

    from .algorithms import _decode_one
    from .persistent_kernel import _persistent_decoder_kernel
    from .triton_so_builder import _build_so
except ImportError as exc:
    raise ImportError("Triton decoders require installed `triton` Python package.") from exc


def _build_triton_decoder(
    decoder_fns: tuple[object, ...],
    *,
    platform: str,
    grid: tuple[int, int, int] = (1, 1, 1),
    num_warps: int = 1,
    num_stages: int = 1,
    compiler: str = "",
    cflags: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """Build a shared library based on the triton functions provided.

    Args:
        decoder_fns (tuple[object, ...]): Tuple of Triton decoder functions.
            ``decoder_id`` selects the tuple index at runtime.

    Keyword Args:
        platform (str): Required Triton platform string of the form
            ``"backend:arch:warp_size"``. For example, ``"hip:gfx942:64"`` or
            ``"cuda:80:32"``.
        grid (tuple[int, int, int]): Launch grid baked into the generated launcher source.
            Defaults to ``(1, 1, 1)``.
        num_warps (int): Triton kernel launch warp count. Defaults to ``1``.
        num_stages (int): Triton pipeline stage count. Defaults to ``1``.
        compiler (str): Optional compiler executable override. Defaults to ``""``.
        cflags (tuple[str, ...]): Extra compiler flags. Defaults to ``()``.

    Returns:
        tuple[Path, str]: Path to the compiled shared library in a temporary location and the
            Triton-generated exported entrypoint name. The caller owns the returned file.

    Raises:
        ValueError: If the build options are invalid.

    **Example**

    >>> import triton
    >>> import triton.language as tl
    >>> from pennylane.backline.decoders.triton.decoder_frontend import _build_triton_decoder
    >>> @triton.jit
    ... def steane_lookup(syndrome):
    ...     one = tl.cast(1, tl.uint64)
    ...     zero = tl.cast(0, tl.uint64)
    ...     return tl.where(syndrome != 0, one << (syndrome - 1), zero)
    >>> so_path, symbol_name = _build_triton_decoder(
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
    tmpdir = Path(tempfile.mkdtemp(prefix="pl_triton_decoder_"))
    out = tmpdir / "librdma_triton_decoder.so"
    try:
        return _build_so(
            _persistent_decoder_kernel,
            signature={
                "ring_u64_ptr": "*u64",
                "handoff_u64_ptr": "*u64",
                "stop_u32_ptr": "*u32",
                "total": "u64",
            },
            constexpr={"decoder_fns": tuple(decoder_fns)},
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


def _build_css_bp_decoder(
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
        The generated decoder consumes one packed syndrome bitmask and returns one packed
        correction bitmask, each stored in a single ``u64``. Bit ``i`` corresponds
        to check and qubit ``i`` in the syndrome and correction, respectively. This limits
        the current implementation to at most 64 parity checks and qubits.

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
        grid (tuple[int, int, int]): Launch grid baked into the generated launcher source.
            Defaults to ``(1, 1, 1)``.
        num_warps (int): Triton kernel launch warp count. Defaults to ``1``.
        num_stages (int): Triton pipeline stage count. Defaults to ``1``.
        compiler (str): Optional compiler executable override. Defaults to ``""``.
        cflags (tuple[str, ...]): Extra compiler flags. Defaults to ``()``.

    Returns:
        tuple[Path, str]: Path to the compiled shared library in a temporary location and the
            Triton-generated exported entrypoint name. The caller owns the returned file.

    Raises:
        ValueError: If the decoder options or parity-check matrices are invalid.

    **Example**

    >>> import numpy as np
    >>> Hx = np.array([[1, 0], [0, 1]])
    >>> Hz = np.array([[1, 1], [0, 1]])
    >>> so_path, symbol_name = _build_css_bp_decoder(
    ...     Hx,
    ...     Hz,
    ...     postprocess="hard",
    ...     num_iters=5,
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
        np.ndarray: Binary parity-check matrix with two dimensions.

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
        decoder_fns (tuple[object, ...]): Triton decoder functions to dispatch.
        num_warps (int): Triton kernel launch warp count.
        num_stages (int): Triton pipeline stage count.
        platform (str): Triton platform string of the form
            ``"backend:arch:warp_size"``.

    Raises:
        ValueError: If any option falls outside the supported range or format.
    """
    if not decoder_fns:
        raise ValueError("decoder_fns must be a non-empty tuple of Triton decoder functions")
    if num_warps <= 0:
        raise ValueError("num_warps must be > 0")
    if num_stages <= 0:
        raise ValueError("num_stages must be > 0")
    if platform.strip().count(":") != 2:
        raise ValueError(
            "platform must be a raw Triton platform string like "
            f"'hip:gfx942:64' or 'cuda:80:32', got {platform!r}"
        )


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
    """Specialize one Triton decoder kernel for a fixed parity-check matrix.

    Args:
        matrix (np.ndarray): Binary parity-check matrix.
        postprocess (str): Postprocessing rule applied after belief propagation.
        num_iters (int): Number of belief-propagation iterations.
        prob (float): Uniform prior error probability.

    Returns:
        object: Triton JIT function that maps one packed syndrome to one packed
            correction mask.
    """
    matrix = tl.constexpr(tuple(tuple(int(value) for value in row) for row in matrix.tolist()))
    postprocess = tl.constexpr(postprocess)
    prob = tl.constexpr(prob)
    num_iters = tl.constexpr(num_iters)

    @triton.jit
    def decode(syndrome):
        return _decode_one(
            syndrome,
            matrix,
            postprocess=postprocess,
            prob=prob,
            num_iters=num_iters,
        )

    return decode

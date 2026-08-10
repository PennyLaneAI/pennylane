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
import triton
import triton.language as tl
from numpy.typing import ArrayLike

from .algorithms import _decode_one
from .persistent_kernel import _persistent_decoder_kernel
from .triton_so_builder import build_so


def build_triton_decoder(
    decoder_fns: tuple[object, ...],
    *,
    platform: str = "hip:gfx90a:64",
    num_warps: int = 1,
    num_stages: int = 1,
    compiler: str = "",
    cflags: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """Build a shared library for a Triton decoder dispatcher.

    Example:

        >>> import triton
        >>> import triton.language as tl
        >>> from pennylane.backline.decoders.triton.decoder_frontend import build_triton_decoder
        >>> # For the standard Steane parity-check matrix ordering, nonzero syndromes
        >>> # 1..7 already encode which qubit to flip, so the lookup is just a shift.
        >>> @triton.jit
        ... def steane_lookup(syndrome):
        ...     one = tl.cast(1, tl.uint64)
        ...     zero = tl.cast(0, tl.uint64)
        ...     return tl.where(syndrome != 0, one << (syndrome - 1), zero)
        >>> # For the Steane CSS code, the same lookup rule can be used for both decoder ids.
        >>> so_path, symbol_name = build_triton_decoder(
        ...     (steane_lookup, steane_lookup),
        ...     platform="hip:gfx90a:64",
        ... )

    Args:
        decoder_fns (tuple[object, ...]): Tuple of Triton decoder functions.
            ``decoder_id`` selects the tuple index at runtime.
        platform (str): Triton target string of the form ``"backend:arch:warp_size"``.
        num_warps (int): Triton kernel launch warp count.
        num_stages (int): Triton pipeline stage count.
        compiler (str): Optional compiler executable override.
        cflags (tuple[str, ...]): Extra compiler flags.

    Returns:
        tuple[Path, str]: Path to the compiled shared library in a temporary location and the
            Triton-generated exported entrypoint name. The caller owns the returned file.
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
        return build_so(
            _persistent_decoder_kernel,
            signature={
                "ring_u64_ptr": "*u64",
                "handoff_u64_ptr": "*u64",
                "stop_u32_ptr": "*u32",
                "total": "u64",
            },
            constexpr={"decoder_fns": tuple(decoder_fns)},
            grid=(1, 1, 1),
            target=platform.strip(),
            out=str(out.resolve()),
            num_warps=num_warps,
            num_stages=num_stages,
            compiler=compiler,
            cflags=tuple(cflags),
        )
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise


def build_css_bp_decoder(
    Hx: ArrayLike,
    Hz: ArrayLike,
    *,
    postprocess: str = "osd",
    niter: int = 10,
    prob: float = 0.1,
    platform: str = "hip:gfx90a:64",
    num_warps: int = 1,
    num_stages: int = 1,
    compiler: str = "",
    cflags: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """Build a shared library for a configured Triton decoder.

    Example:

        >>> from catalyst.python_interface.transforms.qecp.qec_code_lib import QecCode
        >>> steane_code = QecCode.get("Steane")
        >>> Hx = steane_code.x_tanner
        >>> Hz = steane_code.z_tanner
        >>> so_path, symbol_name = build_css_bp_decoder(
        ...     Hx,
        ...     Hz,
        ...     postprocess="osd",
        ...     niter=10,
        ...     platform="hip:gfx90a:64",
        ... )

    Note:
        Takes one syndrome as a ``u64`` and returns one correction mask as a ``u64``.
        In the returned mask, bit ``i`` targets qubit ``i`` and ``0`` means no correction.
        TODO: This is a constraint of the current Payload and Handoff sizes.

    Args:
        Hx (ArrayLike): X parity-check matrix.
        Hz (ArrayLike): Z parity-check matrix.
        postprocess (str): Postprocessing step applied after belief propagation. Use
            ``"hard"`` for hard-decision output or ``"osd"`` for ordered-statistics decoding.
        niter (int): Number of decoder iterations.
        prob (float): Uniform prior error probability across qubits.
        platform (str): Triton target string of the form ``"backend:arch:warp_size"``.
            For instance ``"hip:gfx90a:64"`` targets AMD MI200-class GPUs via the
            HIP backend, gfx90a architecture, and warp size 64, while
            ``"cuda:80:32"`` means CUDA backend, SM80 architecture, warp size 32.
        num_warps (int): Triton kernel launch warp count.
        num_stages (int): Triton pipeline stage count.
        compiler (str): Optional compiler executable override.
        cflags (tuple[str, ...]): Extra compiler flags.

    Returns:
        tuple[Path, str]: Path to the compiled shared library in a temporary location and the
            Triton-generated exported entrypoint name. The caller owns the returned file.
    """
    _validate_css_options(postprocess=postprocess, niter=niter, prob=prob)
    hx = _to_numpy(Hx)
    hz = _to_numpy(Hz)
    decoder_fns = (
        _make_css_decoder(hx, postprocess=postprocess, niter=niter, prob=prob),
        _make_css_decoder(hz, postprocess=postprocess, niter=niter, prob=prob),
    )
    return build_triton_decoder(
        decoder_fns,
        platform=platform,
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
            exceeds the current 64-check or 64-variable packing limit.
    """
    h = np.asarray(H)
    if h.ndim != 2:
        raise ValueError(f"H must be a 2D array, got shape {h.shape!r}")
    if h.shape[0] == 0 or h.shape[1] == 0:
        raise ValueError(f"H must be non-empty, got shape {h.shape!r}")

    if not np.all((h == 0) | (h == 1)):
        raise ValueError("H must contain only binary entries 0/1")
    if h.shape[0] > 64:
        raise ValueError(f"H has {h.shape[0]} checks, but Triton decoder supports at most 64")
    if h.shape[1] > 64:
        raise ValueError(f"H has {h.shape[1]} variables, but Triton decoder supports at most 64")
    return h


def _validate_build_options(
    *, decoder_fns: tuple[object, ...], num_warps: int, num_stages: int, platform: str
) -> None:
    """Validate generic Triton decoder build options.

    Args:
        decoder_fns (tuple[object, ...]): Triton decoder functions to dispatch.
        num_warps (int): Triton kernel launch warp count.
        num_stages (int): Triton pipeline stage count.
        platform (str): Triton target string of the form
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
            "platform must be a raw Triton target string like "
            f"'hip:gfx90a:64' or 'cuda:80:32', got {platform!r}"
        )


def _validate_css_options(*, postprocess: str, niter: int, prob: float) -> None:
    """Validate CSS decoder options.

    Args:
        postprocess (str): Postprocessing rule applied after belief propagation.
        niter (int): Number of belief-propagation iterations.
        prob (float): Uniform prior error probability.

    Raises:
        ValueError: If any option falls outside the supported range.
    """
    if postprocess not in {"hard", "osd"}:
        raise ValueError("postprocess must be 'hard' or 'osd'")
    if niter <= 0:
        raise ValueError("niter must be > 0")
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")


def _make_css_decoder(h: np.ndarray, *, postprocess: str, niter: int, prob: float) -> object:
    """Specialize one Triton decoder kernel for a fixed parity-check matrix.

    Args:
        h (np.ndarray): Binary parity-check matrix.
        postprocess (str): Postprocessing rule applied after belief propagation.
        niter (int): Number of belief-propagation iterations.
        prob (float): Uniform prior error probability.

    Returns:
        object: Triton JIT function that maps one packed syndrome to one packed
            correction mask.
    """
    h = tl.constexpr(tuple(tuple(int(v) for v in row) for row in h.tolist()))
    postprocess = tl.constexpr(postprocess)
    prob = tl.constexpr(prob)
    niter = tl.constexpr(niter)

    @triton.jit
    def decode(syndrome):
        return _decode_one(syndrome, h, postprocess=postprocess, prob=prob, NITER=niter)

    return decode

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

"""Frontend for building Triton CSS decoder shared libraries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .decoder_kernel import _persistent_css_decoder_kernel
from .triton_shared import build_shared


def build_css_decoder(
    Hx: ArrayLike,
    Hz: ArrayLike,
    *,
    bp_variant: str = "sum_product",
    postprocess: str = "osd",
    niter: int = 10,
    prob: float = 0.1,
    alpha: float = 0.75,
    platform: str = "hip:gfx90a:64",
    build_dir: str = "decoder_build_dir",
    library_name: str = "librdma_triton_decoder.so",
    num_warps: int = 1,
    num_stages: int = 1,
    compiler: str = "",
    cflags: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """Build a shared library for a configured Triton decoder.

    Example:

        >>> import numpy as np
        >>> from pennylane.backline.decoders.triton.decoder_frontend import build_css_decoder
        >>> # Steane [[7, 1, 3]] code X parity-check matrix.
        >>> SteaneHx = np.array([
        ...     [1, 0, 1, 0, 1, 0, 1],
        ...     [0, 1, 1, 0, 0, 1, 1],
        ...     [0, 0, 0, 1, 1, 1, 1],
        ... ])
        >>> Hz = Hx = SteaneHx
        >>> so_path, symbol_name = build_css_decoder(
        ...     Hx,
        ...     Hz,
        ...     bp_variant="sum_product",
        ...     postprocess="osd",
        ...     niter=10,
        ...     platform="hip:gfx90a:64",
        ...     build_dir="build",
        ...     library_name="librdma_triton_decoder.so",
        ... )

    Args:
        Hx (ArrayLike): X parity-check matrix.
        Hz (ArrayLike): Z parity-check matrix.
        bp_variant (str): Belief-propagation variant to run, either
            ``"sum_product"`` or ``"norm_min_sum"``.
        postprocess (str): Postprocessing step applied after belief propagation. Use
            ``"hard"`` for hard-decision output or ``"osd"`` for ordered-statistics decoding.
        niter (int): Number of decoder iterations.
        prob (float): Uniform prior error probability across qubits.
        alpha (float): Scaling factor for normalized min-sum decoding.
        platform (str): Triton target string of the form ``"backend:arch:warp_size"``.
            For instance ``"hip:gfx90a:64"`` targets AMD MI200-class GPUs via the
            HIP backend, gfx90a architecture, and warp size 64, while
            ``"cuda:80:32"`` means CUDA backend, SM80 architecture, warp size 32.
        build_dir (str): Directory used during compilation. The compiled shared library is written
            here and temporary wrapper files are cleaned up afterwards.
        library_name (str): Filename of the compiled shared library.
        num_warps (int): Triton kernel launch warp count.
        num_stages (int): Triton pipeline stage count.
        compiler (str): Optional compiler executable override.
        cflags (tuple[str, ...]): Extra compiler flags.

    Returns:
        tuple[Path, str]: Path to the compiled shared library and the exported entrypoint name.
    """
    out = f"{build_dir}/{library_name}"
    _validate_options(
        bp_variant=bp_variant,
        postprocess=postprocess,
        niter=niter,
        prob=prob,
        num_warps=num_warps,
        num_stages=num_stages,
        platform=platform,
    )
    hx = _normalize_h(Hx)
    hz = _normalize_h(Hz)
    return build_shared(
        _persistent_css_decoder_kernel,
        signature={
            "ring_u64_ptr": "*u64",
            "handoff_u64_ptr": "*u64",
            "stop_u32_ptr": "*u32",
            "total": "u64",
        },
        constexpr={
            "Hx": tuple(tuple(int(v) for v in row) for row in hx.tolist()),
            "Hz": tuple(tuple(int(v) for v in row) for row in hz.tolist()),
            "BP_VARIANT": bp_variant,
            "POSTPROCESS": postprocess,
            "PROB": prob,
            "NITER": niter,
            "ALPHA": alpha,
        },
        grid=(1, 1, 1),
        target=platform.strip(),
        build_dir=str(Path(build_dir).resolve()),
        out=str(Path(out).resolve()),
        num_warps=num_warps,
        num_stages=num_stages,
        compiler=compiler,
        cflags=tuple(cflags),
    )


build_decoder = build_css_decoder


def _normalize_h(H: ArrayLike) -> np.ndarray:
    """
    Ensure matrix is compliant: 2 dimensional of size, filled with 1s and 0s
    """
    h = np.asarray(H)
    if h.ndim != 2:
        raise ValueError(f"H must be a 2D array, got shape {h.shape!r}")
    if h.shape[0] == 0 or h.shape[1] == 0:
        raise ValueError(f"H must be non-empty, got shape {h.shape!r}")

    if not np.all((h == 0) | (h == 1)):
        raise ValueError("H must contain only binary entries 0/1")
    if h.shape[0] > 64:
        raise ValueError(
            f"H has {h.shape[0]} checks, but Triton decoder supports at most 64"
        )
    if h.shape[1] > 64:
        raise ValueError(
            f"H has {h.shape[1]} variables, but Triton decoder supports at most 64"
        )
    return h


def _validate_options(
    *,
    bp_variant: str,
    postprocess: str,
    niter: int,
    prob: float,
    num_warps: int,
    num_stages: int,
    platform: str,
) -> None:
    """Validate decoder build options."""
    if bp_variant not in {"sum_product", "norm_min_sum"}:
        raise ValueError("bp_variant must be 'sum_product' or 'norm_min_sum'")
    if postprocess not in {"hard", "osd"}:
        raise ValueError("postprocess must be 'hard' or 'osd'")
    if niter <= 0:
        raise ValueError("niter must be > 0")
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")
    if num_warps <= 0:
        raise ValueError("num_warps must be > 0")
    if num_stages <= 0:
        raise ValueError("num_stages must be > 0")
    if platform.strip().count(":") != 2:
        raise ValueError(
            f"platform must be a raw Triton target string like 'hip:gfx90a:64' or 'cuda:80:32', got {platform!r}"
        )

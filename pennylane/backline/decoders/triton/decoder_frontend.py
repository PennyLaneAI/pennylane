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

import shutil
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .persistent_kernel import _persistent_css_decoder_kernel
from .triton_so_builder import build_so


def build_css_decoder(
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
        ...     postprocess="osd",
        ...     niter=10,
        ...     platform="hip:gfx90a:64",
        ... )

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
    _validate_options(
        postprocess=postprocess,
        niter=niter,
        prob=prob,
        num_warps=num_warps,
        num_stages=num_stages,
        platform=platform,
    )
    hx = _to_numpy(Hx)
    hz = _to_numpy(Hz)
    tmpdir = Path(tempfile.mkdtemp(prefix="pl_triton_decoder_"))
    out = tmpdir / "librdma_triton_decoder.so"
    try:
        return build_so(
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
                "POSTPROCESS": postprocess,
                "PROB": prob,
                "NITER": niter,
            },
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


def _to_numpy(H: ArrayLike) -> np.ndarray:
    """Validate and convert a parity-check matrix to a NumPy array."""
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
    postprocess: str,
    niter: int,
    prob: float,
    num_warps: int,
    num_stages: int,
    platform: str,
) -> None:
    """Validate decoder build options."""
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

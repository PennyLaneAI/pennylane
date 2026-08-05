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

"""Coprocessor functions for backline placement."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoprocessorFunction:
    """A precompiled function run on a :class:`~.Coprocessor` to process messages received from the
    :class:`~.Controller`.

    This is a thin handle over a precompiled library symbol. It contains the information needed to
    locate and dispatch the function (its symbol name, and the library it lives in).
    The compiled artifact is produced separately (cross-compiled or built on
    the same host, e.g., via Triton) and loaded by the runtime.

    See the Attributes section to learn more about the available options.

    .. seealso:: :class:`~.Coprocessor`, :func:`~.css_decoder`
    """

    name: str
    """The name the function is known by; used to resolve the precompiled symbol."""

    lib_path: str | None = None
    """Optional path to the shared library that provides the symbol. When ``None``, the runtime
    resolves ``name`` from the symbols already loaded on the host."""

    @property
    def symbol_name(self) -> str:
        """The symbol the runtime resolves and invokes for this function."""
        return self.name


def css_decoder(
    Hx: np.ndarray,
    Hz: np.ndarray,
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
) -> CoprocessorFunction:
    """Compile a CSS code's Tanner graph into a coprocessor decode function.

    Accepts the X- and Z-type parity-check matrices of a CSS code and compiles a decoder down to a
    shared library that can be used as a :class:`~.CoprocessorFunction`.

    Args:
        Hx (np.ndarray): The X parity-check matrix.
        Hz (np.ndarray): The Z parity-check matrix.

    Returns:
        CoprocessorFunction: The compiled decode function, ready to pass as a
        :class:`~.Coprocessor`'s ``coprocessor_fn``.

    .. seealso:: :class:`~.CoprocessorFunction`, :class:`~.Coprocessor`
    """
    try:
        from .decoders.triton.decoder_frontend import build_css_decoder
    except ImportError as exc:
        raise ImportError("css_decoder requires Triton support.") from exc

    so_path, symbol_name = build_css_decoder(
        Hx,
        Hz,
        bp_variant=bp_variant,
        postprocess=postprocess,
        niter=niter,
        prob=prob,
        alpha=alpha,
        platform=platform,
        build_dir=build_dir,
        library_name=library_name,
        num_warps=num_warps,
        num_stages=num_stages,
        compiler=compiler,
        cflags=cflags,
    )
    return CoprocessorFunction(name=symbol_name, lib_path=str(so_path))

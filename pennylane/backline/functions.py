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
from importlib import import_module

from numpy.typing import ArrayLike


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
    Hx: ArrayLike,
    Hz: ArrayLike,
    *,
    postprocess: str = "osd",
    num_iters: int = 10,
    prob: float = 0.1,
    **build_options,
) -> CoprocessorFunction:
    """Compile a CSS code's Tanner graph into a coprocessor decode function.

    Accepts the X- and Z-type parity-check matrices of a CSS code and compiles a decoder down to a
    shared library that can be used as a :class:`~.CoprocessorFunction`.

    Args:
        Hx (ArrayLike): X parity-check matrix.
        Hz (ArrayLike): Z parity-check matrix.
        postprocess (str): Postprocessing step applied after belief propagation. Use
            ``"hard"`` for hard-decision output or ``"osd"`` for ordered-statistics decoding.
        num_iters (int): Number of belief-propagation iterations.
        prob (float): Uniform prior error probability across qubits.

    Keyword Args:
        platform (str): Triton target string of the form ``"backend:arch:warp_size"``.
            Defaults to ``"hip:gfx90a:64"``.
        num_warps (int): Triton kernel launch warp count. Defaults to ``1``.
        num_stages (int): Triton pipeline stage count. Defaults to ``1``.
        compiler (str): Optional compiler executable override. Defaults to ``""``.
        cflags (tuple[str, ...]): Extra compiler flags. Defaults to ``()``.

    Returns:
        CoprocessorFunction: The compiled decode function, ready to pass as a
            :class:`~.Coprocessor`'s ``coprocessor_fn``.

    Raises:
        ImportError: If Triton decoder support is unavailable.
        ValueError: If the decoder options or parity-check matrices are invalid.

    .. seealso:: :class:`~.CoprocessorFunction`, :class:`~.Coprocessor`

    **Example**

    >>> Hx = ((1, 0, 1), (0, 1, 1))
    >>> Hz = ((1, 1, 0), (1, 0, 1))
    >>> decoder = qp.backline.css_decoder(Hx, Hz, postprocess="hard", num_iters=5)
    """
    try:
        build_css_bp_decoder = import_module(
            "pennylane.backline.decoders.triton.decoder_frontend"
        ).build_css_bp_decoder
    except ImportError as exc:
        raise ImportError("css_decoder requires Triton support.") from exc

    so_path, symbol_name = build_css_bp_decoder(
        Hx, Hz, postprocess=postprocess, num_iters=num_iters, prob=prob, **build_options
    )
    return CoprocessorFunction(name=symbol_name, lib_path=str(so_path))

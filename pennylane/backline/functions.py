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

from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CoprocessorFunction:
    """A precompiled function run on a :class:`~.Coprocessor` to process messages received from the
    :class:`~.Controller`.

    This is a thin handle over a precompiled library symbol. It contains the information needed to
    locate and dispatch the function - its symbol name, and the library it lives in. The compiled
    artifact is produced separately (cross-compiled, or built on the same host, e.g., via Triton) and
    loaded by the runtime.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    Args:
        name (str): The name the function is known by, used to resolve the precompiled symbol.
        lib_path (str, None): Path to the shared library providing the symbol. Defaults to
            ``None``, in which case the runtime resolves :attr:`name` from the symbols already
            loaded on the host.

    .. seealso:: :class:`~.Coprocessor`, :func:`~.css_bp_decoder`, :func:`~.triton_decoder`

    **Example**

    A coprocessor function is usually named rather than constructed --- passing a string to
    :class:`~.Coprocessor` resolves it:

    >>> coproc = qp.Coprocessor(coprocessor_fn="decoder")
    >>> coproc.coprocessor_fn
    CoprocessorFunction(name='decoder', lib_path=None)

    Construct one directly to point at a symbol in a specific shared library. The path is what the
    coprocessor's node passes to the runtime as its backend library:

    >>> fn = qp.CoprocessorFunction(
    ...     name="decode_syndrome", lib_path="/opt/backline/libdecoder.so"
    ... )
    >>> fn.symbol_name
    'decode_syndrome'
    """

    name: str
    """The name the function is known by; used to resolve the precompiled symbol."""

    lib_path: str | None = None
    """Path to the shared library that provides the symbol. Defaults to ``None``, in which case the
    runtime resolves :attr:`name` from the symbols already loaded on the host."""

    @property
    def symbol_name(self) -> str:
        """The symbol the runtime resolves and invokes for this function."""
        return self.name


def triton_decoder(
    decoder_fns: tuple[object, ...],
    **build_options,
) -> CoprocessorFunction:
    """Compile Triton quantum error correction decoder functions into a coprocessor function
    for use with :mod:`~.backline`.

    This function accepts a tuple of un-jitted Triton decoder functions, and compiles them into a
    shared library that can be used as a :class:`~.CoprocessorFunction`.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

    Args:
        decoder_fns (tuple[object, ...]): Un-jitted Triton decoder functions. Each entry is
            jit compiled internally, and ``decoder_id`` selects the tuple index at runtime.

    Keyword Args:
        platform (str): Required Triton platform string of the form ``"backend:arch:warp_size"``.
            For example, ``"hip:gfx942:64"`` or ``"cuda:80:32"``.
        grid (tuple[int, int, int]): Triton kernel launch grid dimensions.
            Defaults to ``(1, 1, 1)``.
        num_warps (int): Triton kernel launch warp count. Defaults to ``1``.
        num_stages (int): Triton pipeline stage count. Defaults to ``1``.
        compiler (str): Optional compiler executable override. Defaults to ``""``.
        cflags (tuple[str, ...]): Extra compiler flags. Defaults to ``()``.

    Returns:
        CoprocessorFunction: The compiled decode function, ready to pass as
            :attr:`~.Coprocessor.coprocessor_fn`.

    Raises:
        ImportError: If Triton decoder support is unavailable.
        TypeError: If ``decoder_fns`` contains already jit compiled Triton functions.
        ValueError: If the decoder build options are invalid.

    .. seealso:: :class:`~.CoprocessorFunction`, :class:`~.Coprocessor`,
        :func:`~.css_bp_decoder`

    **Example**

    >>> import pennylane as qp
    >>> import triton.language as tl
    >>> def steane_lookup(syndrome):
    ...     return tl.where(syndrome != 0, 1 << (syndrome - 1), 0)
    >>> decoder = qp.backline.triton_decoder(  # doctest: +SKIP
    ...     (steane_lookup, steane_lookup),
    ...     platform="hip:gfx942:64",
    ... )
    """
    try:
        from pennylane.backline.decoders.triton.decoder_frontend import (  # pylint: disable=import-outside-toplevel
            _build_triton_decoder,
        )
    except ImportError as exc:
        raise ImportError("Triton decoders require installed `triton` Python package.") from exc

    so_path, symbol_name = _build_triton_decoder(decoder_fns, **build_options)  # pragma: no cover
    return CoprocessorFunction(name=symbol_name, lib_path=str(so_path))  # pragma: no cover


def css_bp_decoder(
    Hx: ArrayLike,
    Hz: ArrayLike,
    *,
    postprocess: str = "osd",
    num_iters: int = 10,
    prob: float = 0.1,
    **build_options,
) -> CoprocessorFunction:
    """Compile a CSS code's Tanner graph into a coprocessor decode function for use with
    :mod:`~.backline`.

    Accepts the X- and Z-type parity-check matrices of a CSS code and compiles a decoder down to a
    shared library that can be used as a :class:`~.CoprocessorFunction`.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

    Args:
        Hx (ArrayLike): X parity-check matrix.
        Hz (ArrayLike): Z parity-check matrix.

    Keyword Args:
        postprocess (str): Postprocessing step applied after belief propagation. Use
            ``"hard"`` for hard-decision output or ``"osd"`` for ordered-statistics decoding.
        num_iters (int): Number of belief-propagation iterations.
        prob (float): Uniform prior error probability across qubits.
        platform (str): Required Triton platform string of the form ``"backend:arch:warp_size"``.
            For example, ``"hip:gfx942:64"`` or ``"cuda:80:32"``.
        grid (tuple[int, int, int]): Triton kernel launch grid dimensions.
            Defaults to ``(1, 1, 1)``.
        num_warps (int): Triton kernel launch warp count. Defaults to ``1``.
        num_stages (int): Triton pipeline stage count. Defaults to ``1``.
        compiler (str): Optional compiler executable override. Defaults to ``""``.
        cflags (tuple[str, ...]): Extra compiler flags. Defaults to ``()``.

    Returns:
        CoprocessorFunction: The compiled decode function, ready to pass as
            :attr:`~.Coprocessor.coprocessor_fn`.

    Raises:
        ImportError: If Triton decoder support is unavailable.
        ValueError: If the decoder options or parity-check matrices are invalid.

    .. seealso:: :class:`~.CoprocessorFunction`, :class:`~.Coprocessor`,
        :func:`~.triton_decoder`

    **Example**

    >>> import numpy as np
    >>> import pennylane as qp
    >>> Hz = Hx = np.array([
    ...     [1, 0, 1, 0, 1, 0, 1],
    ...     [0, 1, 1, 0, 0, 1, 1],
    ...     [0, 0, 0, 1, 1, 1, 1],
    ... ])
    >>> decoder = qp.backline.css_bp_decoder(  # doctest: +SKIP
    ...     Hx,
    ...     Hz,
    ...     postprocess="hard",
    ...     num_iters=5,
    ...     platform="hip:gfx942:64",
    ... )
    """
    try:
        from pennylane.backline.decoders.triton.decoder_frontend import (  # pylint: disable=import-outside-toplevel
            _build_css_bp_decoder,
        )
    except ImportError as exc:
        raise ImportError("Triton decoders require installed `triton` Python package.") from exc

    so_path, symbol_name = _build_css_bp_decoder(  # pragma: no cover
        Hx, Hz, postprocess=postprocess, num_iters=num_iters, prob=prob, **build_options
    )
    return CoprocessorFunction(name=symbol_name, lib_path=str(so_path))  # pragma: no cover

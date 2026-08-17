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
    locate and dispatch the function - its symbol name, and the library it lives in. The compiled
    artifact is produced separately (cross-compiled, or built on the same host, e.g. via Triton) and
    loaded by the runtime.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    Args:
        name (str): The name the function is known by, used to resolve the precompiled symbol.
        lib_path (str | None): Path to the shared library providing the symbol. Defaults to
            ``None``, in which case the runtime resolves ``name`` from the symbols already loaded
            on the host.

    .. seealso:: :class:`~.Coprocessor`, :func:`~.css_decoder`

    **Example**

    A coprocessor function is usually named rather than constructed --- passing a string to
    :class:`~.Coprocessor` resolves it:

    >>> coproc = qp.Coprocessor(coprocessor_fn="decoder", comm_host="127.0.0.1")
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


def css_decoder(Hx: np.ndarray, Hz: np.ndarray) -> CoprocessorFunction:
    """Compile a CSS code's Tanner graph into a coprocessor decode function.

    Accepts the X- and Z-type parity-check matrices of a CSS code and compiles a decoder down to a
    shared library that can be used as a :class:`~.CoprocessorFunction`.

    .. note::
        Not yet implemented — this is a placeholder for the Triton-based decoder compiler.

    Args:
        Hx (np.ndarray): The X parity-check matrix.
        Hz (np.ndarray): The Z parity-check matrix.

    Returns:
        CoprocessorFunction: The compiled decode function, ready to pass as a
        :class:`~.Coprocessor`'s ``coprocessor_fn``.

    .. seealso:: :class:`~.CoprocessorFunction`, :class:`~.Coprocessor`
    """
    raise NotImplementedError(
        "css_decoder is not yet implemented; it will compile a CSS code's Tanner graph "
        "into a CoprocessorFunction via Triton."
    )

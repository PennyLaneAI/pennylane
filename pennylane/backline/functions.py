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
    """A precompiled function to run on a Coprocessor to process messages received from Controller.

    This is a thin handle over a precompiled library symbol. It contains the information needed to
    locate and dispatch the function (its symbol name, and the library it lives in).
    The compiled artifact is produced separately (cross-compiled or built on
    the same host, e.g., via Triton) and loaded by the runtime.

    See the Attributes section to learn more about the available options.
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


def css_decoder(Hx: np.ndarray, Hz: np.ndarray) -> CoprocessorFunction:
    """Compile a CSS code's Tanner graph into a coprocessor decode function.

    Accepts the X- and Z-type parity-check matrices of a CSS code and compiles a decoder down to a
    shared library that can be used as a :class:`CoprocessorFunction`.

    .. note::
        Not yet implemented — this is a placeholder for the Triton-based decoder compiler.

    Args:
        Hx (np.ndarray): The X parity-check matrix.
        Hz (np.ndarray): The Z parity-check matrix.

    Returns:
        CoprocessorFunction: The compiled decode function.
    """
    raise NotImplementedError(
        "css_decoder is not yet implemented; it will compile a CSS code's Tanner graph "
        "into a CoprocessorFunction via Triton."
    )

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

"""
This module provides the functionality to call a runtime entry point directly, by its C symbol
name.

A symbol is declared with :func:`~.runtime_declare` and called with :func:`~.runtime_call` from
inside a compiled program. The call can be dispatched to an executor, which invokes the symbol on
the machine the runtime lives on.

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~runtime_call
    ~runtime_declare

Types
-----

.. currentmodule:: pennylane.runtime

.. autosummary::
    :toctree: api

    ~CSignature
    ~CType

**Example**

Declare a symbol once, then call it:

.. code-block:: python

    import pennylane as qp

    qp.runtime_declare("example_run_rounds", "(ptr, u32) -> u64")

    def program(session):
        return qp.runtime_call("example_run_rounds", session, 100000, address="board:9000")
"""

from .runtime_call import runtime_call
from .signature import CSignature, CType, declare, declared_symbols, signature_of

__all__ = (
    "CSignature",
    "CType",
    "declare",
    "declared_symbols",
    "runtime_call",
    "signature_of",
)

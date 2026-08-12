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

A symbol is declared with :func:`~pennylane.runtime_declare` and called with
:func:`~pennylane.runtime_call` from inside a compiled program. The call can be dispatched to an
executor, which invokes the symbol on the machine the runtime lives on.

:class:`CSignature` holds a symbol's C signature, over the types in :class:`CType`.
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

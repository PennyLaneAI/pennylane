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
"""Conversions between operators and other representations."""

from importlib import import_module
from typing import TYPE_CHECKING

from .pauli_to_operator import to_operator

if TYPE_CHECKING:
    from .operator_to_pauli import is_pauli_sentence, is_pauli_word, pauli_sentence
    from .pauli_strings import pauli_word_to_string, string_to_pauli_word

_LAZY_IMPORTS = {
    "is_pauli_sentence": ".operator_to_pauli",
    "is_pauli_word": ".operator_to_pauli",
    "pauli_sentence": ".operator_to_pauli",
    "pauli_word_to_string": ".pauli_strings",
    "string_to_pauli_word": ".pauli_strings",
}

__all__ = ["to_operator", *_LAZY_IMPORTS]


def __getattr__(name):
    """Defer conversions that require operator classes until they are requested."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_IMPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    """Include deferred conversions in module attribute discovery."""
    return sorted(set(globals()) | set(__all__))

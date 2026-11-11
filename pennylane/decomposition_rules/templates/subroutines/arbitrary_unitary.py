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

"""Built-in decomposition rules for templates.subroutines.arbitrary_unitary."""

from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import PauliRot
from pennylane.templates.subroutines.arbitrary_unitary import (
    ArbitraryUnitary,
    _all_pauli_words_but_identity,
)
from pennylane.typing import Float, Wire
from pennylane.wires import WiresLike


def _arbitrary_unitary_resources(num_wires: int) -> dict:
    resources = {}
    for pauli_word in _all_pauli_words_but_identity(num_wires):
        resources[PauliRot(Float, pauli_word=pauli_word, wires=Wire[len(pauli_word)])] = 1
    return resources


@register_resources(_arbitrary_unitary_resources)
def _arbitrary_unitary_decomposition(weights: list, wires: WiresLike, **_):
    words = list(_all_pauli_words_but_identity(len(wires)))

    for i, word in enumerate(words):
        PauliRot(weights[..., i], word, wires=wires)


add_decomps(ArbitraryUnitary, _arbitrary_unitary_decomposition)

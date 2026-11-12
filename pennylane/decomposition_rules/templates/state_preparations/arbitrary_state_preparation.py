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

"""Built-in decomposition rules for templates.state_preparations.arbitrary_state_preparation."""

from collections import Counter

import pennylane as qp
from pennylane import register_resources
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps
from pennylane.templates.state_preparations.arbitrary_state_preparation import (
    ArbitraryStatePreparation,
    _state_preparation_pauli_words,
)
from pennylane.typing import Float, Wire


def _arbitrary_state_preparation_resources(num_wires):
    return dict(
        Counter(
            qp.PauliRot(Float, pauli_word=pauli_word, wires=Wire[len(pauli_word)])
            for pauli_word in _state_preparation_pauli_words(num_wires)
        )
    )


@register_resources(_arbitrary_state_preparation_resources)
def _arbitrary_state_preparation_decomposition(weights, wires, **_):
    pauli_words = _state_preparation_pauli_words(len(wires))

    @for_loop(len(pauli_words))
    def pauli_loop(i):
        pauli_word = pauli_words[i]
        qp.PauliRot(weights[i], pauli_word, wires=wires)

    pauli_loop()  # pylint: disable=no-value-for-parameter


add_decomps(ArbitraryStatePreparation, _arbitrary_state_preparation_decomposition)

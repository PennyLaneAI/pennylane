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

"""Built-in decomposition rules for templates.subroutines.iqp."""

from collections import defaultdict

from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import PPR_4, Hadamard, MultiRZ
from pennylane.templates.subroutines.iqp import IQP
from pennylane.typing import Float, Wire


def _instantaneous_quantum_polynomial_resources(
    weights, wires, pattern, spin_sym
):  # pylint: disable=unused-argument
    num_wires = len(wires)
    resources = defaultdict(int)
    if spin_sym:
        resources[PPR_4(1, pauli_word="Y" + "X" * (num_wires - 1), wires=Wire[num_wires])] = 1

    resources[Hadamard] = 2 * num_wires

    for gate in pattern:
        for gen in gate:
            resources[MultiRZ(Float, Wire[len(gen)])] += 1

    return resources


@register_resources(_instantaneous_quantum_polynomial_resources)
def _instantaneous_quantum_polynomial_decomposition(weights, wires, pattern, spin_sym):
    num_wires = len(wires)

    if spin_sym:
        PPR_4(1, "Y" + "X" * (num_wires - 1), wires=wires)

    for i in range(num_wires):
        Hadamard(wires[i])

    for par, gate in zip(weights, pattern, strict=True):
        for gen in gate:
            MultiRZ(2 * par, wires=[wires[g] for g in gen])

    for i in range(num_wires):
        Hadamard(wires[i])


add_decomps(IQP, _instantaneous_quantum_polynomial_decomposition)

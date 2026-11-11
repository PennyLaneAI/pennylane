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

"""Built-in decomposition rules for templates.subroutines.time_evolution.approx_time_evolution."""

# pylint: disable=protected-access

from collections import defaultdict

from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import PauliRot
from pennylane.ops.op_math.linear_combination import Hamiltonian
from pennylane.pauli import PauliWord
from pennylane.templates.subroutines.time_evolution.approx_time_evolution import ApproxTimeEvolution
from pennylane.typing import Float, Wire
from pennylane.wires import Wires, WiresLike


def _approx_time_evolution_resources(words: tuple[PauliWord], n: int):
    resources = defaultdict(int)

    for _ in range(n):
        for pw in words:
            if len(pw) != 0:
                term_str = "".join(pw.values())
                resources[PauliRot(Float, pauli_word=term_str, wires=Wire[len(term_str)])] += 1

    return resources


@register_resources(_approx_time_evolution_resources)
def _approx_time_evolution_decomposition(
    *coeffs_and_time: list, wires: WiresLike, hamiltonian: Hamiltonian, n: int
):  # pylint: disable=unused-argument
    time = coeffs_and_time[-1]

    @for_loop(n)
    def rounds_loop(_):

        for pauli_key in list(hamiltonian.pauli_rep.keys()):
            for pk, coeff in hamiltonian.pauli_rep.items():
                if pauli_key == pk:
                    break

            if len(pauli_key) != 0:
                theta = 2 * time * coeff / n  # pylint: disable=undefined-loop-variable
                term_str = "".join(pauli_key.values())
                wires = Wires(pauli_key.keys())
                PauliRot(theta, term_str, wires=wires)

    rounds_loop()  # pylint: disable=no-value-for-parameter


add_decomps(ApproxTimeEvolution, _approx_time_evolution_decomposition)

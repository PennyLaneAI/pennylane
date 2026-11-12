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

"""Built-in decomposition rules for templates.subroutines.qchem.fermionic_single_excitation."""

import numpy as np
from jax import numpy as jnp

from pennylane import capture
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, RX, RZ, Hadamard
from pennylane.templates.subroutines.qchem.fermionic_single_excitation import (
    FermionicSingleExcitation,
)


def _fermionic_single_excitation_resources(num_wires):
    return {RX: 4, Hadamard: 4, CNOT: 4 * (num_wires - 1), RZ: 2}


@register_resources(_fermionic_single_excitation_resources)
def _fermionic_single_excitation_decomposition(weight, wires, **_):
    # Interpret first and last wire as r and p
    r = wires[0]
    p = wires[-1]

    # Sequence of the wires entering the CNOTs between wires 'r' and 'p'
    set_cnot_wires = [list(wires.labels[l : l + 2]) for l in range(len(wires) - 1)]

    if capture.enabled():
        set_cnot_wires = jnp.array(set_cnot_wires)

    # ------------------------------------------------------------------
    # Apply the first layer

    # U_1, U_2 acting on wires 'r' and 'p'
    RX(-np.pi / 2, wires=r)
    Hadamard(wires=p)

    # Applying CNOTs
    @for_loop(len(set_cnot_wires))
    def apply_cnots(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots()  # pylint: disable=no-value-for-parameter

    # Z rotation acting on wire 'p'
    RZ(weight / 2, wires=p)

    # Applying CNOTs in reverse order
    @for_loop(len(set_cnot_wires) - 1, -1, -1)
    def apply_cnots_reversed(i):
        CNOT(wires=set_cnot_wires[i])

    apply_cnots_reversed()  # pylint: disable=no-value-for-parameter

    # U_1^+, U_2^+ acting on wires 'r' and 'p'
    RX(np.pi / 2, wires=r)
    Hadamard(wires=p)

    # ------------------------------------------------------------------
    # Apply the second layer

    # U_1, U_2 acting on wires 'r' and 'p'
    Hadamard(wires=r)
    RX(-np.pi / 2, wires=p)

    apply_cnots()  # pylint: disable=no-value-for-parameter

    # Z rotation acting on wire 'p'
    RZ(-weight / 2, wires=p)

    apply_cnots_reversed()  # pylint: disable=no-value-for-parameter

    # U_1^+, U_2^+ acting on wires 'r' and 'p'
    Hadamard(wires=r)
    RX(np.pi / 2, wires=p)


add_decomps(FermionicSingleExcitation, _fermionic_single_excitation_decomposition)

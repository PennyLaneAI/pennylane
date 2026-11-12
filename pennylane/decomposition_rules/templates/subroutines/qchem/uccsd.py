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

"""Built-in decomposition rules for templates.subroutines.qchem.uccsd."""

# pylint: disable-msg=too-many-arguments,protected-access,too-many-positional-arguments

from collections import Counter

from jax import numpy as jnp

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.ops import BasisState
from pennylane.templates.subroutines.qchem.fermionic_double_excitation import (
    FermionicDoubleExcitation,
)
from pennylane.templates.subroutines.qchem.fermionic_single_excitation import (
    FermionicSingleExcitation,
)
from pennylane.templates.subroutines.qchem.uccsd import UCCSD
from pennylane.typing import Bool, Wire


def _UCCSD_resources(num_wires, n_repeats, num_d_wires, num_s_wires):
    resources = Counter({BasisState(Bool[num_wires], Wire[num_wires]): 1})

    for _ in range(n_repeats):
        for w1, w2 in num_d_wires:
            resources[resource_rep(FermionicDoubleExcitation, num_wires_1=w1, num_wires_2=w2)] += 1

        for s in num_s_wires:
            resources[resource_rep(FermionicSingleExcitation, num_wires=s)] += 1

    return dict(resources)


@register_resources(_UCCSD_resources)
def _UCCSD_decomposition(weights, wires, s_wires, d_wires, init_state, n_repeats):
    BasisState(init_state, wires=wires)

    if n_repeats == 1 and len(math.shape(weights)) == 1:
        weights = math.expand_dims(weights, 0)

    if capture.enabled():
        weights, d_wires, s_wires = jnp.array(weights), jnp.array(d_wires), jnp.array(s_wires)

    @for_loop(n_repeats)
    def apply_layers(layer):
        @for_loop(len(d_wires))
        def double_excitation(i):
            w1, w2 = d_wires[i]
            FermionicDoubleExcitation(weights[layer][len(s_wires) + i], wires1=w1, wires2=w2)

        @for_loop(len(s_wires))
        def single_excitation(j):
            s_wires_ = s_wires[j]
            FermionicSingleExcitation(weights[layer][j], wires=s_wires_)

        double_excitation()  # pylint: disable=no-value-for-parameter
        single_excitation()  # pylint: disable=no-value-for-parameter

    apply_layers()  # pylint: disable=no-value-for-parameter


add_decomps(UCCSD, _UCCSD_decomposition)

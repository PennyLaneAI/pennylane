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

"""Built-in decomposition rules for templates.subroutines.qchem.all_singles_doubles."""

# pylint: disable=too-many-arguments,protected-access

from jax import numpy as jnp

from pennylane import capture
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import BasisState, DoubleExcitation, SingleExcitation
from pennylane.templates.subroutines.qchem.all_singles_doubles import AllSinglesDoubles
from pennylane.typing import Bool, Wire


def _all_singles_doubles_resouces(num_singles, num_doubles, num_wires):
    return {
        BasisState(Bool[num_wires], Wire[num_wires]): 1,
        DoubleExcitation: num_doubles,
        SingleExcitation: num_singles,
    }


@register_resources(_all_singles_doubles_resouces)
def _all_singles_doubles_decomposition(weights, wires, hf_state, singles, doubles):
    BasisState(hf_state, wires=wires)

    if capture.enabled():
        weights, doubles, singles = jnp.array(weights), jnp.array(doubles), jnp.array(singles)

    @for_loop(len(doubles))
    def doubles_loop(i):
        DoubleExcitation(weights[len(singles) + i], wires=doubles[i])

    doubles_loop()  # pylint: disable=no-value-for-parameter

    @for_loop(len(singles))
    def singles_loop(j):
        SingleExcitation(weights[j], wires=singles[j])

    singles_loop()  # pylint: disable=no-value-for-parameter


add_decomps(AllSinglesDoubles, _all_singles_doubles_decomposition)

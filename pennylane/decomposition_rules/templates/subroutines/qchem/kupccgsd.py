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

"""Built-in decomposition rules for templates.subroutines.qchem.kupccgsd."""

# pylint: disable-msg=too-many-arguments,protected-access,too-many-positional-arguments

from collections import defaultdict

from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.ops.qubit import BasisState
from pennylane.templates.subroutines.qchem.fermionic_double_excitation import (
    FermionicDoubleExcitation,
)
from pennylane.templates.subroutines.qchem.fermionic_single_excitation import (
    FermionicSingleExcitation,
)
from pennylane.templates.subroutines.qchem.kupccgsd import kUpCCGSD
from pennylane.typing import Bool, Wire
from pennylane.wires import WiresLike


def _kupccgsd_resources(num_wires: int, k: int, d_wires: list, s_wires: list):
    resources = defaultdict(int)
    resources[BasisState(Bool[num_wires], Wire[num_wires])] = 1

    for _ in range(k):
        for w1, w2 in d_wires:
            resources[
                resource_rep(FermionicDoubleExcitation, num_wires_1=len(w1), num_wires_2=len(w2))
            ] += 1

        for s_wires_ in s_wires:
            resources[resource_rep(FermionicSingleExcitation, num_wires=len(s_wires_))] += 1

    return resources


@register_resources(_kupccgsd_resources)
def _kupccgsd_decomposition(
    weights: list,
    wires: WiresLike,
    s_wires: list,
    d_wires: list,
    k: int,
    init_state: tuple[int],
    delta_sz: int = None,
):  # pylint: disable=too-many-arguments, arguments-differ, unused-argument
    BasisState(init_state, wires=wires)

    @for_loop(k)
    def layer_loop(layer):
        @for_loop(len(d_wires))
        def double_loop(i):
            w1, w2 = d_wires[i]
            FermionicDoubleExcitation(weights[layer][len(s_wires) + i], wires1=w1, wires2=w2)

        double_loop()  # pylint: disable=no-value-for-parameter

        @for_loop(len(s_wires))
        def single_loop(j):
            s_wires_ = s_wires[j]
            FermionicSingleExcitation(weights[layer][j], wires=s_wires_)

        single_loop()  # pylint: disable=no-value-for-parameter

    layer_loop()  # pylint: disable=no-value-for-parameter


add_decomps(kUpCCGSD, _kupccgsd_decomposition)

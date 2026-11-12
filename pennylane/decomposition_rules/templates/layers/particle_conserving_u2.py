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

"""Built-in decomposition rules for templates.layers.particle_conserving_u2."""

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, CRX, RZ, BasisState
from pennylane.templates.layers.particle_conserving_u2 import ParticleConservingU2
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires, WiresLike


def _particle_conserving_u2_resources(num_wires: int, n_layers: int):
    num_nm_wires = num_wires - 1  # number of pairs of even-indexed of wires
    return {
        BasisState(Bool[num_wires], Wire[num_wires]): 1,
        RZ: n_layers * num_wires,
        CNOT: 2 * num_nm_wires * n_layers,
        CRX: num_nm_wires * n_layers,
    }


@register_resources(_particle_conserving_u2_resources)
def _particle_conserving_u2_decomposition(weights: list, wires: WiresLike, init_state: tuple[int]):
    nm_wires = [wires[l : l + 2] for l in range(0, len(wires) - 1, 2)]
    nm_wires += [wires[l : l + 2] for l in range(1, len(wires) - 1, 2)]
    n_layers = math.shape(weights)[0]

    if isinstance(wires, Wires):
        wires = wires.labels
        # If we are using capture, then it is going to be problematic to use Wires.
        # Instead, we need to use the wire labels. Here we have a collection of Wires,
        # and this line goes through each of them and turns them into their labels.
        nm_wires = list(map(lambda w: w.labels if isinstance(w, Wires) else w, nm_wires))

    if capture.enabled():
        nm_wires, weights, wires, init_state = (
            math.array(nm_wires, like="jax"),
            math.array(weights, like="jax"),
            math.array(wires, like="jax"),
            math.array(init_state, like="jax"),
        )

    BasisState(init_state, wires=wires)

    @for_loop(n_layers)
    def layers_loop(l):
        @for_loop(len(wires))
        def rz_loop(j):
            wires_ = wires[j]
            RZ(weights[l, j], wires=wires_)

        rz_loop()  # pylint: disable=no-value-for-parameter

        @for_loop(len(nm_wires))
        def nm_loop(i):
            wires_ = nm_wires[i]
            CNOT(wires=wires_)
            CRX(2 * weights[l, len(wires) + i], wires=wires_[::-1])
            CNOT(wires=wires_)

        nm_loop()  # pylint: disable=no-value-for-parameter

    layers_loop()  # pylint: disable=no-value-for-parameter


add_decomps(ParticleConservingU2, _particle_conserving_u2_decomposition)

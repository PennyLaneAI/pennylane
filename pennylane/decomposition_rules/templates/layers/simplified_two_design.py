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

"""Built-in decomposition rules for templates.layers.simplified_two_design."""

from jax import numpy as jnp

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CZ, RY
from pennylane.templates.layers.simplified_two_design import SimplifiedTwoDesign


def _simplified_two_design_resources(n_layers, num_wires):
    if num_wires > 1:
        return {
            RY: num_wires + (n_layers * num_wires - n_layers) * 2,
            CZ: n_layers * num_wires - n_layers,
        }
    return {RY: num_wires}


@register_resources(_simplified_two_design_resources)
def _simplified_two_design_decomposition(initial_layer_weights, weights, wires):
    n_layers = math.shape(weights)[0]

    if capture.enabled():
        initial_layer_weights, weights, wires = (
            jnp.array(initial_layer_weights),
            jnp.array(weights),
            jnp.array(wires),
        )

    # initial rotations
    @for_loop(len(wires))
    def initial_rotation_loop(i):
        RY(initial_layer_weights[i], wires=wires[i])

    initial_rotation_loop()  # pylint: disable=no-value-for-parameter

    @for_loop(n_layers)
    def layers_loop(layer):

        all_wire_pairs = [wires[i : i + 2] for i in range(0, len(wires) - 1, 2)] + [
            wires[i : i + 2] for i in range(1, len(wires) - 1, 2)
        ]
        if capture.enabled():
            all_wire_pairs = jnp.array(all_wire_pairs)

        @for_loop(len(all_wire_pairs))
        def block_loop(i):
            wire_pair = all_wire_pairs[i]
            CZ(wires=wire_pair)
            RY(weights[layer, i, 0], wires=wire_pair[0])
            RY(weights[layer, i, 1], wires=wire_pair[1])

        block_loop()  # pylint: disable=no-value-for-parameter

    layers_loop()  # pylint: disable=no-value-for-parameter


add_decomps(SimplifiedTwoDesign, _simplified_two_design_decomposition)

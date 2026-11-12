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

"""Built-in decomposition rules for templates.layers.gate_fabric."""

# pylint: disable=too-many-arguments

import numpy as np
from jax import numpy as jnp

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import BasisState, DoubleExcitation, OrbitalRotation, cond
from pennylane.templates.layers.gate_fabric import GateFabric
from pennylane.typing import Bool, Wire
from pennylane.wires import Wires


def _gate_fabric_resources(n_layers, num_wires, len_wire_pattern, include_pi):
    rotation_count = 2 * n_layers * len_wire_pattern if include_pi else n_layers * len_wire_pattern
    return {
        BasisState(Bool[num_wires], Wire[num_wires]): 1,
        DoubleExcitation: n_layers * len_wire_pattern,
        OrbitalRotation: rotation_count,
    }


@register_resources(_gate_fabric_resources)
def _gate_fabric_decomposition(weights, wires, init_state, include_pi):
    if isinstance(wires, Wires):
        wires = wires.labels

    n_layers = math.shape(weights)[0]

    wire_pattern = [wires[i : i + 4] for i in range(0, len(wires), 4) if len(wires[i : i + 4]) == 4]
    if len(wires) > 4:
        wire_pattern += [
            wires[i : i + 4] for i in range(2, len(wires), 4) if len(wires[i : i + 4]) == 4
        ]

    if capture.enabled():
        weights, wires, init_state, wire_pattern, n_layers = (
            jnp.array(weights),
            jnp.array(wires),
            jnp.array(init_state),
            jnp.array(wire_pattern),
            jnp.array(n_layers),
        )

    BasisState(init_state, wires=wires)

    @for_loop(n_layers)
    def layers_loop(layer):

        @for_loop(len(wire_pattern))
        def wires_loop(idx):
            wires_ = wire_pattern[idx]

            def true_body():
                OrbitalRotation(np.pi, wires=wires_)

            cond(include_pi, true_body)()

            DoubleExcitation(weights[layer][idx][0], wires=wires_)
            OrbitalRotation(weights[layer][idx][1], wires=wires_)

        wires_loop()  # pylint: disable=no-value-for-parameter

    layers_loop()  # pylint: disable=no-value-for-parameter


add_decomps(GateFabric, _gate_fabric_decomposition)

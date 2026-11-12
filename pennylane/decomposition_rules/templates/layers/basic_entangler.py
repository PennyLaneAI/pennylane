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

"""Built-in decomposition rules for templates.layers.basic_entangler."""

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, cond
from pennylane.templates.layers.basic_entangler import BasicEntanglerLayers


def _basic_entangler_resources(repeat, num_wires, rotation):
    resources = {abstractify(rotation): repeat * num_wires}

    if num_wires == 2:
        resources[CNOT] = repeat

    elif num_wires > 2:
        resources[CNOT] = repeat * num_wires

    return resources


@register_resources(_basic_entangler_resources)
def _basic_entangler_decomposition(weights, wires, rotation):
    repeat = math.shape(weights)[-2]

    if capture.enabled():
        weights, wires = math.array(weights, like="jax"), math.array(wires, like="jax")

    @for_loop(repeat)
    def repeat_loop(layer):

        @for_loop(len(wires))
        def wires_loop(i):
            rotation(weights[..., layer, i], wires=wires[i])

        wires_loop()  # pylint: disable=no-value-for-parameter

        def elif_body():
            for i in range(len(wires)):  # pylint: disable=consider-using-enumerate
                j = (i + 1) % len(wires)
                CNOT(wires=[wires[i], wires[j]])

        def true_body():
            CNOT(wires=wires)

        cond(len(wires) == 2, true_body, false_fn=None, elifs=((len(wires) > 2, elif_body),))()

    repeat_loop()  # pylint: disable=no-value-for-parameter


add_decomps(BasicEntanglerLayers, _basic_entangler_decomposition)

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

"""Built-in decomposition rules for templates.layers.strongly_entangling."""

# pylint: disable=too-many-arguments

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import Rot
from pennylane.ops.op_math import cond
from pennylane.templates.layers.strongly_entangling import StronglyEntanglingLayers


def _strongly_entangling_resources(imprimitive, n_wires, n_layers):
    resources = {}

    resources[Rot] = n_wires * n_layers
    if n_wires > 1:
        resources[imprimitive] = n_wires * n_layers

    return resources


@register_resources(_strongly_entangling_resources)
def _strongly_entangling_decomposition(weights, wires, ranges, imprimitive):

    if capture.enabled():
        wires = math.array(wires, like="jax")
        ranges = math.array(ranges, like="jax")
        weights = math.array(weights, like="jax")

    n_wires = len(wires)
    n_layers = weights.shape[-3]

    @for_loop(n_layers)
    def layers(l):
        @for_loop(n_wires)
        def rot_loop(i):
            Rot(
                weights[..., l, i, 0],
                weights[..., l, i, 1],
                weights[..., l, i, 2],
                wires=wires[i],
            )

        def imprim_true():
            @for_loop(n_wires)
            def imprimitive_loop(i):
                if capture.enabled():
                    act_on = math.array([i, i + ranges[l]], like="jax") % n_wires
                else:
                    act_on = wires.subset([i, i + ranges[l]], periodic_boundary=True)
                imprimitive(wires=act_on)

            imprimitive_loop()  # pylint: disable=no-value-for-parameter

        def imprim_false():
            pass

        rot_loop()  # pylint: disable=no-value-for-parameter
        cond(n_wires > 1, imprim_true, imprim_false)()

    layers()  # pylint: disable=no-value-for-parameter


add_decomps(StronglyEntanglingLayers, _strongly_entangling_decomposition)

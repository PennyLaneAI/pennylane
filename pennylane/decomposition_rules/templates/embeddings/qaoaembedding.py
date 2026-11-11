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

"""Built-in decomposition rules for templates.embeddings.qaoaembedding."""

# pylint: disable=too-many-arguments

from collections import defaultdict

from jax import numpy as jnp

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import RX, H, MultiRZ, cond
from pennylane.templates.embeddings.qaoaembedding import QAOAEmbedding
from pennylane.typing import Float, Wire


def _qaoa_embedding_resources(repeat, n_features, num_wires, local_field):
    multi_rz_count = num_wires * repeat if num_wires > 2 else repeat
    resources = defaultdict(
        int,
        {
            RX: n_features * (repeat + 1),
            H: (num_wires - n_features) * (repeat + 1),
            MultiRZ(Float, Wire[2]): multi_rz_count,
        },
    )
    resources[abstractify(local_field)] += num_wires * repeat
    return resources


@register_resources(_qaoa_embedding_resources)
def _qaoa_embedding_decomposition(features, weights, wires, local_field):
    if capture.enabled():
        weights, wires, features = jnp.array(weights), jnp.array(wires), jnp.array(features)

    repeat = math.shape(weights)[-2]
    n_features = math.shape(features)[-1]
    if math.ndim(features) > 1:
        features = math.T(features)
    if math.ndim(weights) > 2:
        weights = math.moveaxis(weights, 0, -1)

    @for_loop(repeat)
    def repeat_loop(l):

        @for_loop(n_features)
        def features_loop(i):
            RX(features[i], wires=wires[i])

        features_loop()  # pylint: disable=no-value-for-parameter

        @for_loop(n_features, len(wires))
        def hadamard_loop(i):
            H(wires=wires[i])

        hadamard_loop()  # pylint: disable=no-value-for-parameter

        def true_body():
            local_field(weights[l][0], wires=wires)

        def false_body():

            @for_loop(len(wires))
            def multi_rz_loop(i):
                MultiRZ(weights[l][i], wires=[wires[i], wires[(i + 1) % len(wires)]])

            multi_rz_loop()  # pylint: disable=no-value-for-parameter

            @for_loop(len(wires))
            def local_field_loop(i):
                local_field(weights[l][len(wires) + i], wires=wires[i])

            local_field_loop()  # pylint: disable=no-value-for-parameter

        def elif_body():
            MultiRZ(weights[l][0], wires=[wires[0], wires[1]])
            local_field(weights[l][1], wires=wires[0:1])
            local_field(weights[l][2], wires=wires[1:2])

        cond(len(wires) == 1, true_body, false_body, ((len(wires) == 2, elif_body),))()

    repeat_loop()  # pylint: disable=no-value-for-parameter

    @for_loop(n_features)
    def final_rx_loop(i):
        RX(features[i], wires=wires[i])

    final_rx_loop()  # pylint: disable=no-value-for-parameter

    @for_loop(n_features, len(wires))
    def final_hadamard_loop(i):
        H(wires=wires[i])

    final_hadamard_loop()  # pylint: disable=no-value-for-parameter


add_decomps(QAOAEmbedding, _qaoa_embedding_decomposition)

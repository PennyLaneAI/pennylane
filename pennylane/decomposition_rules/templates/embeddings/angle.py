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

"""Built-in decomposition rules for templates.embeddings.angle."""

from jax import numpy as jnp

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operation, abstractify
from pennylane.decomposition import add_decomps, register_resources
from pennylane.templates.embeddings.angle import AngleEmbedding
from pennylane.wires import WiresLike


def _angle_embedding_resources(rotation: Operation, num_wires: int) -> dict:
    return {abstractify(rotation): num_wires}


@register_resources(_angle_embedding_resources)
def _angle_embedding_decomposition(features: list, wires: WiresLike, rotation: Operation):
    batched = math.ndim(features) > 1
    # We will iterate over the first axis of `features` together with iterating over the wires.
    # If the leading dimension is a batch dimension, exchange the wire and batching axes.
    features = math.T(features) if batched else features

    if capture.enabled():
        features, wires = jnp.array(features), jnp.array(wires)

    @for_loop(len(wires))
    def rotation_loop(i):
        rotation(features[i], wires=wires[i])

    rotation_loop()  # pylint: disable=no-value-for-parameter


add_decomps(AngleEmbedding, _angle_embedding_decomposition)

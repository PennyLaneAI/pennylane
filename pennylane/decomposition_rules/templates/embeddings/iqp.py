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

"""Built-in decomposition rules for templates.embeddings.iqp."""

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import RZ, H, MultiRZ
from pennylane.templates.embeddings.iqp import IQPEmbedding
from pennylane.typing import Float, Wire
from pennylane.wires import WiresLike


# pylint: disable=unused-argument
def _iqp_embedding_resources(features, wires, n_repeats, pattern):
    return {
        RZ: n_repeats * len(wires),
        H: n_repeats * len(wires),
        MultiRZ(Float, Wire[2]): len(pattern) * n_repeats,
    }


@register_resources(_iqp_embedding_resources, exact=False)
def _iqp_embedding_decomposition(features, wires: WiresLike, n_repeats, pattern):

    if capture.enabled() or compiler.active():
        wires = math.array(wires, like="jax")

    if math.ndim(features) > 1:
        features = math.T(features)

    @for_loop(n_repeats)
    def outer_loop(_):

        @for_loop(len(wires))
        def single_qubit_loop(i):
            H(wires=wires[i])
            RZ(features[i], wires=wires[i])

        single_qubit_loop()  # pylint: disable=no-value-for-parameter

        @for_loop(len(pattern))
        def pattern_loop(j):
            idx1, idx2 = pattern[j][0], pattern[j][1]
            MultiRZ(features[idx1] * features[idx2], wires=[wires[idx1], wires[idx2]])

        pattern_loop()  # pylint: disable=no-value-for-parameter

    outer_loop()  # pylint: disable=no-value-for-parameter


add_decomps(IQPEmbedding, _iqp_embedding_decomposition)

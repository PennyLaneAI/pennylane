# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Entangling layers benchmark.
"""
# pylint: disable=invalid-name
import numpy as np

import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.init import strong_ent_layers_uniform

import benchmark_utils as bu


def circuit(weights, *, features=None):
    """Immutable quantum circuit."""

    # normally not allowed in immutable circuits, but here we know the len will not change...
    n_wires = len(features)
    AngleEmbedding(features, wires=range(n_wires))
    StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0))


class Benchmark(bu.BaseBenchmark):
    """Entangling layers benchmark.

    Creates an immutable QNode using the StronglyEntanglingLayers template,
    then evaluates it and its Jacobian.
    """

    name = "entangling layers"
    min_wires = 2
    n_vals = range(1, 5)

    def benchmark(self, n=3):
        # n is the number of layers in the circuit
        if self.verbose:
            print("circuit: {} layers, {} wires".format(n, self.n_wires))

        features = np.arange(self.n_wires)
        init_weights = strong_ent_layers_uniform(n_layers=n, n_wires=self.n_wires)

        qnode = bu.create_qnode(circuit, self.device, mutable=False)
        qnode(init_weights, features=features)
        qnode.jacobian((init_weights,), {"features": features})
        return True

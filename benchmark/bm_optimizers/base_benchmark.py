# Copyright 2020 Xanadu Quantum Technologies Inc.

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
PennyLane builtin optimizers base benchmark
"""
import numpy as np

import benchmark_utils as bu
import pennylane as qml
from pennylane.init import strong_ent_layers_uniform
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.layers import StronglyEntanglingLayers


def circuit(weights, features=None):
    """Mutable strongly entangling layers.

    A quantum circuit that embeds the ``features`` using ``AngleEmbedding`` and then
    applies a sequence of ``StronglyEntanglingLayers``.

    Args:
        weights (array): a weights array of size (n_layers, n_wires, 3)
        features (array): the input features with dimension equal to the number of wires

    Returns:
        float: the expectation value of the the PauliZ observable on the first wire
    """
    n_wires = len(features)
    AngleEmbedding(features, wires=range(n_wires))
    StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.expval(qml.PauliZ(wires=0))


class BaseBenchmark(bu.BaseBenchmark):
    """Base benchmark class for the optimizer-based benchmarks.

    This class provides the benchmark method, which creates the mutable QNode using
    ``AngleEmbedding`` and ``StronglyEntanglingLayers`` and then executes one step of the
    benchmark's optimizer.

    Inheriting classes need simply to define benchmark name and the ``optimizer`` attribute. For
    example, the ``gds`` benchmark defines ``optimizer = qml.GradientDescentOptimizer()``.
    """

    min_wires = 1
    n_vals = range(1, 5)

    def benchmark(self, n=3):
        """n is the number of layers in the circuit."""
        if self.verbose:
            print("circuit: {} layers, {} wires using {}".format(n, self.n_wires, self.name))

        weights = strong_ent_layers_uniform(n_layers=n, n_wires=self.n_wires, seed=1967)
        features = np.ones(self.n_wires)

        qnode = bu.create_qnode(circuit, self.device, mutable=True)
        self.optimizer.step(lambda x: qnode(x, features=features), weights)
        return True

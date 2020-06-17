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
Benchmark for evaluating a circuit consisting only of nearest-neighbour
two-qubit gates.
"""
from math import pi

import numpy as np
import pennylane as qml

import benchmark_utils as bu


class Benchmark(bu.BaseBenchmark):
    """Nearest-neighbour circuit benchmark.

    Creates a parametrized quantum circuit with n layers.
    Each layer consists of single-qubit parametrized rotations,
    and two-qubit gates between nearest-neighbour qubits.
    """

    name = "Nearest neighbour"
    min_wires = 2
    n_vals = range(1, 10)

    def setup(self):
        # pylint: disable=attribute-defined-outside-init,no-member
        np.random.seed(143)
        self.params1 = np.random.uniform(high=2 * pi, size=self.n_wires)
        self.params2 = np.random.uniform(high=2 * pi, size=self.n_wires-1)
        self.all_wires = range(self.n_wires)

    def benchmark(self, n=10):
        # n is the number of layers in the circuit
        if self.verbose:
            print("circuit: {} parameters, {} wires".format(n * self.n_wires, self.n_wires))

        params1 = [qml.numpy.array(self.params1, copy=True, requires_grad=True) for _ in range(n)]
        params2 = [qml.numpy.array(self.params2, copy=True, requires_grad=True) for _ in range(n)]
        def circuit(params1, params2):
            """Parametrized circuit with nearest-neighbour gates."""
            for layer in range(n):
                qml.broadcast(qml.RX, pattern="single", wires=self.all_wires, parameters=params1[layer])
                qml.broadcast(qml.CRY, pattern="chain", wires=self.all_wires, parameters=params2[layer])
            return bu.expval(qml.PauliZ(0))

        qnode = bu.create_qnode(circuit, self.device, mutable=True, qnode_type=self.qnode_type)
        qnode(params1, params2)

        return True

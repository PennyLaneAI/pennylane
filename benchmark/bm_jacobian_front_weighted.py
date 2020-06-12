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
Benchmark for a computing the jacobian of a QNode, where the parametrized
gates are uniformly distributed throughout the circuit.
"""
from math import pi

import numpy as np
import pennylane as qml

import benchmark_utils as bu


class Benchmark(bu.BaseBenchmark):
    """Jacobian computation benchmark.

    Creates a parametrized quantum circuit with a front-weighted
    distribution of parametrized gates in the circuit
    and evaluates its Jacobian.
    """

    name = "Jacobian evaluation uniform"
    min_wires = 2
    n_vals = range(3, 27, 3)

    def setup(self):
        # pylint: disable=attribute-defined-outside-init,no-member
        np.random.seed(143)
        angles = np.random.uniform(high=2 * pi, size=self.n_wires)
        self.random_angles = angles
        self.all_wires = range(self.n_wires)

    def benchmark(self, n=10):
        # n is the number of parametrized layers in the circuit
        if self.verbose:
            print("circuit: {} parameters, {} wires".format(n * self.n_wires, self.n_wires))

        params = [qml.numpy.array(self.random_angles, copy=True, requires_grad=True) for _ in range(n)]
        def circuit(params):
            """Parametrized circuit."""
            for layer in range(n):
                qml.broadcast(qml.RX, pattern="single", wires=self.all_wires, parameters=params[layer])
            for _layer in range(n):
                qml.broadcast(qml.CNOT, pattern="double", wires=self.all_wires)
            return [bu.expval(qml.PauliZ(w)) for w in self.all_wires]

        qnode = bu.create_qnode(circuit, self.device, mutable=True, qnode_type=self.qnode_type)
        qnode.jacobian([params])

        return True

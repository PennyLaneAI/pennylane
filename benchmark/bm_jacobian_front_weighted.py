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

import pennylane as qml

import benchmark_utils as bu


class Benchmark(bu.BaseBenchmark):
    """Jacobian computation benchmark.

    Creates a parametrized quantum circuit with a front-weighted
    distribution of parametrized gates throughout in circuit
    and evaluates its Jacobian.
    """

    name = "Jacobian evaluation uniform"
    min_wires = 2
    n_vals = range(3, 27, 3)

    def setup(self):
        # pylint: disable=attribute-defined-outside-init,no-member
        qml.numpy.random.seed(143)
        angles = qml.numpy.random.uniform(high=2 * pi, size=self.n_wires)
        angles.requires_grad = True
        self.random_angles = angles

    def benchmark(self, n=10):
        # n is the number of parametrized layers in the circuit
        if self.verbose:
            print("circuit: {} parameters, {} wires".format(n * self.n_wires, self.n_wires))

        all_wires = range(self.n_wires)

        def circuit(angles):
            """Parametrized circuit."""
            for _ in range(n):
                qml.broadcast(qml.RX, pattern="single", wires=all_wires, parameters=angles)
            for _ in range(n):
                qml.broadcast(qml.CNOT, pattern="double", wires=all_wires)
            return [bu.expval(qml.PauliZ(w)) for w in all_wires]

        qnode = bu.create_qnode(circuit, self.device, mutable=True, qnode_type=self.qnode_type)
        qnode.jacobian([self.random_angles])

        return True

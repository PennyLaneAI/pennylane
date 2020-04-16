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
Benchmark for a circuit of the Instantaneous quantum polynomial-time (IQP) complexity class.
"""
# pylint: disable=invalid-name
import math
import random

import numpy as np
import pennylane as qml

import benchmark_utils as bu

CCZ_diag = np.array([1, 1, 1, 1, 1, 1, 1, -1])
CCZ_matrix = np.diag(CCZ_diag)

if hasattr(qml, "DiagonalQubitUnitary"):
    CCZ = lambda wires: qml.DiagonalQubitUnitary(CCZ_diag, wires=wires)
else:
    CCZ = lambda wires: qml.QubitUnitary(CCZ_matrix, wires=wires)


def random_iqp_wires(n_wires):
    """Create a random set of IQP wires.

    Returns a list of either 1, 2, or 3 distinct integers
    in the range of n_wires.

    Args:
        n_wires (int): Number of wires of the device.

    Returns:
        List[int]: The IQP wires.
    """
    # The global seed was fixed during benchmark construction
    # so this is actually deterministic
    a = random.random()
    return random.sample(range(n_wires), math.ceil(3 * a))


def circuit(n=10, n_wires=3):
    """Mutable IQP quantum circuit."""

    for i in range(n_wires):
        qml.Hadamard(i)

    for i in range(n * n_wires):
        wires = random_iqp_wires(n_wires)

        if len(wires) == 1:
            qml.PauliZ(wires=wires)
        elif len(wires) == 2:
            qml.CZ(wires=wires)
        elif len(wires) == 3:
            CCZ(wires)

    for i in range(n_wires):
        qml.Hadamard(i)

    return qml.expval(qml.PauliZ(0))


class Benchmark(bu.BaseBenchmark):
    """IQP circuit benchmark.

    Creates an immutable QNode using an example IQP circuit.
    """

    name = "IQP circuit"
    min_wires = 3
    n_vals = range(3, 27, 3)

    def benchmark(self, n=10):
        # Set seed to make iqp circuits deterministic
        random.seed(135)

        # n is the number of layers in the circuit
        if self.verbose:
            print("circuit: {} IQP gates, {} wires".format(n * self.n_wires, self.n_wires))

        qnode = bu.create_qnode(circuit, self.device, mutable=True)
        qnode(n=n, n_wires=self.n_wires)

        return True

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
Mutable sequence of rotations benchmark.
"""
# pylint: disable=invalid-name
import numpy as np

import pennylane as qml
import benchmark_utils as bu



def circuit(a, *, b=1):
    """Mutable quantum circuit."""
    for idx in range(b):
        qml.RY(a[idx], wires=[0])
    return qml.expval(qml.PauliX(0))


class Benchmark(bu.BaseBenchmark):
    """Mutable sequence of rotations benchmark.

    The benchmark consists of a single mutable QNode evaluated several times,
    each evaluation having a larger number of rotations in the circuit.
    """
    name = 'mutable sequence of rotations'
    min_wires = 1
    n_vals = range(10, 60, 10)

    def setup(self):
        self.qnode = bu.create_qnode(circuit, self.device, mutable=True)

    def benchmark(self, n=30):
        # n is the number of circuit evaluations.
        # Execution time should grow quadratically in n as
        # the circuit size grows linearly with evaluation number.

        wrong_results = 0
        for i in range(n):
            params = np.random.rand(i)
            res = self.qnode(params, b=i)

            expected = np.sin(np.sum(params))
            if np.abs(res - expected) > 1e-6:
                wrong_results += 1

        # print("Wrong results: {}/{}".format(wrong_results, n))
        return True

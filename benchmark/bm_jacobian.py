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
Benchmark for a computing the jacobian of a QNode.
"""
from types import ModuleType

import numpy as np
import pennylane as qml

import benchmark_utils as bu

if type(qml.expval) == ModuleType:
    meas_function = lambda w: qml.expval.PauliZ(w)
else:
    meas_function = lambda w: qml.expval(qml.PauliZ(w))

class Benchmark(bu.BaseBenchmark):
    """Jacobian computation benchmark.

    Creates a parametrized quantum circuit and computes its jacobian
    """

    name = "Jacobian computation"
    min_wires = 2
    n_vals = range(3, 27, 3)

    def benchmark(self, n=10):
        np.random.seed(135)

        # n is the number of parameters in the circuit
        if self.verbose:
            print("circuit: {} parameters, {} wires".format(n, self.n_wires))

        def circuit():
            """Parametrized circuit."""
            qml.RX(0.5, wires=0)

            return meas_function(0)

        qnode = bu.create_qnode(circuit, self.device, mutable=True, qnode_type=self.qnode_type)
        qnode()

        return True

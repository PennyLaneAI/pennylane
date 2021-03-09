# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from pennylane.tape.transforms.adjoint import adjoint
import pennylane as qml

def test_adjoint_on_function():
    """Test that adjoint works when applied to a function"""

    dev = qml.device("default.qubit", wires=1)

    def my_op():
        qml.RX(0.123, wires=0)
        qml.RY(2.32, wires=0)
        qml.RZ(1.95, wires=0)

    @qml.qnode(dev)
    def my_circuit():
        qml.PauliX(wires=0)
        my_op()
        adjoint(my_op)()
        return qml.state()

    np.testing.assert_allclose(my_circuit(), np.array([0.0, 1.0]), atol=1e-6, rtol=1e-6)

def test_adjoint_directly_on_op():
    """Test that adjoint works when directly applyed to an op"""

    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev)
    def my_circuit():
        qml.RX(0.123, wires=0)
        adjoint(qml.RX)(0.123, wires=0)
        return qml.state()

    np.testing.assert_allclose(my_circuit(), np.array([1.0, 0.0]))

def test_nested_adjoint():
    """Test that adjoint works when nested with other adjoints"""
    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev)
    def my_circuit():
        adjoint(qml.RX)(0.123, wires=0)
        adjoint(adjoint(qml.RX))(0.123, wires=0)
        return qml.state()

    np.testing.assert_allclose(my_circuit(), np.array([1.0, 0.0]))

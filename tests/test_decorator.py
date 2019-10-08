# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.qnode` decorator.
"""
# pylint: disable=protected-access,cell-var-from-loop
import numpy as np

import pennylane as qml


class TestMethodBinding:
    """Test QNode methods are correctly bound to
    the wrapped function"""

    def test_jacobian(self):
        """Test binding of jacobian method"""
        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        assert hasattr(circuit, 'jacobian')

        def circuit2(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        qnode = qml.QNode(circuit2, dev)

        assert qnode.jacobian(0.5) == circuit.jacobian(0.5)

    def test_metric_tensor(self, tol):
        """Test binding of metric tensor methods"""
        dev = qml.device('default.qubit', wires=1)

        a, b = 0.4, 0.1

        @qml.qnode(dev)
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        assert hasattr(circuit, 'metric_tensor')

        def circuit2(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        qnode = qml.QNode(circuit2, dev)

        # check that both QNode constructions agree
        res = circuit.metric_tensor(a, b)
        res2 = qnode.metric_tensor(a, b)
        assert np.allclose(res, res2, atol=tol, rtol=0)

        # check metric tensor is correct
        expected = np.diag(np.array([1, np.cos(a)**2])/4)
        assert np.allclose(res, expected, atol=tol, rtol=0)

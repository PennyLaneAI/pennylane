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
Unit tests for the aqft template.
"""
import pytest

import numpy as np

import pennylane as qml


class TestAQFT:
    """Tests for the aqft operations"""

    @pytest.mark.parametrize("order,n_qubits", [(o, w) for w in range(2, 10) for o in range(1, w)])
    def test_AQFT_adjoint_identity(self, order, n_qubits, tol):
        """Test if after using the qml.adjoint transform the resulting operation is
        the inverse of AQFT."""

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circ(n_qubits, order):
            qml.adjoint(qml.AQFT)(order=order, wires=range(n_qubits))
            qml.AQFT(order=order, wires=range(n_qubits))
            return qml.state()

        assert np.allclose(1, circ(n_qubits, order)[0], tol)

        for i in range(1, n_qubits):
            assert np.allclose(0, circ(n_qubits, order)[i], tol)

    @pytest.mark.parametrize("order", [-1, -5.4])
    def test_negative_order(self, order):
        """Test if ValueError is raised for negative orders"""
        with pytest.raises(ValueError, match="Order can not be less than 0"):
            qml.AQFT(order=order, wires=range(5))

    @pytest.mark.parametrize("order", [1.2, 4.6])
    def test_float_order(self, order):
        """Test if float order is handled correctly"""
        with pytest.warns(UserWarning, match="The order must be an integer"):
            op = qml.AQFT(order=order, wires=range(9))
            assert op.hyperparameters["order"] == round(order)

    @pytest.mark.parametrize("wires", range(3, 10))
    def test_zero_order(self, wires):
        """Test if Hadamard transform is applied for zero order"""
        with pytest.warns(UserWarning, match="order=0"):
            op = qml.AQFT(order=0, wires=range(wires))
            for gate in op.decomposition()[: -wires // 2]:
                assert gate.name == "Hadamard"

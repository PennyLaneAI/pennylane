# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    op = qp.AQFT(order=2, wires=(0, 1, 2))
    qp.ops.functions.assert_valid(op)


class TestAQFT:
    """Tests for the aqft operations"""

    @pytest.mark.parametrize("order,n_qubits", [(o, w) for w in range(2, 10) for o in range(1, w)])
    def test_AQFT_adjoint_identity(self, order, n_qubits, tol):
        """Test if after using the qp.adjoint transform the resulting operation is
        the inverse of AQFT."""

        dev = qp.device("default.qubit", wires=n_qubits)

        @qp.qnode(dev)
        def circ(n_qubits, order):
            qp.adjoint(qp.AQFT)(order=order, wires=range(n_qubits))
            qp.AQFT(order=order, wires=range(n_qubits))
            return qp.state()

        assert np.allclose(1, circ(n_qubits, order)[0], tol)

        for i in range(1, n_qubits):
            assert np.allclose(0, circ(n_qubits, order)[i], tol)

    @pytest.mark.parametrize("order", [-1, -5.4])
    def test_negative_order(self, order):
        """Test if ValueError is raised for negative orders"""
        with pytest.raises(ValueError, match="Order can not be less than 0"):
            qp.AQFT(order=order, wires=range(5))

    @pytest.mark.parametrize("order", [1.2, 4.6])
    def test_float_order(self, order):
        """Test if float order is handled correctly"""
        with pytest.warns(UserWarning, match="The order must be an integer"):
            op = qp.AQFT(order=order, wires=range(9))
            assert op.hyperparameters["order"] == round(order)

    @pytest.mark.parametrize("wires", range(3, 10))
    def test_zero_order(self, wires):
        """Test if Hadamard transform is applied for zero order"""
        with pytest.warns(UserWarning, match="order=0"):
            op = qp.AQFT(order=0, wires=range(wires))
            for gate in op.decomposition()[: -wires // 2]:
                assert gate.name == "Hadamard"

    @pytest.mark.parametrize("order", [4, 5, 6])
    def test_higher_order(self, order):
        """Test if higher order recommends using QFT"""
        with pytest.warns(UserWarning, match="Using the QFT class is recommended in this case"):
            qp.AQFT(order=order, wires=range(5))

    @pytest.mark.parametrize("wires", [3, 4, 5, 6, 7, 8, 9])
    def test_matrix_higher_order(self, wires):
        """Test if the matrix from AQFT and QFT are same for higher order"""

        m1 = qp.matrix(qp.AQFT(order=10, wires=range(wires)))
        m2 = qp.matrix(qp.QFT(wires=range(wires)))

        assert np.allclose(m1, m2)

    @pytest.mark.parametrize("order,wires", [(o, w) for w in range(2, 10) for o in range(1, w)])
    def test_decomposition_new(self, order, wires):
        """Tests the decomposition rule implemented with the new system."""
        op = qp.AQFT(order=order, wires=range(wires))

        for rule in qp.list_decomps(qp.AQFT):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize("order,wires", [(o, w) for w in range(2, 10) for o in range(1, w)])
    @pytest.mark.capture
    def test_decomposition_new_capture(self, order, wires):
        """Tests the decomposition rule implemented with the new system when program capture is enabled."""
        op = qp.AQFT(order=order, wires=range(wires))

        for rule in qp.list_decomps(qp.AQFT):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize("order,wires", [(o, w) for w in range(2, 10) for o in range(1, w)])
    def test_gates(self, order, wires):
        """Test if the AQFT operation consists of only 3 type of gates"""

        op = qp.AQFT(order=order, wires=range(wires))
        decomp = op.decomposition()

        for gate in decomp:
            assert gate.name in ["Hadamard", "ControlledPhaseShift", "SWAP"]

    @pytest.mark.jax
    def test_jax_jit(self):
        import jax

        wires = 3
        dev = qp.device("default.qubit", wires=wires)

        @qp.qnode(dev)
        def circuit():
            qp.X(0)
            qp.Hadamard(1)
            qp.AQFT(order=1, wires=range(wires))
            return qp.state()

        jit_circuit = jax.jit(circuit)

        res = circuit()
        res2 = jit_circuit()
        assert qp.math.allclose(res, res2)

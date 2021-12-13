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
Unit tests for the qft template.
"""
import pytest

import numpy as np

from gate_data import QFT
import pennylane as qml


class TestQFT:
    """Tests for the qft operations"""

    @pytest.mark.parametrize("inverse", [True, False])
    def test_QFT(self, inverse):
        """Test if the QFT matrix is equal to a manually-calculated version for 3 qubits"""
        op = qml.QFT(wires=range(3)).inv() if inverse else qml.QFT(wires=range(3))
        res = op.matrix
        exp = QFT.conj().T if inverse else QFT
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("num_inversions", [1, 2, 3])
    def test_QFT_adjoint_method(self, num_inversions):
        """Test the adjoint method of the QFT class"""
        op = qml.QFT(wires=range(3))

        for _ in range(num_inversions):
            op = op.adjoint()

        res = op.matrix
        inverse = num_inversions % 2 == 1
        exp = QFT.conj().T if inverse else QFT
        assert np.allclose(res, exp)
        assert op.inverse is inverse

    @pytest.mark.parametrize("n_qubits", range(2, 6))
    def test_QFT_decomposition(self, n_qubits):
        """Test if the QFT operation is correctly decomposed"""
        op = qml.QFT(wires=range(n_qubits))
        decomp = op.decompose()

        dev = qml.device("default.qubit", wires=n_qubits)

        out_states = []
        for state in np.eye(2 ** n_qubits):
            dev.reset()
            ops = [qml.QubitStateVector(state, wires=range(n_qubits))] + decomp
            dev.apply(ops)
            out_states.append(dev.state)

        reconstructed_unitary = np.array(out_states).T
        expected_unitary = qml.QFT(wires=range(n_qubits)).matrix

        assert np.allclose(reconstructed_unitary, expected_unitary)

    @pytest.mark.parametrize("n_qubits", range(2, 6))
    def test_QFT_adjoint_identity(self, n_qubits, tol):
        """Test if using the qml.adjoint transform the resulting operation is
        the inverse of QFT."""

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circ(n_qubits):
            qml.adjoint(qml.QFT)(wires=range(n_qubits))
            qml.QFT(wires=range(n_qubits))
            return qml.state()

        assert np.allclose(1, circ(n_qubits)[0], tol)

        for i in range(1, n_qubits):
            assert np.allclose(0, circ(n_qubits)[i], tol)

    @pytest.mark.parametrize("n_qubits", range(2, 6))
    def test_QFT_adjoint_decomposition(self, n_qubits):  # tol
        """Test if using the qml.adjoint transform results in the right
        decomposition."""

        # QFT adjoint has right decompositions
        qft = qml.QFT(wires=range(n_qubits))
        qft_dec = qft.expand().operations

        expected_op = [x.adjoint() for x in qft_dec]
        expected_op.reverse()

        adj = qml.QFT(wires=range(n_qubits)).adjoint()
        op = adj.expand().operations

        for j in range(0, len(op)):
            assert op[j].name == expected_op[j].name
            assert op[j].wires == expected_op[j].wires
            assert op[j].parameters == expected_op[j].parameters

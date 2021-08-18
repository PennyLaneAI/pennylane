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

import pennylane as qml

from gate_data import (
    I,
    X,
    Y,
    Z,
    H,
    StateZeroProjector,
    StateOneProjector,
    CNOT,
    SWAP,
    ISWAP,
    CZ,
    S,
    T,
    CSWAP,
    Toffoli,
    QFT,
    ControlledPhaseShift,
    SingleExcitation,
    SingleExcitationPlus,
    SingleExcitationMinus,
    DoubleExcitation,
    DoubleExcitationPlus,
    DoubleExcitationMinus,
)

# Non-parametrized operations and their matrix representation
NON_PARAMETRIZED_OPERATIONS = [
    (qml.CNOT, CNOT),
    (qml.SWAP, SWAP),
    (qml.ISWAP, ISWAP),
    (qml.CZ, CZ),
    (qml.S, S),
    (qml.T, T),
    (qml.CSWAP, CSWAP),
    (qml.Toffoli, Toffoli),
]

PARAMETRIZED_OPERATIONS = [
    qml.RX(0.123, wires=0),
    qml.RY(1.434, wires=0),
    qml.RZ(2.774, wires=0),
    qml.S(wires=0),
    qml.SX(wires=0),
    qml.T(wires=0),
    qml.CNOT(wires=[0, 1]),
    qml.CZ(wires=[0, 1]),
    qml.CY(wires=[0, 1]),
    qml.SWAP(wires=[0, 1]),
    qml.ISWAP(wires=[0, 1]),
    qml.CSWAP(wires=[0, 1, 2]),
    qml.PauliRot(0.123, "Y", wires=0),
    qml.IsingXX(0.123, wires=[0, 1]),
    qml.IsingYY(0.123, wires=[0, 1]),
    qml.IsingZZ(0.123, wires=[0, 1]),
    qml.Rot(0.123, 0.456, 0.789, wires=0),
    qml.Toffoli(wires=[0, 1, 2]),
    qml.PhaseShift(2.133, wires=0),
    qml.ControlledPhaseShift(1.777, wires=[0, 2]),
    qml.CPhase(1.777, wires=[0, 2]),
    qml.MultiRZ(0.112, wires=[1, 2, 3]),
    qml.CRX(0.836, wires=[2, 3]),
    qml.CRY(0.721, wires=[2, 3]),
    qml.CRZ(0.554, wires=[2, 3]),
    qml.U1(0.123, wires=0),
    qml.U2(3.556, 2.134, wires=0),
    qml.U3(2.009, 1.894, 0.7789, wires=0),
    qml.Hadamard(wires=0),
    qml.PauliX(wires=0),
    qml.PauliZ(wires=0),
    qml.PauliY(wires=0),
    qml.CRot(0.123, 0.456, 0.789, wires=[0, 1]),
    qml.QubitUnitary(np.eye(2) * 1j, wires=0),
    qml.DiagonalQubitUnitary(np.array([1.0, 1.0j]), wires=1),
    qml.ControlledQubitUnitary(np.eye(2) * 1j, wires=[0], control_wires=[2]),
    qml.MultiControlledX(control_wires=[0, 1], wires=2, control_values="01"),
    qml.SingleExcitation(0.123, wires=[0, 3]),
    qml.SingleExcitationPlus(0.123, wires=[0, 3]),
    qml.SingleExcitationMinus(0.123, wires=[0, 3]),
    qml.DoubleExcitation(0.123, wires=[0, 1, 2, 3]),
    qml.DoubleExcitationPlus(0.123, wires=[0, 1, 2, 3]),
    qml.DoubleExcitationMinus(0.123, wires=[0, 1, 2, 3]),
    qml.QubitSum(wires=[0, 1, 2]),
]


class TestOperations:
    """Tests for the qft operations"""

    @pytest.mark.parametrize("ops, mat", NON_PARAMETRIZED_OPERATIONS)
    def test_matrices(self, ops, mat, tol):
        """Test matrices of non-parametrized operations are correct"""
        op = ops(wires=range(ops.num_wires))
        res = op.matrix
        assert np.allclose(res, mat, atol=tol, rtol=0)

    @pytest.mark.parametrize("op", PARAMETRIZED_OPERATIONS)
    def test_adjoint_unitaries(self, op, tol):
        op_d = op.adjoint()
        res1 = np.dot(op.matrix, op_d.matrix)
        res2 = np.dot(op_d.matrix, op.matrix)
        np.testing.assert_allclose(res1, np.eye(2 ** len(op.wires)), atol=tol)
        np.testing.assert_allclose(res2, np.eye(2 ** len(op.wires)), atol=tol)
        assert op.wires == op_d.wires

    @pytest.mark.parametrize(
        "op_builder",
        [
            lambda: qml.QubitCarry(wires=[0, 1, 2, 3]),
        ],
    )
    def test_adjoint_with_decomposition(self, op_builder):
        op = op_builder()
        decomposed_ops = op.decomposition(wires=op.wires)
        with qml.tape.QuantumTape() as adjoint_tape:
            qml.adjoint(op_builder)()
        for a, b in zip(decomposed_ops, reversed(adjoint_tape.operations)):
            np.testing.assert_allclose(a.matrix, np.conj(b.matrix).T)

    @pytest.mark.parametrize(
        "op",
        [
            qml.BasisState(np.array([0, 1]), wires=0),
            qml.QubitStateVector(np.array([1.0, 0.0]), wires=0),
        ],
    )
    def test_adjoint_error_exception(self, op, tol):
        with pytest.raises(qml.ops.AdjointError):
            op.adjoint()

    @pytest.mark.parametrize("inverse", [True, False])
    def test_QFT(self, inverse):
        """Test if the QFT matrix is equal to a manually-calculated version for 3 qubits"""
        op = qml.QFT(wires=range(3)).inv() if inverse else qml.QFT(wires=range(3))
        res = op.matrix
        exp = QFT.conj().T if inverse else QFT
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("n_qubits", range(2, 6))
    def test_QFT_decomposition(self, n_qubits):
        """Test if the QFT operation is correctly decomposed"""
        op = qml.QFT(wires=range(n_qubits))
        decomp = op.decomposition(wires=range(n_qubits))

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
        """Test if the QFT adjoint operation is the inverse of QFT."""

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
        """Test if the QFT adjoint operation has the right decomposition"""

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

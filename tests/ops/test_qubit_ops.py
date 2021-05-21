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
Unit tests for the available built-in discrete-variable quantum operations.
"""
import itertools
import re
import pytest
import functools
import numpy as np
from numpy.linalg import multi_dot
from scipy.stats import unitary_group
from scipy.linalg import expm

import pennylane as qml
from pennylane.wires import Wires

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


# Standard observables, their matrix representation, and eigenvalues
OBSERVABLES = [
    (qml.PauliX, X, [1, -1]),
    (qml.PauliY, Y, [1, -1]),
    (qml.PauliZ, Z, [1, -1]),
    (qml.Hadamard, H, [1, -1]),
    (qml.Identity, I, [1, 1]),
]

# Hermitian matrices, their corresponding eigenvalues and eigenvectors.
EIGVALS_TEST_DATA = [
    (np.array([[1, 0], [0, 1]]), np.array([1.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]])),
    (
        np.array([[0, 1], [1, 0]]),
        np.array([-1.0, 1.0]),
        np.array([[-0.70710678, 0.70710678], [0.70710678, 0.70710678]]),
    ),
    (
        np.array([[0, -1j], [1j, 0]]),
        np.array([-1.0, 1.0]),
        np.array(
            [[-0.70710678 + 0.0j, -0.70710678 + 0.0j], [0.0 + 0.70710678j, 0.0 - 0.70710678j]]
        ),
    ),
    (np.array([[1, 0], [0, -1]]), np.array([-1.0, 1.0]), np.array([[0.0, 1.0], [1.0, 0.0]])),
    (
        1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]),
        np.array([-1.0, 1.0]),
        np.array([[0.38268343, -0.92387953], [-0.92387953, -0.38268343]]),
    ),
]

EIGVALS_TEST_DATA_MULTI_WIRES = [functools.reduce(np.kron, [Y, I, Z])]


@pytest.mark.usefixtures("tear_down_hermitian")
class TestObservables:
    """Tests for observables"""

    @pytest.mark.parametrize("obs, mat, eigs", OBSERVABLES)
    def test_diagonalization(self, obs, mat, eigs, tol):
        """Test the method transforms standard observables into the Z-gate."""
        ob = obs(wires=0)
        A = ob.matrix

        diag_gates = ob.diagonalizing_gates()
        U = np.eye(2)

        if diag_gates:
            mats = [i.matrix for i in diag_gates]
            # Need to revert the order in which the matrices are applied such that they adhere to the order
            # of matrix multiplication
            # E.g. for PauliY: [PauliZ(wires=self.wires), S(wires=self.wires), Hadamard(wires=self.wires)]
            # becomes Hadamard @ S @ PauliZ, where @ stands for matrix multiplication
            mats = mats[::-1]
            U = multi_dot([np.eye(2)] + mats)

        res = U @ A @ U.conj().T
        expected = np.diag(eigs)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("obs, mat, eigs", OBSERVABLES)
    def test_eigvals(self, obs, mat, eigs, tol):
        """Test eigenvalues of standard observables are correct"""
        obs = obs(wires=0)
        res = obs.eigvals
        assert np.allclose(res, eigs, atol=tol, rtol=0)

    @pytest.mark.parametrize("obs, mat, eigs", OBSERVABLES)
    def test_matrices(self, obs, mat, eigs, tol):
        """Test matrices of standard observables are correct"""
        obs = obs(wires=0)
        res = obs.matrix
        assert np.allclose(res, mat, atol=tol, rtol=0)

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_eigegendecomposition_single_wire(self, observable, eigvals, eigvecs, tol):
        """Tests that the eigendecomposition property of the Hermitian class returns the correct results
        for a single wire."""

        eigendecomp = qml.Hermitian(observable, wires=0).eigendecomposition
        assert np.allclose(eigendecomp["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(eigendecomp["eigvec"], eigvecs, atol=tol, rtol=0)

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

    @pytest.mark.parametrize("observable", EIGVALS_TEST_DATA_MULTI_WIRES)
    def test_hermitian_eigegendecomposition_multiple_wires(self, observable, tol):
        """Tests that the eigendecomposition property of the Hermitian class returns the correct results
        for multiple wires."""

        num_wires = int(np.log2(len(observable)))
        eigendecomp = qml.Hermitian(observable, wires=list(range(num_wires))).eigendecomposition

        eigvals, eigvecs = np.linalg.eigh(observable)

        assert np.allclose(eigendecomp["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(eigendecomp["eigvec"], eigvecs, atol=tol, rtol=0)

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

    @pytest.mark.parametrize("obs1", EIGVALS_TEST_DATA)
    @pytest.mark.parametrize("obs2", EIGVALS_TEST_DATA)
    def test_hermitian_eigvals_eigvecs_two_different_observables(self, obs1, obs2, tol):
        """Tests that the eigvals method of the Hermitian class returns the correct results
        for two observables."""
        if np.all(obs1[0] == obs2[0]):
            pytest.skip("Test only runs for pairs of differing observable")

        observable_1 = obs1[0]
        observable_1_eigvals = obs1[1]
        observable_1_eigvecs = obs1[2]

        key = tuple(observable_1.flatten().tolist())

        qml.Hermitian(observable_1, 0).eigvals
        assert np.allclose(
            qml.Hermitian._eigs[key]["eigval"], observable_1_eigvals, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.Hermitian._eigs[key]["eigvec"], observable_1_eigvecs, atol=tol, rtol=0
        )
        assert len(qml.Hermitian._eigs) == 1

        observable_2 = obs2[0]
        observable_2_eigvals = obs2[1]
        observable_2_eigvecs = obs2[2]

        key_2 = tuple(observable_2.flatten().tolist())

        qml.Hermitian(observable_2, 0).eigvals
        assert np.allclose(
            qml.Hermitian._eigs[key_2]["eigval"], observable_2_eigvals, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.Hermitian._eigs[key_2]["eigvec"], observable_2_eigvecs, atol=tol, rtol=0
        )
        assert len(qml.Hermitian._eigs) == 2

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_eigvals_eigvecs_same_observable_twice(
        self, observable, eigvals, eigvecs, tol
    ):
        """Tests that the eigvals method of the Hermitian class keeps the same dictionary entries upon multiple calls."""
        key = tuple(observable.flatten().tolist())

        qml.Hermitian(observable, 0).eigvals
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

        qml.Hermitian(observable, 0).eigvals
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_diagonalizing_gates(self, observable, eigvals, eigvecs, tol):
        """Tests that the diagonalizing_gates method of the Hermitian class returns the correct results."""
        qubit_unitary = qml.Hermitian(observable, wires=[0]).diagonalizing_gates()

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

        assert np.allclose(qubit_unitary[0].data, eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

    @pytest.mark.parametrize("obs1", EIGVALS_TEST_DATA)
    @pytest.mark.parametrize("obs2", EIGVALS_TEST_DATA)
    def test_hermitian_diagonalizing_gates_two_different_observables(self, obs1, obs2, tol):
        """Tests that the diagonalizing_gates method of the Hermitian class returns the correct results
        for two observables."""
        if np.all(obs1[0] == obs2[0]):
            pytest.skip("Test only runs for pairs of differing observable")

        observable_1 = obs1[0]
        observable_1_eigvals = obs1[1]
        observable_1_eigvecs = obs1[2]

        qubit_unitary = qml.Hermitian(observable_1, wires=[0]).diagonalizing_gates()

        key = tuple(observable_1.flatten().tolist())
        assert np.allclose(
            qml.Hermitian._eigs[key]["eigval"], observable_1_eigvals, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.Hermitian._eigs[key]["eigvec"], observable_1_eigvecs, atol=tol, rtol=0
        )

        assert np.allclose(qubit_unitary[0].data, observable_1_eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

        observable_2 = obs2[0]
        observable_2_eigvals = obs2[1]
        observable_2_eigvecs = obs2[2]

        qubit_unitary_2 = qml.Hermitian(observable_2, wires=[0]).diagonalizing_gates()

        key = tuple(observable_2.flatten().tolist())
        assert np.allclose(
            qml.Hermitian._eigs[key]["eigval"], observable_2_eigvals, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.Hermitian._eigs[key]["eigvec"], observable_2_eigvecs, atol=tol, rtol=0
        )

        assert np.allclose(qubit_unitary_2[0].data, observable_2_eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 2

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_diagonalizing_gatesi_same_observable_twice(
        self, observable, eigvals, eigvecs, tol
    ):
        """Tests that the diagonalizing_gates method of the Hermitian class keeps the same dictionary entries upon multiple calls."""
        qubit_unitary = qml.Hermitian(observable, wires=[0]).diagonalizing_gates()

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

        assert np.allclose(qubit_unitary[0].data, eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

        qubit_unitary = qml.Hermitian(observable, wires=[0]).diagonalizing_gates()

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

        assert np.allclose(qubit_unitary[0].data, eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_diagonalizing_gates_integration(self, observable, eigvals, eigvecs, tol):
        """Tests that the diagonalizing_gates method of the Hermitian class
        diagonalizes the given observable."""
        tensor_obs = np.kron(observable, observable)
        eigvals = np.kron(eigvals, eigvals)

        diag_gates = qml.Hermitian(tensor_obs, wires=[0, 1]).diagonalizing_gates()

        assert len(diag_gates) == 1

        U = diag_gates[0].parameters[0]
        x = U @ tensor_obs @ U.conj().T
        assert np.allclose(np.diag(np.sort(eigvals)), x, atol=tol, rtol=0)

    def test_hermitian_matrix(self, tol):
        """Test that the hermitian matrix method produces the correct output."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        out = qml.Hermitian(H, wires=0).matrix

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert np.allclose(out, H, atol=tol, rtol=0)

    def test_hermitian_exceptions(self):
        """Tests that the hermitian matrix method raises the proper errors."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be a square matrix"):
            qml.Hermitian(H[1:], wires=0).matrix

        # test non-Hermitian matrix
        H2 = H.copy()
        H2[0, 1] = 2
        with pytest.raises(ValueError, match="must be Hermitian"):
            qml.Hermitian(H2, wires=0).matrix


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


class TestOperations:
    """Tests for the operations"""

    @pytest.mark.parametrize("ops, mat", NON_PARAMETRIZED_OPERATIONS)
    def test_matrices(self, ops, mat, tol):
        """Test matrices of non-parametrized operations are correct"""
        op = ops(wires=range(ops.num_wires))
        res = op.matrix
        assert np.allclose(res, mat, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op",
        [
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
        ],
    )
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
            lambda: qml.QFT(wires=[1, 2, 3]),
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
        with pytest.raises(qml.ops.qubit.AdjointError):
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

    def test_x_decomposition(self, tol):
        """Tests that the decomposition of the PauliX is correct"""
        op = qml.PauliX(wires=0)
        res = op.decomposition(0)

        assert len(res) == 3

        assert res[0].name == "PhaseShift"

        assert res[0].wires == qml.wires.Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[1].name == "RX"
        assert res[1].wires == qml.wires.Wires([0])
        assert res[1].data[0] == np.pi

        assert res[2].name == "PhaseShift"
        assert res[2].wires == qml.wires.Wires([0])
        assert res[2].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_y_decomposition(self, tol):
        """Tests that the decomposition of the PauliY is correct"""
        op = qml.PauliY(wires=0)
        res = op.decomposition(0)

        assert len(res) == 3

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[1].name == "RY"
        assert res[1].wires == Wires([0])
        assert res[1].data[0] == np.pi

        assert res[2].name == "PhaseShift"
        assert res[2].wires == Wires([0])
        assert res[2].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_z_decomposition(self, tol):
        """Tests that the decomposition of the PauliZ is correct"""
        op = qml.PauliZ(wires=0)
        res = op.decomposition(0)

        assert len(res) == 1

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi

        decomposed_matrix = res[0].matrix
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_s_decomposition(self, tol):
        """Tests that the decomposition of the S gate is correct"""
        op = qml.S(wires=0)
        res = op.decomposition(0)

        assert len(res) == 1

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        decomposed_matrix = res[0].matrix
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_t_decomposition(self, tol):
        """Tests that the decomposition of the T gate is correct"""
        op = qml.T(wires=0)
        res = op.decomposition(0)

        assert len(res) == 1

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 4

        decomposed_matrix = res[0].matrix
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_sx_decomposition(self, tol):
        """Tests that the decomposition of the SX gate is correct"""
        op = qml.SX(wires=0)
        res = op.decomposition(0)

        assert len(res) == 4

        assert all([res[i].wires == Wires([0]) for i in range(4)])

        assert res[0].name == "RZ"
        assert res[1].name == "RY"
        assert res[2].name == "RZ"
        assert res[3].name == "PhaseShift"

        assert res[0].data[0] == np.pi / 2
        assert res[1].data[0] == np.pi / 2
        assert res[2].data[0] == -np.pi
        assert res[3].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_hadamard_decomposition(self, tol):
        """Tests that the decomposition of the Hadamard gate is correct"""
        op = qml.Hadamard(wires=0)
        res = op.decomposition(0)

        assert len(res) == 3

        assert res[0].name == "PhaseShift"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[1].name == "RX"
        assert res[1].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        assert res[2].name == "PhaseShift"
        assert res[2].wires == Wires([0])
        assert res[0].data[0] == np.pi / 2

        decomposed_matrix = np.linalg.multi_dot([i.matrix for i in reversed(res)])
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_phase_decomposition(self, tol):
        """Tests that the decomposition of the Phase gate is correct"""
        phi = 0.3
        op = qml.PhaseShift(phi, wires=0)
        res = op.decomposition(phi, 0)

        assert len(res) == 1

        assert res[0].name == "RZ"

        assert res[0].wires == Wires([0])
        assert res[0].data[0] == 0.3

        decomposed_matrix = res[0].matrix
        global_phase = (decomposed_matrix[op.matrix != 0] / op.matrix[op.matrix != 0])[0]

        assert np.allclose(decomposed_matrix, global_phase * op.matrix, atol=tol, rtol=0)

    def test_CY_decomposition(self, tol):
        """Tests that the decomposition of the CY gate is correct"""
        op = qml.CY(wires=[0, 1])
        res = op.decomposition(op.wires)

        mats = []
        for i in reversed(res):
            if len(i.wires) == 1:
                mats.append(np.kron(i.matrix, np.eye(2)))
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)
        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    @pytest.mark.parametrize("phi, theta, omega", [[0.5, 0.6, 0.7], [0.1, -0.4, 0.7], [-10, 5, -1]])
    def test_CRot_decomposition(self, tol, phi, theta, omega, monkeypatch):
        """Tests that the decomposition of the CRot gate is correct"""
        op = qml.CRot(phi, theta, omega, wires=[0, 1])
        res = op.decomposition(phi, theta, omega, op.wires)

        mats = []
        for i in reversed(res):
            if len(i.wires) == 1:
                mats.append(np.kron(np.eye(2), i.matrix))
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_ISWAP_decomposition(self, tol):
        """Tests that the decomposition of the ISWAP gate is correct"""
        op = qml.ISWAP(wires=[0, 1])
        res = op.decomposition(op.wires)

        assert len(res) == 6

        assert res[0].wires == Wires([0])
        assert res[1].wires == Wires([1])
        assert res[2].wires == Wires([0])
        assert res[3].wires == Wires([0, 1])
        assert res[4].wires == Wires([1, 0])
        assert res[5].wires == Wires([1])

        assert res[0].name == "S"
        assert res[1].name == "S"
        assert res[2].name == "Hadamard"
        assert res[3].name == "CNOT"
        assert res[4].name == "CNOT"
        assert res[5].name == "Hadamard"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([1]):
                mats.append(np.kron(np.eye(2), i.matrix))
            elif i.wires == Wires([0]):
                mats.append(np.kron(i.matrix, np.eye(2)))
            elif i.wires == Wires([1, 0]) and i.name == "CNOT":
                mats.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_isingxx_decomposition(self, tol):
        """Tests that the decomposition of the IsingXX gate is correct"""
        param = 0.1234
        op = qml.IsingXX(param, wires=[3, 2])
        res = op.decomposition(param, op.wires)

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([3])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CNOT"
        assert res[1].name == "RX"
        assert res[2].name == "CNOT"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([3]):
                # RX gate
                mats.append(np.kron(i.matrix, np.eye(2)))
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_isingzz_decomposition(self, tol):
        """Tests that the decomposition of the IsingZZ gate is correct"""
        param = 0.1234
        op = qml.IsingZZ(param, wires=[3, 2])
        res = op.decomposition(param, op.wires)

        assert len(res) == 3

        assert res[0].wires == Wires([3, 2])
        assert res[1].wires == Wires([2])
        assert res[2].wires == Wires([3, 2])

        assert res[0].name == "CNOT"
        assert res[1].name == "RZ"
        assert res[2].name == "CNOT"

        mats = []
        for i in reversed(res):
            if i.wires == Wires([2]):
                # RZ gate
                mats.append(np.kron(np.eye(2), i.matrix))
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_toffoli_decomposition(self, tol):
        """Tests that the decomposition of the Toffoli gate is correct"""
        op = qml.Toffoli(wires=[0, 1, 2])
        res = op.decomposition(op.wires)

        assert len(res) == 15

        mats = []

        for i in reversed(res):
            if i.wires == Wires([2]):
                mats.append(np.kron(np.eye(4), i.matrix))
            elif i.wires == Wires([1]):
                mats.append(np.kron(np.eye(2), np.kron(i.matrix, np.eye(2))))
            elif i.wires == Wires([0]):
                mats.append(np.kron(i.matrix, np.eye(4)))
            elif i.wires == Wires([0, 1]) and i.name == "CNOT":
                mats.append(np.kron(i.matrix, np.eye(2)))
            elif i.wires == Wires([1, 2]) and i.name == "CNOT":
                mats.append(np.kron(np.eye(2), i.matrix))
            elif i.wires == Wires([0, 2]) and i.name == "CNOT":
                mats.append(
                    np.array(
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                        ]
                    )
                )

        decomposed_matrix = np.linalg.multi_dot(mats)

        assert np.allclose(decomposed_matrix, op.matrix, atol=tol, rtol=0)

    def test_phase_shift(self, tol):
        """Test phase shift is correct"""

        # test identity for theta=0
        assert np.allclose(qml.PhaseShift._matrix(0), np.identity(2), atol=tol, rtol=0)
        assert np.allclose(qml.U1._matrix(0), np.identity(2), atol=tol, rtol=0)

        # test arbitrary phase shift
        phi = 0.5432
        expected = np.array([[1, 0], [0, np.exp(1j * phi)]])
        assert np.allclose(qml.PhaseShift._matrix(phi), expected, atol=tol, rtol=0)
        assert np.allclose(qml.U1._matrix(phi), expected, atol=tol, rtol=0)

    def test_x_rotation(self, tol):
        """Test x rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.RX._matrix(0), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
        assert np.allclose(qml.RX._matrix(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = -1j * np.array([[0, 1], [1, 0]])
        assert np.allclose(qml.RX._matrix(np.pi), expected, atol=tol, rtol=0)

    def test_y_rotation(self, tol):
        """Test y rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.RY._matrix(0), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        assert np.allclose(qml.RY._matrix(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[0, -1], [1, 0]])
        assert np.allclose(qml.RY._matrix(np.pi), expected, atol=tol, rtol=0)

    def test_z_rotation(self, tol):
        """Test z rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.RZ._matrix(0), np.identity(2), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.diag(np.exp([-1j * np.pi / 4, 1j * np.pi / 4]))
        assert np.allclose(qml.RZ._matrix(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        assert np.allclose(qml.RZ._matrix(np.pi), -1j * Z, atol=tol, rtol=0)

    def test_isingxx(self, tol):
        """Test that the IsingXX operation is correct"""
        assert np.allclose(qml.IsingXX._matrix(0), np.identity(4), atol=tol, rtol=0)

        def get_expected(theta):
            expected = np.array(np.diag([np.cos(theta / 2)] * 4), dtype=np.complex128)
            sin_coeff = -1j * np.sin(theta / 2)
            expected[3, 0] = sin_coeff
            expected[2, 1] = sin_coeff
            expected[1, 2] = sin_coeff
            expected[0, 3] = sin_coeff
            return expected

        param = np.pi / 2
        assert np.allclose(qml.IsingXX._matrix(param), get_expected(param), atol=tol, rtol=0)

        param = np.pi
        assert np.allclose(qml.IsingXX._matrix(param), get_expected(param), atol=tol, rtol=0)

    def test_isingzz(self, tol):
        """Test that the IsingZZ operation is correct"""
        assert np.allclose(qml.IsingZZ._matrix(0), np.identity(4), atol=tol, rtol=0)

        def get_expected(theta):
            neg_imag = np.exp(-1j * theta / 2)
            plus_imag = np.exp(1j * theta / 2)
            expected = np.array(
                np.diag([neg_imag, plus_imag, plus_imag, neg_imag]), dtype=np.complex128
            )
            return expected

        param = np.pi / 2
        assert np.allclose(qml.IsingZZ._matrix(param), get_expected(param), atol=tol, rtol=0)

        param = np.pi
        assert np.allclose(qml.IsingZZ._matrix(param), get_expected(param), atol=tol, rtol=0)

    def test_arbitrary_rotation(self, tol):
        """Test arbitrary single qubit rotation is correct"""

        # test identity for phi,theta,omega=0
        assert np.allclose(qml.Rot._matrix(0, 0, 0), np.identity(2), atol=tol, rtol=0)

        # expected result
        def arbitrary_rotation(x, y, z):
            """arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [np.exp(-0.5j * (x + z)) * c, -np.exp(0.5j * (x - z)) * s],
                    [np.exp(-0.5j * (x - z)) * s, np.exp(0.5j * (x + z)) * c],
                ]
            )

        a, b, c = 0.432, -0.152, 0.9234
        assert np.allclose(qml.Rot._matrix(a, b, c), arbitrary_rotation(a, b, c), atol=tol, rtol=0)

    def test_C_x_rotation(self, tol):
        """Test controlled x rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.CRX._matrix(0), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), -1j / np.sqrt(2)],
                [0, 0, -1j / np.sqrt(2), 1 / np.sqrt(2)],
            ]
        )
        assert np.allclose(qml.CRX._matrix(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]])
        assert np.allclose(qml.CRX._matrix(np.pi), expected, atol=tol, rtol=0)

    def test_C_y_rotation(self, tol):
        """Test controlled y rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.CRY._matrix(0), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            ]
        )
        assert np.allclose(qml.CRY._matrix(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        assert np.allclose(qml.CRY._matrix(np.pi), expected, atol=tol, rtol=0)

    def test_C_z_rotation(self, tol):
        """Test controlled z rotation is correct"""

        # test identity for theta=0
        assert np.allclose(qml.CRZ._matrix(0), np.identity(4), atol=tol, rtol=0)

        # test identity for theta=pi/2
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * np.pi / 4), 0],
                [0, 0, 0, np.exp(1j * np.pi / 4)],
            ]
        )
        assert np.allclose(qml.CRZ._matrix(np.pi / 2), expected, atol=tol, rtol=0)

        # test identity for theta=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1j, 0], [0, 0, 0, 1j]])
        assert np.allclose(qml.CRZ._matrix(np.pi), expected, atol=tol, rtol=0)

    def test_controlled_arbitrary_rotation(self, tol):
        """Test controlled arbitrary rotation is correct"""

        # test identity for phi,theta,omega=0
        assert np.allclose(qml.CRot._matrix(0, 0, 0), np.identity(4), atol=tol, rtol=0)

        # test identity for phi,theta,omega=pi
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
        assert np.allclose(qml.CRot._matrix(np.pi, np.pi, np.pi), expected, atol=tol, rtol=0)

        def arbitrary_Crotation(x, y, z):
            """controlled arbitrary single qubit rotation"""
            c = np.cos(y / 2)
            s = np.sin(y / 2)
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, np.exp(-0.5j * (x + z)) * c, -np.exp(0.5j * (x - z)) * s],
                    [0, 0, np.exp(-0.5j * (x - z)) * s, np.exp(0.5j * (x + z)) * c],
                ]
            )

        a, b, c = 0.432, -0.152, 0.9234
        assert np.allclose(
            qml.CRot._matrix(a, b, c), arbitrary_Crotation(a, b, c), atol=tol, rtol=0
        )

    def test_U2_gate(self, tol):
        """Test U2 gate matrix matches the documentation"""
        phi = 0.432
        lam = -0.12
        res = qml.U2._matrix(phi, lam)
        expected = np.array(
            [[1, -np.exp(1j * lam)], [np.exp(1j * phi), np.exp(1j * (phi + lam))]]
        ) / np.sqrt(2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_U3_gate(self, tol):
        """Test U3 gate matrix matches the documentation"""
        theta = 0.65
        phi = 0.432
        lam = -0.12

        res = qml.U3._matrix(theta, phi, lam)
        expected = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lam)) * np.cos(theta / 2),
                ],
            ]
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_qubit_unitary(self, tol):
        """Test that the unitary operator produces the correct output."""
        U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        out = qml.QubitUnitary(U, wires=0).matrix

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert np.allclose(out, U, atol=tol, rtol=0)

    def test_qubit_unitary_exceptions(self):
        """Tests that the unitary operator raises the proper errors."""
        U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be a square matrix"):
            qml.QubitUnitary(U[1:], wires=0).matrix

        # test non-unitary matrix
        U3 = U.copy()
        U3[0, 0] += 0.5
        with pytest.raises(ValueError, match="must be unitary"):
            qml.QubitUnitary(U3, wires=0).matrix

    @pytest.mark.parametrize(
        "U", [np.array([0]), np.array([1, 0, 0, 1]), np.array([[[1, 0], [0, 1]]])]
    )
    def test_qubit_unitary_not_matrix_exception(self, U):
        """Tests that the unitary operator raises the proper errors for arrays
        that are not two-dimensional."""

        # test non-square matrix
        with pytest.raises(ValueError, match="must be a square matrix"):
            qml.QubitUnitary(U, wires=0).matrix

    def test_iswap_eigenval(self):
        """Tests that the ISWAP eigenvalue matches the numpy eigenvalues of the ISWAP matrix"""
        op = qml.ISWAP(wires=[0, 1])
        exp = np.linalg.eigvals(op.matrix)
        res = op.eigvals
        assert np.allclose(res, exp)

    def test_swap_decomposition(self):
        """Tests the swap operator produces the correct output"""
        opr = qml.SWAP(wires=[0, 1])
        decomp = opr.decomposition([0, 1])

        mat = []
        for op in reversed(decomp):
            if isinstance(op, qml.CNOT) and op.wires.tolist() == [0, 1]:
                mat.append(CNOT)
            elif isinstance(op, qml.CNOT) and op.wires.tolist() == [1, 0]:
                mat.append(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]))

        decomposed_matrix = np.linalg.multi_dot(mat)

        assert np.allclose(decomposed_matrix, opr.matrix)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    @pytest.mark.parametrize("cphase_op", [qml.ControlledPhaseShift, qml.CPhase])
    def test_controlled_phase_shift_matrix_and_eigvals(self, phi, cphase_op):
        """Tests that the ControlledPhaseShift and CPhase operation calculates the correct matrix and
        eigenvalues"""
        op = cphase_op(phi, wires=[0, 1])
        res = op.matrix
        exp = ControlledPhaseShift(phi)
        assert np.allclose(res, exp)

        res = op.eigvals
        assert np.allclose(res, np.diag(exp))

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    @pytest.mark.parametrize("cphase_op", [qml.ControlledPhaseShift, qml.CPhase])
    def test_controlled_phase_shift_decomp(self, phi, cphase_op):
        """Tests that the ControlledPhaseShift and CPhase operation
        calculates the correct decomposition"""
        op = cphase_op(phi, wires=[0, 2])
        decomp = op.decomposition(phi, wires=[0, 2])

        mats = []
        for i in reversed(decomp):
            if i.wires.tolist() == [0]:
                mats.append(np.kron(i.matrix, np.eye(4)))
            elif i.wires.tolist() == [1]:
                mats.append(np.kron(np.eye(2), np.kron(i.matrix, np.eye(2))))
            elif i.wires.tolist() == [2]:
                mats.append(np.kron(np.eye(4), i.matrix))
            elif isinstance(i, qml.CNOT) and i.wires.tolist() == [0, 1]:
                mats.append(np.kron(i.matrix, np.eye(2)))
            elif isinstance(i, qml.CNOT) and i.wires.tolist() == [0, 2]:
                mats.append(
                    np.array(
                        [
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                        ]
                    )
                )

        decomposed_matrix = np.linalg.multi_dot(mats)
        lam = np.exp(1j * phi)
        exp = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, lam, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, lam],
            ]
        )

        assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    def test_single_excitation_plus_decomp(self, phi):
        """Tests that the SingleExcitationPlus operation calculates the correct decomposition.

        Need to consider the matrix of CRY separately, as the control is wire 1
        and the target is wire 0 in the decomposition. (Not applicable for
        ControlledPhase as it has the same matrix representation regardless of the
        control and target wires.)"""
        decomp = qml.SingleExcitationPlus.decomposition(phi, wires=[0, 1])

        mats = []
        for i in reversed(decomp):
            if i.wires.tolist() == [0]:
                mats.append(np.kron(i.matrix, np.eye(2)))
            elif i.wires.tolist() == [1]:
                mats.append(np.kron(np.eye(2), i.matrix))
            elif i.wires.tolist() == [1, 0] and isinstance(i, qml.CRY):
                new_mat = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(phi / 2), 0, -np.sin(phi / 2)],
                        [0, 0, 1, 0],
                        [0, np.sin(phi / 2), 0, np.cos(phi / 2)],
                    ]
                )

                mats.append(new_mat)
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)
        exp = SingleExcitationPlus(phi)

        assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    def test_single_excitation_minus_decomp(self, phi):
        """Tests that the SingleExcitationMinus operation calculates the correct decomposition.

        Need to consider the matrix of CRY separately, as the control is wire 1
        and the target is wire 0 in the decomposition. (Not applicable for
        ControlledPhase as it has the same matrix representation regardless of the
        control and target wires.)"""
        decomp = qml.SingleExcitationMinus.decomposition(phi, wires=[0, 1])

        mats = []
        for i in reversed(decomp):
            if i.wires.tolist() == [0]:
                mats.append(np.kron(i.matrix, np.eye(2)))
            elif i.wires.tolist() == [1]:
                mats.append(np.kron(np.eye(2), i.matrix))
            elif i.wires.tolist() == [1, 0] and isinstance(i, qml.CRY):
                new_mat = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(phi / 2), 0, -np.sin(phi / 2)],
                        [0, 0, 1, 0],
                        [0, np.sin(phi / 2), 0, np.cos(phi / 2)],
                    ]
                )

                mats.append(new_mat)
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)
        exp = SingleExcitationMinus(phi)

        assert np.allclose(decomposed_matrix, exp)


class TestSingleExcitation:
    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_matrix(self, phi):
        """Tests that the SingleExcitation operation calculates the correct matrix"""
        op = qml.SingleExcitation(phi, wires=[0, 1])
        res = op.matrix
        exp = SingleExcitation(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_decomp(self, phi):
        """Tests that the SingleExcitation operation calculates the correct decomposition.

        Need to consider the matrix of CRY separately, as the control is wire 1
        and the target is wire 0 in the decomposition."""
        decomp = qml.SingleExcitation.decomposition(phi, wires=[0, 1])

        mats = []
        for i in reversed(decomp):
            if i.wires.tolist() == [1, 0] and isinstance(i, qml.CRY):
                new_mat = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(phi / 2), 0, -np.sin(phi / 2)],
                        [0, 0, 1, 0],
                        [0, np.sin(phi / 2), 0, np.cos(phi / 2)],
                    ]
                )
                mats.append(new_mat)
            else:
                mats.append(i.matrix)

        decomposed_matrix = np.linalg.multi_dot(mats)
        exp = SingleExcitation(phi)

        assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_generator(self, phi):
        """Tests that the SingleExcitation operation calculates the correct generator"""
        op = qml.SingleExcitation(phi, wires=[0, 1])
        g, a = op.generator
        res = expm(1j * a * g * phi)
        exp = SingleExcitation(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_plus_matrix(self, phi):
        """Tests that the SingleExcitationPlus operation calculates the correct matrix"""
        op = qml.SingleExcitationPlus(phi, wires=[0, 1])
        res = op.matrix
        exp = SingleExcitationPlus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_plus_generator(self, phi):
        """Tests that the SingleExcitationPlus operation calculates the correct generator"""
        op = qml.SingleExcitationPlus(phi, wires=[0, 1])
        g, a = op.generator
        res = expm(1j * a * g * phi)
        exp = SingleExcitationPlus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_minus_matrix(self, phi):
        """Tests that the SingleExcitationMinus operation calculates the correct matrix"""
        op = qml.SingleExcitationMinus(phi, wires=[0, 1])
        res = op.matrix
        exp = SingleExcitationMinus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_single_excitation_minus_generator(self, phi):
        """Tests that the SingleExcitationMinus operation calculates the correct generator"""
        op = qml.SingleExcitationMinus(phi, wires=[0, 1])
        g, a = op.generator
        res = expm(1j * a * g * phi)
        exp = SingleExcitationMinus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize(
        "excitation", [qml.SingleExcitation, qml.SingleExcitationPlus, qml.SingleExcitationMinus]
    )
    def test_autograd(self, excitation):
        """Tests that operations are computed correctly using the
        autograd interface"""

        pytest.importorskip("autograd")
        dev = qml.device("default.qubit.autograd", wires=2)
        state = np.array([0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0])

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.SingleExcitation, -0.1),
            (qml.SingleExcitationPlus, 0.2),
            (qml.SingleExcitationMinus, np.pi / 4),
        ],
    )
    def test_autograd_grad(self, diff_method, excitation, phi):
        """Tests that gradients are computed correctly using the
        autograd interface"""

        pytest.importorskip("autograd")
        dev = qml.device("default.qubit.autograd", wires=2)

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(qml.grad(circuit)(phi), np.sin(phi))

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.SingleExcitation, -0.1),
            (qml.SingleExcitationPlus, 0.2),
            (qml.SingleExcitationMinus, np.pi / 4),
        ],
    )
    def test_tf(self, excitation, phi, diff_method):
        """Tests that gradients and operations are computed correctly using the
        tensorflow interface"""

        tf = pytest.importorskip("tensorflow")
        dev = qml.device("default.qubit.tf", wires=2)

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        phi_t = tf.Variable(phi, dtype=tf.float64)
        with tf.GradientTape() as tape:
            res = circuit(phi_t)

        grad = tape.gradient(res, phi_t)
        assert np.allclose(grad, np.sin(phi))

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.SingleExcitation, -0.1),
            (qml.SingleExcitationPlus, 0.2),
            (qml.SingleExcitationMinus, np.pi / 4),
        ],
    )
    def test_jax(self, excitation, phi, diff_method):
        """Tests that gradients and operations are computed correctly using the
        jax interface"""

        if diff_method == "parameter-shift":
            pytest.skip("JAX support for the parameter-shift method is still TBD")

        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(phi):
            qml.PauliX(wires=0)
            excitation(phi, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(jax.grad(circuit)(phi), np.sin(phi))


PAULI_ROT_PARAMETRIC_MATRIX_TEST_DATA = [
    (
        "XY",
        lambda theta: np.array(
            [
                [np.cos(theta / 2), 0, 0, -np.sin(theta / 2)],
                [0, np.cos(theta / 2), np.sin(theta / 2), 0],
                [0, -np.sin(theta / 2), np.cos(theta / 2), 0],
                [np.sin(theta / 2), 0, 0, np.cos(theta / 2)],
            ],
            dtype=complex,
        ),
    ),
    (
        "ZZ",
        lambda theta: np.diag(
            [
                np.exp(-1j * theta / 2),
                np.exp(1j * theta / 2),
                np.exp(1j * theta / 2),
                np.exp(-1j * theta / 2),
            ],
        ),
    ),
    (
        "XI",
        lambda theta: np.array(
            [
                [np.cos(theta / 2), 0, -1j * np.sin(theta / 2), 0],
                [0, np.cos(theta / 2), 0, -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), 0, np.cos(theta / 2), 0],
                [0, -1j * np.sin(theta / 2), 0, np.cos(theta / 2)],
            ],
        ),
    ),
    ("X", qml.RX._matrix),
    ("Y", qml.RY._matrix),
    ("Z", qml.RZ._matrix),
]

PAULI_ROT_MATRIX_TEST_DATA = [
    (
        np.pi,
        "XIZ",
        np.array(
            [
                [0, 0, 0, 0, -1j, 0, 0, 0],
                [0, 0, 0, 0, 0, 1j, 0, 0],
                [0, 0, 0, 0, 0, 0, -1j, 0],
                [0, 0, 0, 0, 0, 0, 0, 1j],
                [-1j, 0, 0, 0, 0, 0, 0, 0],
                [0, 1j, 0, 0, 0, 0, 0, 0],
                [0, 0, -1j, 0, 0, 0, 0, 0],
                [0, 0, 0, 1j, 0, 0, 0, 0],
            ]
        ),
    ),
    (
        np.pi / 3,
        "XYZ",
        np.array(
            [
                [np.sqrt(3) / 2, 0, 0, 0, 0, 0, -(1 / 2), 0],
                [0, np.sqrt(3) / 2, 0, 0, 0, 0, 0, 1 / 2],
                [0, 0, np.sqrt(3) / 2, 0, 1 / 2, 0, 0, 0],
                [0, 0, 0, np.sqrt(3) / 2, 0, -(1 / 2), 0, 0],
                [0, 0, -(1 / 2), 0, np.sqrt(3) / 2, 0, 0, 0],
                [0, 0, 0, 1 / 2, 0, np.sqrt(3) / 2, 0, 0],
                [1 / 2, 0, 0, 0, 0, 0, np.sqrt(3) / 2, 0],
                [0, -(1 / 2), 0, 0, 0, 0, 0, np.sqrt(3) / 2],
            ]
        ),
    ),
]


class TestDoubleExcitation:
    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_matrix(self, phi):
        """Tests that the DoubleExcitation operation calculates the correct matrix"""
        op = qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
        res = op.matrix
        exp = DoubleExcitation(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_decomp(self, phi):
        """Tests that the DoubleExcitation operation calculates the correct decomposition"""
        op = qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
        decomp = op.decomposition(phi, wires=[0, 1, 2, 3])

        mats = [m.matrix for m in decomp]
        decomposed_matrix = mats[0] @ mats[1]
        exp = DoubleExcitation(phi)

        assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_generator(self, phi):
        """Tests that the DoubleExcitation operation calculates the correct generator"""
        op = qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
        g, a = op.generator

        res = expm(1j * a * g * phi)
        exp = DoubleExcitation(phi)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, 0.5])
    def test_double_excitation_decomp(self, phi):
        """Tests that the DoubleExcitation operation calculates the correct decomposition.

        The decomposition has already been expressed in terms of single-qubit rotations
        and CNOTs. For each term in the decomposition we need to construct the appropriate
        four-qubit tensor product matrix and then multiply them together.
        """
        decomp = qml.DoubleExcitation.decomposition(phi, wires=[0, 1, 2, 3])

        from functools import reduce

        # To compute the matrix for CX on an arbitrary number of qubits, use the fact that
        # CU  = |0><0| \otimes I + |1><1| \otimes U
        def cnot_four_qubits(wires):
            proj_0_term = [StateZeroProjector if idx == wires[0] else np.eye(2) for idx in range(4)]

            proj_1_term = [np.eye(2) for idx in range(4)]
            proj_1_term[wires[0]] = StateOneProjector
            proj_1_term[wires[1]] = X

            proj_0_kron = reduce(np.kron, proj_0_term)
            proj_1_kron = reduce(np.kron, proj_1_term)

            return proj_0_kron + proj_1_kron

        # Inserts a single-qubit matrix into a four-qubit matrix at the right place
        def single_mat_four_qubits(mat, wire):
            individual_mats = [mat if idx == wire else np.eye(2) for idx in range(4)]
            return reduce(np.kron, individual_mats)

        mats = []
        for i in reversed(decomp):
            # Single-qubit gate
            if len(i.wires.tolist()) == 1:
                mat = single_mat_four_qubits(i.matrix, i.wires.tolist()[0])
                mats.append(mat)
            # Two-qubit gate
            else:
                mat = cnot_four_qubits(i.wires.tolist())
                mats.append(mat)

        decomposed_matrix = np.linalg.multi_dot(mats)
        exp = DoubleExcitation(phi)

        assert np.allclose(decomposed_matrix, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_plus_matrix(self, phi):
        """Tests that the DoubleExcitationPlus operation calculates the correct matrix"""
        op = qml.DoubleExcitationPlus(phi, wires=[0, 1, 2, 3])
        res = op.matrix
        exp = DoubleExcitationPlus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_plus_generator(self, phi):
        """Tests that the DoubleExcitationPlus operation calculates the correct generator"""
        op = qml.DoubleExcitationPlus(phi, wires=[0, 1, 2, 3])
        g, a = op.generator

        res = expm(1j * a * g * phi)
        exp = DoubleExcitationPlus(phi)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_minus_matrix(self, phi):
        """Tests that the DoubleExcitationMinus operation calculates the correct matrix"""
        op = qml.DoubleExcitationMinus(phi, wires=[0, 1, 2, 3])
        res = op.matrix
        exp = DoubleExcitationMinus(phi)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("phi", [-0.1, 0.2, np.pi / 4])
    def test_double_excitation_minus_generator(self, phi):
        """Tests that the DoubleExcitationMinus operation calculates the correct generator"""
        op = qml.DoubleExcitationMinus(phi, wires=[0, 1, 2, 3])
        g, a = op.generator

        res = expm(1j * a * g * phi)
        exp = DoubleExcitationMinus(phi)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize(
        "excitation", [qml.DoubleExcitation, qml.DoubleExcitationPlus, qml.DoubleExcitationMinus]
    )
    def test_autograd(self, excitation):
        """Tests that operations are computed correctly using the
        autograd interface"""

        pytest.importorskip("autograd")

        dev = qml.device("default.qubit.autograd", wires=4)
        state = np.array(
            [0, 0, 0, -1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]
        )

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.parametrize(
        "excitation", [qml.DoubleExcitation, qml.DoubleExcitationPlus, qml.DoubleExcitationMinus]
    )
    def test_tf(self, excitation):
        """Tests that operations are computed correctly using the
        tensorflow interface"""

        pytest.importorskip("tensorflow")

        dev = qml.device("default.qubit.tf", wires=4)
        state = np.array(
            [0, 0, 0, -1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]
        )

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.parametrize(
        "excitation", [qml.DoubleExcitation, qml.DoubleExcitationPlus, qml.DoubleExcitationMinus]
    )
    def test_jax(self, excitation):
        """Tests that operations are computed correctly using the
        jax interface"""

        pytest.importorskip("jax")

        dev = qml.device("default.qubit.jax", wires=4)
        state = np.array(
            [0, 0, 0, -1 / np.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 0, 0]
        )

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.state()

        assert np.allclose(state, circuit(np.pi / 2))

    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.DoubleExcitation, -0.1),
            (qml.DoubleExcitationPlus, 0.2),
            (qml.DoubleExcitationMinus, np.pi / 4),
        ],
    )
    def test_autograd_grad(self, excitation, phi):
        """Tests that gradients are computed correctly using the
        autograd interface"""

        pytest.importorskip("autograd")

        dev = qml.device("default.qubit.autograd", wires=4)

        @qml.qnode(dev)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])

            return qml.expval(qml.PauliZ(0))

        assert np.allclose(qml.grad(circuit)(phi), np.sin(phi))

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.DoubleExcitation, -0.1),
            (qml.DoubleExcitationPlus, 0.2),
            (qml.DoubleExcitationMinus, np.pi / 4),
        ],
    )
    def test_tf_grad(self, excitation, phi, diff_method):
        """Tests that gradients are computed correctly using the
        tensorflow interface"""

        tf = pytest.importorskip("tensorflow")
        dev = qml.device("default.qubit.tf", wires=4)

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        phi_t = tf.Variable(phi, dtype=tf.float64)
        with tf.GradientTape() as tape:
            res = circuit(phi_t)

        grad = tape.gradient(res, phi_t)
        assert np.allclose(grad, np.sin(phi))

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop"])
    @pytest.mark.parametrize(
        ("excitation", "phi"),
        [
            (qml.DoubleExcitation, -0.1),
            (qml.DoubleExcitationPlus, 0.2),
            (qml.DoubleExcitationMinus, np.pi / 4),
        ],
    )
    def test_jax_grad(self, excitation, phi, diff_method):
        """Tests that gradients and operations are computed correctly using the
        jax interface"""

        if diff_method == "parameter-shift":
            pytest.skip("JAX support for the parameter-shift method is still TBD")

        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit.jax", wires=4)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def circuit(phi):
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            excitation(phi, wires=[0, 1, 2, 3])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(jax.grad(circuit)(phi), np.sin(phi))


class TestPauliRot:
    """Test the PauliRot operation."""

    @pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize(
        "pauli_word,expected_matrix",
        PAULI_ROT_PARAMETRIC_MATRIX_TEST_DATA,
    )
    def test_PauliRot_matrix_parametric(self, theta, pauli_word, expected_matrix, tol):
        """Test parametrically that the PauliRot matrix is correct."""

        res = qml.PauliRot._matrix(theta, pauli_word)
        expected = expected_matrix(theta)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "theta,pauli_word,expected_matrix",
        PAULI_ROT_MATRIX_TEST_DATA,
    )
    def test_PauliRot_matrix(self, theta, pauli_word, expected_matrix, tol):
        """Test non-parametrically that the PauliRot matrix is correct."""

        res = qml.PauliRot._matrix(theta, pauli_word)
        expected = expected_matrix

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "theta,pauli_word,compressed_pauli_word,wires,compressed_wires",
        [
            (np.pi, "XIZ", "XZ", [0, 1, 2], [0, 2]),
            (np.pi / 3, "XIYIZI", "XYZ", [0, 1, 2, 3, 4, 5], [0, 2, 4]),
            (np.pi / 7, "IXI", "X", [0, 1, 2], [1]),
            (np.pi / 9, "IIIIIZI", "Z", [0, 1, 2, 3, 4, 5, 6], [5]),
            (np.pi / 11, "XYZIII", "XYZ", [0, 1, 2, 3, 4, 5], [0, 1, 2]),
            (np.pi / 11, "IIIXYZ", "XYZ", [0, 1, 2, 3, 4, 5], [3, 4, 5]),
        ],
    )
    def test_PauliRot_matrix_identity(
        self, theta, pauli_word, compressed_pauli_word, wires, compressed_wires, tol
    ):
        """Test PauliRot matrix correctly accounts for identities."""

        res = qml.PauliRot._matrix(theta, pauli_word)
        expected = qml.utils.expand(
            qml.PauliRot._matrix(theta, compressed_pauli_word), compressed_wires, wires
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_PauliRot_wire_as_int(self):
        """Test that passing a single wire as an integer works."""

        theta = 0.4
        op = qml.PauliRot(theta, "Z", wires=0)
        decomp_ops = op.decomposition(theta, "Z", wires=0)

        assert np.allclose(op.eigvals, np.array([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]))
        assert np.allclose(op.matrix, np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]))

        assert len(decomp_ops) == 1

        assert decomp_ops[0].name == "MultiRZ"

        assert decomp_ops[0].wires == Wires([0])
        assert decomp_ops[0].data[0] == theta

    def test_PauliRot_all_Identity(self):
        """Test handling of the all-identity Pauli."""

        theta = 0.4
        op = qml.PauliRot(theta, "II", wires=[0, 1])
        decomp_ops = op.decomposition(theta, "II", wires=[0, 1])

        assert np.allclose(op.eigvals, np.exp(-1j * theta / 2) * np.ones(4))
        assert np.allclose(op.matrix / op.matrix[0, 0], np.eye(4))

        assert len(decomp_ops) == 0

    def test_PauliRot_decomposition_Identity(self):
        """Test that decomposing the all-identity Pauli has no effect."""

        theta = 0.4
        op = qml.PauliRot(theta, "II", wires=[0, 1])
        decomp_ops = op.decomposition(theta, "II", wires=[0, 1])

        assert len(decomp_ops) == 0

    def test_PauliRot_decomposition_ZZ(self):
        """Test that the decomposition for a ZZ rotation is correct."""

        theta = 0.4
        op = qml.PauliRot(theta, "ZZ", wires=[0, 1])
        decomp_ops = op.decomposition(theta, "ZZ", wires=[0, 1])

        assert len(decomp_ops) == 1

        assert decomp_ops[0].name == "MultiRZ"

        assert decomp_ops[0].wires == Wires([0, 1])
        assert decomp_ops[0].data[0] == theta

    def test_PauliRot_decomposition_XY(self):
        """Test that the decomposition for a XY rotation is correct."""

        theta = 0.4
        op = qml.PauliRot(theta, "XY", wires=[0, 1])
        decomp_ops = op.decomposition(theta, "XY", wires=[0, 1])

        assert len(decomp_ops) == 5

        assert decomp_ops[0].name == "Hadamard"
        assert decomp_ops[0].wires == Wires([0])

        assert decomp_ops[1].name == "RX"

        assert decomp_ops[1].wires == Wires([1])
        assert decomp_ops[1].data[0] == np.pi / 2

        assert decomp_ops[2].name == "MultiRZ"
        assert decomp_ops[2].wires == Wires([0, 1])
        assert decomp_ops[2].data[0] == theta

        assert decomp_ops[3].name == "Hadamard"
        assert decomp_ops[3].wires == Wires([0])

        assert decomp_ops[4].name == "RX"

        assert decomp_ops[4].wires == Wires([1])
        assert decomp_ops[4].data[0] == -np.pi / 2

    def test_PauliRot_decomposition_XIYZ(self):
        """Test that the decomposition for a XIYZ rotation is correct."""

        theta = 0.4
        op = qml.PauliRot(theta, "XIYZ", wires=[0, 1, 2, 3])
        decomp_ops = op.decomposition(theta, "XIYZ", wires=[0, 1, 2, 3])

        assert len(decomp_ops) == 5

        assert decomp_ops[0].name == "Hadamard"
        assert decomp_ops[0].wires == Wires([0])

        assert decomp_ops[1].name == "RX"

        assert decomp_ops[1].wires == Wires([2])
        assert decomp_ops[1].data[0] == np.pi / 2

        assert decomp_ops[2].name == "MultiRZ"
        assert decomp_ops[2].wires == Wires([0, 2, 3])
        assert decomp_ops[2].data[0] == theta

        assert decomp_ops[3].name == "Hadamard"
        assert decomp_ops[3].wires == Wires([0])

        assert decomp_ops[4].name == "RX"

        assert decomp_ops[4].wires == Wires([2])
        assert decomp_ops[4].data[0] == -np.pi / 2

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize("pauli_word", ["XX", "ZZ"])
    def test_differentiability(self, angle, pauli_word, tol):
        """Test that differentiation of PauliRot works."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.PauliRot(theta, pauli_word, wires=[0, 1])

            return qml.expval(qml.PauliZ(0))

        res = circuit(angle)
        gradient = np.squeeze(qml.grad(circuit)(angle))

        assert gradient == pytest.approx(
            0.5 * (circuit(angle + np.pi / 2) - circuit(angle - np.pi / 2)), abs=tol
        )

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_decomposition_integration(self, angle, tol):
        """Test that the decompositon of PauliRot yields the same results."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.PauliRot(theta, "XX", wires=[0, 1])

            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def decomp_circuit(theta):
            qml.PauliRot.decomposition(theta, "XX", wires=[0, 1])

            return qml.expval(qml.PauliZ(0))

        assert circuit(angle) == pytest.approx(decomp_circuit(angle), abs=tol)
        assert np.squeeze(qml.grad(circuit)(angle)) == pytest.approx(
            np.squeeze(qml.grad(decomp_circuit)(angle)), abs=tol
        )

    def test_matrix_incorrect_pauli_word_error(self):
        """Test that _matrix throws an error if a wrong Pauli word is supplied."""

        with pytest.raises(
            ValueError,
            match='The given Pauli word ".*" contains characters that are not allowed.'
            " Allowed characters are I, X, Y and Z",
        ):
            qml.PauliRot._matrix(0.3, "IXYZV")

    def test_init_incorrect_pauli_word_error(self):
        """Test that __init__ throws an error if a wrong Pauli word is supplied."""

        with pytest.raises(
            ValueError,
            match='The given Pauli word ".*" contains characters that are not allowed.'
            " Allowed characters are I, X, Y and Z",
        ):
            qml.PauliRot(0.3, "IXYZV", wires=[0, 1, 2, 3, 4])

    @pytest.mark.parametrize(
        "pauli_word,wires",
        [
            ("XYZ", [0, 1]),
            ("XYZ", [0, 1, 2, 3]),
        ],
    )
    def test_init_incorrect_pauli_word_length_error(self, pauli_word, wires):
        """Test that __init__ throws an error if a Pauli word of wrong length is supplied."""

        with pytest.raises(
            ValueError,
            match="The given Pauli word has length .*, length .* was expected for wires .*",
        ):
            qml.PauliRot(0.3, pauli_word, wires=wires)

    @pytest.mark.parametrize(
        "pauli_word",
        [
            ("XIZ"),
            ("IIII"),
            ("XIYIZI"),
            ("IXI"),
            ("IIIIIZI"),
            ("XYZIII"),
            ("IIIXYZ"),
        ],
    )
    def test_multirz_generator(self, pauli_word):
        """Test that the generator of the MultiRZ gate is correct."""
        op = qml.PauliRot(0.3, pauli_word, wires=range(len(pauli_word)))
        gen = op.generator

        if pauli_word[0] == "I":
            # this is the identity
            expected_gen = qml.Identity(wires=0)
        else:
            expected_gen = getattr(qml, "Pauli{}".format(pauli_word[0]))(wires=0)

        for i, pauli in enumerate(pauli_word[1:]):
            i += 1
            if pauli == "I":
                expected_gen = expected_gen @ qml.Identity(wires=i)
            else:
                expected_gen = expected_gen @ getattr(qml, "Pauli{}".format(pauli))(wires=i)

        expected_gen_mat = expected_gen.matrix

        assert np.allclose(gen[0], expected_gen_mat)
        assert gen[1] == -0.5


class TestMultiRZ:
    """Test the MultiRZ operation."""

    @pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize(
        "wires,expected_matrix",
        [
            ([0], qml.RZ._matrix),
            (
                [0, 1],
                lambda theta: np.diag(
                    np.exp(1j * np.array([-1, 1, 1, -1]) * theta / 2),
                ),
            ),
            (
                [0, 1, 2],
                lambda theta: np.diag(
                    np.exp(1j * np.array([-1, 1, 1, -1, 1, -1, -1, 1]) * theta / 2),
                ),
            ),
        ],
    )
    def test_MultiRZ_matrix_parametric(self, theta, wires, expected_matrix, tol):
        """Test parametrically that the MultiRZ matrix is correct."""

        res = qml.MultiRZ._matrix(theta, len(wires))
        expected = expected_matrix(theta)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_MultiRZ_decomposition_ZZ(self):
        """Test that the decomposition for a ZZ rotation is correct."""

        theta = 0.4
        op = qml.MultiRZ(theta, wires=[0, 1])
        decomp_ops = op.decomposition(theta, wires=[0, 1])

        assert decomp_ops[0].name == "CNOT"
        assert decomp_ops[0].wires == Wires([1, 0])

        assert decomp_ops[1].name == "RZ"

        assert decomp_ops[1].wires == Wires([0])
        assert decomp_ops[1].data[0] == theta

        assert decomp_ops[2].name == "CNOT"
        assert decomp_ops[2].wires == Wires([1, 0])

    def test_MultiRZ_decomposition_ZZZ(self):
        """Test that the decomposition for a ZZZ rotation is correct."""

        theta = 0.4
        op = qml.MultiRZ(theta, wires=[0, 2, 3])
        decomp_ops = op.decomposition(theta, wires=[0, 2, 3])

        assert decomp_ops[0].name == "CNOT"
        assert decomp_ops[0].wires == Wires([3, 2])

        assert decomp_ops[1].name == "CNOT"
        assert decomp_ops[1].wires == Wires([2, 0])

        assert decomp_ops[2].name == "RZ"

        assert decomp_ops[2].wires == Wires([0])
        assert decomp_ops[2].data[0] == theta

        assert decomp_ops[3].name == "CNOT"
        assert decomp_ops[3].wires == Wires([2, 0])

        assert decomp_ops[4].name == "CNOT"
        assert decomp_ops[4].wires == Wires([3, 2])

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_differentiability(self, angle, tol):
        """Test that differentiation of MultiRZ works."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(0)
            qml.MultiRZ(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        res = circuit(angle)
        gradient = np.squeeze(qml.grad(circuit)(angle))

        assert gradient == pytest.approx(
            0.5 * (circuit(angle + np.pi / 2) - circuit(angle - np.pi / 2)), abs=tol
        )

    @pytest.mark.parametrize("angle", np.linspace(0, 2 * np.pi, 7))
    def test_decomposition_integration(self, angle, tol):
        """Test that the decompositon of MultiRZ yields the same results."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(0)
            qml.MultiRZ(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        @qml.qnode(dev)
        def decomp_circuit(theta):
            qml.Hadamard(0)
            qml.MultiRZ.decomposition(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        assert circuit(angle) == pytest.approx(decomp_circuit(angle), abs=tol)
        assert np.squeeze(qml.jacobian(circuit)(angle)) == pytest.approx(
            np.squeeze(qml.jacobian(decomp_circuit)(angle)), abs=tol
        )

    @pytest.mark.parametrize("qubits", range(3, 6))
    def test_multirz_generator(self, qubits, mocker):
        """Test that the generator of the MultiRZ gate is correct."""
        op = qml.MultiRZ(0.3, wires=range(qubits))
        gen = op.generator

        expected_gen = qml.PauliZ(wires=0)
        for i in range(1, qubits):
            expected_gen = expected_gen @ qml.PauliZ(wires=i)

        expected_gen_mat = expected_gen.matrix

        assert np.allclose(gen[0], expected_gen_mat)
        assert gen[1] == -0.5

        spy = mocker.spy(qml.utils, "pauli_eigs")

        op.generator
        spy.assert_not_called()


class TestDiagonalQubitUnitary:
    """Test the DiagonalQubitUnitary operation."""

    def test_decomposition(self):
        """Test that DiagonalQubitUnitary falls back to QubitUnitary."""
        D = np.array([1j, 1, 1, -1, -1j, 1j, 1, -1])

        decomp = qml.DiagonalQubitUnitary.decomposition(D, [0, 1, 2])

        assert decomp[0].name == "QubitUnitary"
        assert decomp[0].wires == Wires([0, 1, 2])
        assert np.allclose(decomp[0].data[0], np.diag(D))


def test_identity_eigvals(tol):
    """Test identity eigenvalues are correct"""
    res = qml.Identity._eigvals()
    expected = np.array([1, 1])
    assert np.allclose(res, expected, atol=tol, rtol=0)


class TestControlledQubitUnitary:
    """Tests for the ControlledQubitUnitary operation"""

    X = np.array([[0, 1], [1, 0]])

    def test_matrix(self):
        """Test if ControlledQubitUnitary returns the correct matrix for a control-control-X
        (Toffoli) gate"""
        mat = qml.ControlledQubitUnitary(X, control_wires=[0, 1], wires=2).matrix
        mat2 = qml.Toffoli(wires=[0, 1, 2]).matrix
        assert np.allclose(mat, mat2)

    def test_no_control(self):
        """Test if ControlledQubitUnitary raises an error if control wires are not specified"""
        with pytest.raises(ValueError, match="Must specify control wires"):
            qml.ControlledQubitUnitary(X, wires=2)

    def test_shared_control(self):
        """Test if ControlledQubitUnitary raises an error if control wires are shared with wires"""
        with pytest.raises(ValueError, match="The control wires must be different from the wires"):
            qml.ControlledQubitUnitary(X, control_wires=[0, 2], wires=2)

    def test_wrong_shape(self):
        """Test if ControlledQubitUnitary raises a ValueError if a unitary of shape inconsistent
        with wires is provided"""
        with pytest.raises(ValueError, match=r"Input unitary must be of shape \(2, 2\)"):
            qml.ControlledQubitUnitary(np.eye(4), control_wires=[0, 1], wires=2)

    @pytest.mark.parametrize("target_wire", range(3))
    def test_toffoli(self, target_wire):
        """Test if ControlledQubitUnitary acts like a Toffoli gate when the input unitary is a
        single-qubit X. This test allows the target wire to be any of the three wires."""
        control_wires = list(range(3))
        del control_wires[target_wire]

        # pick some random unitaries (with a fixed seed) to make the circuit less trivial
        U1 = unitary_group.rvs(8, random_state=1)
        U2 = unitary_group.rvs(8, random_state=2)

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def f1():
            qml.QubitUnitary(U1, wires=range(3))
            qml.ControlledQubitUnitary(X, control_wires=control_wires, wires=target_wire)
            qml.QubitUnitary(U2, wires=range(3))
            return qml.state()

        @qml.qnode(dev)
        def f2():
            qml.QubitUnitary(U1, wires=range(3))
            qml.Toffoli(wires=control_wires + [target_wire])
            qml.QubitUnitary(U2, wires=range(3))
            return qml.state()

        state_1 = f1()
        state_2 = f2()

        assert np.allclose(state_1, state_2)

    def test_arbitrary_multiqubit(self):
        """Test if ControlledQubitUnitary applies correctly for a 2-qubit unitary with 2-qubit
        control, where the control and target wires are not ordered."""
        control_wires = [1, 3]
        target_wires = [2, 0]

        # pick some random unitaries (with a fixed seed) to make the circuit less trivial
        U1 = unitary_group.rvs(16, random_state=1)
        U2 = unitary_group.rvs(16, random_state=2)

        # the two-qubit unitary
        U = unitary_group.rvs(4, random_state=3)

        # the 4-qubit representation of the unitary if the control wires were [0, 1] and the target
        # wires were [2, 3]
        U_matrix = np.eye(16, dtype=np.complex128)
        U_matrix[12:16, 12:16] = U

        # We now need to swap wires so that the control wires are [1, 3] and the target wires are
        # [2, 0]
        swap = qml.SWAP.matrix

        # initial wire permutation: 0123
        # target wire permutation: 1302
        swap1 = np.kron(swap, np.eye(4))  # -> 1023
        swap2 = np.kron(np.eye(4), swap)  # -> 1032
        swap3 = np.kron(np.kron(np.eye(2), swap), np.eye(2))  # -> 1302
        swap4 = np.kron(np.eye(4), swap)  # -> 1320

        all_swap = swap4 @ swap3 @ swap2 @ swap1
        U_matrix = all_swap.T @ U_matrix @ all_swap

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def f1():
            qml.QubitUnitary(U1, wires=range(4))
            qml.ControlledQubitUnitary(U, control_wires=control_wires, wires=target_wires)
            qml.QubitUnitary(U2, wires=range(4))
            return qml.state()

        @qml.qnode(dev)
        def f2():
            qml.QubitUnitary(U1, wires=range(4))
            qml.QubitUnitary(U_matrix, wires=range(4))
            qml.QubitUnitary(U2, wires=range(4))
            return qml.state()

        state_1 = f1()
        state_2 = f2()

        assert np.allclose(state_1, state_2)

    @pytest.mark.parametrize(
        "control_wires,wires,control_values,expected_error_message",
        [
            ([0, 1], 2, "ab", "String of control values can contain only '0' or '1'."),
            ([0, 1], 2, "011", "Length of control bit string must equal number of control wires."),
            ([0, 1], 2, [0, 1], "Alternative control values must be passed as a binary string."),
        ],
    )
    def test_invalid_mixed_polarity_controls(
        self, control_wires, wires, control_values, expected_error_message
    ):
        """Test if ControlledQubitUnitary properly handles invalid mixed-polarity
        control values."""
        target_wires = Wires(wires)

        with pytest.raises(ValueError, match=expected_error_message):
            qml.ControlledQubitUnitary(
                X, control_wires=control_wires, wires=target_wires, control_values=control_values
            )

    @pytest.mark.parametrize(
        "control_wires,wires,control_values",
        [
            ([0], 1, "0"),
            ([0, 1], 2, "00"),
            ([0, 1], 2, "10"),
            ([0, 1], 2, "11"),
            ([1, 0], 2, "01"),
            ([0, 1], [2, 3], "11"),
            ([0, 2], [3, 1], "10"),
            ([1, 2, 0], [3, 4], "100"),
            ([1, 0, 2], [4, 3], "110"),
        ],
    )
    def test_mixed_polarity_controls(self, control_wires, wires, control_values):
        """Test if ControlledQubitUnitary properly applies mixed-polarity
        control values."""
        target_wires = Wires(wires)

        dev = qml.device("default.qubit", wires=len(control_wires + target_wires))

        # Pick a random unitary
        U = unitary_group.rvs(2 ** len(target_wires), random_state=1967)

        # Pick random starting state for the control and target qubits
        control_state_weights = np.random.normal(size=(2 ** (len(control_wires) + 1) - 2))
        target_state_weights = np.random.normal(size=(2 ** (len(target_wires) + 1) - 2))

        @qml.qnode(dev)
        def circuit_mixed_polarity():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            qml.ControlledQubitUnitary(
                U, control_wires=control_wires, wires=target_wires, control_values=control_values
            )
            return qml.state()

        # The result of applying the mixed-polarity gate should be the same as
        # if we conjugated the specified control wires with Pauli X and applied the
        # "regular" ControlledQubitUnitary in between.

        x_locations = [x for x in range(len(control_values)) if control_values[x] == "0"]

        @qml.qnode(dev)
        def circuit_pauli_x():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            qml.ControlledQubitUnitary(U, control_wires=control_wires, wires=wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            return qml.state()

        mixed_polarity_state = circuit_mixed_polarity()
        pauli_x_state = circuit_pauli_x()

        assert np.allclose(mixed_polarity_state, pauli_x_state)


class TestMultiControlledX:
    """Tests for the MultiControlledX"""

    X = np.array([[0, 1], [1, 0]])

    @pytest.mark.parametrize(
        "control_wires,wires,control_values,expected_error_message",
        [
            ([0, 1], 2, "ab", "String of control values can contain only '0' or '1'."),
            ([0, 1], 2, "011", "Length of control bit string must equal number of control wires."),
            ([0, 1], 2, [0, 1], "Alternative control values must be passed as a binary string."),
            (
                [0, 1],
                [2, 3],
                "10",
                "MultiControlledX accepts a single target wire.",
            ),
        ],
    )
    def test_invalid_mixed_polarity_controls(
        self, control_wires, wires, control_values, expected_error_message
    ):
        """Test if MultiControlledX properly handles invalid mixed-polarity
        control values."""
        target_wires = Wires(wires)

        with pytest.raises(ValueError, match=expected_error_message):
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wires, control_values=control_values
            )

    @pytest.mark.parametrize(
        "control_wires,wires,control_values",
        [
            ([0], 1, "0"),
            ([0, 1], 2, "00"),
            ([0, 1], 2, "10"),
            ([1, 0], 2, "10"),
            ([0, 1], 2, "11"),
            ([0, 2], 1, "10"),
            ([1, 2, 0], 3, "100"),
            ([1, 0, 2, 4], 3, "1001"),
            ([0, 1, 2, 5, 3, 6], 4, "100001"),
        ],
    )
    def test_mixed_polarity_controls(self, control_wires, wires, control_values):
        """Test if MultiControlledX properly applies mixed-polarity
        control values."""
        target_wires = Wires(wires)

        dev = qml.device("default.qubit", wires=len(control_wires + target_wires))

        # Pick random starting state for the control and target qubits
        control_state_weights = np.random.normal(size=(2 ** (len(control_wires) + 1) - 2))
        target_state_weights = np.random.normal(size=(2 ** (len(target_wires) + 1) - 2))

        @qml.qnode(dev)
        def circuit_mpmct():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wires, control_values=control_values
            )
            return qml.state()

        # The result of applying the mixed-polarity gate should be the same as
        # if we conjugated the specified control wires with Pauli X and applied the
        # "regular" ControlledQubitUnitary in between.

        x_locations = [x for x in range(len(control_values)) if control_values[x] == "0"]

        @qml.qnode(dev)
        def circuit_pauli_x():
            qml.templates.ArbitraryStatePreparation(control_state_weights, wires=control_wires)
            qml.templates.ArbitraryStatePreparation(target_state_weights, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            qml.ControlledQubitUnitary(X, control_wires=control_wires, wires=target_wires)

            for wire in x_locations:
                qml.PauliX(wires=control_wires[wire])

            return qml.state()

        mpmct_state = circuit_mpmct()
        pauli_x_state = circuit_pauli_x()

        assert np.allclose(mpmct_state, pauli_x_state)

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 6))
    def test_decomposition_with_many_workers(self, n_ctrl_wires):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there are many work wires."""
        control_wires = range(n_ctrl_wires)
        target_wire = n_ctrl_wires
        work_wires = range(n_ctrl_wires + 1, 2 * n_ctrl_wires + 1)

        dev = qml.device("default.qubit", wires=2 * n_ctrl_wires + 1)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX._decomposition_with_many_workers(
                control_wires, target_wire, work_wires
            )
        assert all(isinstance(op, qml.Toffoli) for op in tape.operations)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(control_wires=control_wires, wires=target_wire).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array([f(b) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    @pytest.mark.parametrize("n_ctrl_wires", range(3, 6))
    def test_decomposition_with_one_worker(self, n_ctrl_wires):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on the case where there is one work wire."""
        control_wires = Wires(range(n_ctrl_wires))
        target_wire = n_ctrl_wires
        work_wires = n_ctrl_wires + 1

        dev = qml.device("default.qubit", wires=n_ctrl_wires + 2)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX._decomposition_with_one_worker(
                control_wires, target_wire, work_wires
            )
        tape = tape.expand(depth=1)
        assert all(
            isinstance(op, qml.Toffoli) or isinstance(op, qml.CNOT) for op in tape.operations
        )

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(control_wires=control_wires, wires=target_wire).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array([f(b) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]).T
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    def test_not_enough_workers(self):
        """Test that a ValueError is raised when more than 2 control wires are to be decomposed with
        no work wires supplied"""
        control_wires = range(3)
        target_wire = 4
        op = qml.MultiControlledX(control_wires=control_wires, wires=target_wire)

        match = (
            f"At least one work wire is required to decompose operation: {re.escape(op.__repr__())}"
        )
        with pytest.raises(ValueError, match=match):
            op.decomposition()

    def test_not_unique_wires(self):
        """Test that a ValueError is raised when work_wires is not complementary to control_wires"""
        control_wires = range(3)
        target_wire = 4
        work_wires = range(2)
        with pytest.raises(ValueError, match="The work wires must be different from the control"):
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wire, work_wires=work_wires
            )

    @pytest.mark.parametrize("control_val", ["0", "1"])
    @pytest.mark.parametrize("n_ctrl_wires", range(1, 6))
    def test_decomposition_with_flips(self, n_ctrl_wires, control_val, mocker):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on varying the control values."""
        control_values = control_val * n_ctrl_wires
        control_wires = range(n_ctrl_wires)
        target_wire = n_ctrl_wires
        work_wires = range(n_ctrl_wires + 1, 2 * n_ctrl_wires + 1)

        spy = mocker.spy(qml.MultiControlledX, "decomposition")
        dev = qml.device("default.qubit", wires=2 * n_ctrl_wires + 1)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX(
                control_wires=control_wires,
                wires=target_wire,
                work_wires=work_wires,
                control_values=control_values,
            )
        tape = tape.expand(depth=1)
        assert all(not isinstance(op, qml.MultiControlledX) for op in tape.operations)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=range(n_ctrl_wires + 1))
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wire, control_values=control_values
            ).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=range(n_ctrl_wires + 1))

        u = np.array([f(b) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]).T
        spy.assert_called()
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    def test_decomposition_with_custom_wire_labels(self, mocker):
        """Test that the decomposed MultiControlledX gate performs the same unitary as the
        matrix-based version by checking if U^dagger U applies the identity to each basis
        state. This test focuses on using custom wire labels."""
        n_ctrl_wires = 4
        control_wires = [-1, "alice", 42, 3.14]
        target_wire = ["bob"]
        work_wires = ["charlie"]
        all_wires = control_wires + target_wire + work_wires

        spy = mocker.spy(qml.MultiControlledX, "decomposition")
        dev = qml.device("default.qubit", wires=all_wires)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wire, work_wires=work_wires
            )
        tape = tape.expand(depth=2)
        assert all(not isinstance(op, qml.MultiControlledX) for op in tape.operations)

        @qml.qnode(dev)
        def f(bitstring):
            qml.BasisState(bitstring, wires=control_wires + target_wire)
            qml.MultiControlledX(control_wires=control_wires, wires=target_wire).inv()
            for op in tape.operations:
                op.queue()
            return qml.probs(wires=control_wires + target_wire)

        u = np.array([f(b) for b in itertools.product(range(2), repeat=n_ctrl_wires + 1)]).T
        spy.assert_called()
        assert np.allclose(u, np.eye(2 ** (n_ctrl_wires + 1)))

    def test_worker_state_unperturbed(self, mocker):
        """Test that the state of the worker wires is unperturbed after the decomposition has used
        them. To do this, a random state over all the qubits (control, target and workers) is
        loaded and U^dagger U(decomposed) is applied. If the workers are uncomputed, the output
        state will be the same as the input."""
        control_wires = range(4)
        target_wire = 4
        worker_wires = [5, 6]
        n_all_wires = 7

        rnd_state = unitary_group.rvs(2 ** n_all_wires, random_state=1)[0]
        spy = mocker.spy(qml.MultiControlledX, "decomposition")
        dev = qml.device("default.qubit", wires=n_all_wires)

        with qml.tape.QuantumTape() as tape:
            qml.MultiControlledX(
                control_wires=control_wires, wires=target_wire, work_wires=worker_wires
            )
        tape = tape.expand(depth=1)
        assert all(not isinstance(op, qml.MultiControlledX) for op in tape.operations)

        @qml.qnode(dev)
        def f():
            qml.QubitStateVector(rnd_state, wires=range(n_all_wires))
            qml.MultiControlledX(control_wires=control_wires, wires=target_wire).inv()
            for op in tape.operations:
                op.queue()
            return qml.state()

        assert np.allclose(f(), rnd_state)
        spy.assert_called()


class TestArithmetic:
    """Tests the arithmetic operations."""

    @pytest.mark.parametrize(
        "wires,input_string,output_string,expand",
        [
            ([0, 1, 2, 3], "0000", "0000", True),
            ([0, 1, 2, 3], "0001", "0001", True),
            ([0, 1, 2, 3], "0010", "0010", True),
            ([0, 1, 2, 3], "0011", "0011", True),
            ([0, 1, 2, 3], "0100", "0110", True),
            ([0, 1, 2, 3], "0101", "0111", True),
            ([0, 1, 2, 3], "0110", "0101", True),
            ([0, 1, 2, 3], "0111", "0100", True),
            ([0, 1, 2, 3], "1000", "1000", True),
            ([0, 1, 2, 3], "1001", "1001", True),
            ([0, 1, 2, 3], "1010", "1011", True),
            ([0, 1, 2, 3], "1011", "1010", True),
            ([0, 1, 2, 3], "1100", "1111", True),
            ([0, 1, 2, 3], "1101", "1110", True),
            ([0, 1, 2, 3], "1110", "1101", True),
            ([0, 1, 2, 3], "1111", "1100", True),
            ([3, 1, 2, 0], "0110", "1100", True),
            ([3, 2, 0, 1], "1010", "0110", True),
            ([0, 1, 2, 3], "0000", "0000", False),
            ([0, 1, 2, 3], "0001", "0001", False),
            ([0, 1, 2, 3], "0010", "0010", False),
            ([0, 1, 2, 3], "0011", "0011", False),
            ([0, 1, 2, 3], "0100", "0110", False),
            ([0, 1, 2, 3], "0101", "0111", False),
            ([0, 1, 2, 3], "0110", "0101", False),
            ([0, 1, 2, 3], "0111", "0100", False),
            ([0, 1, 2, 3], "1000", "1000", False),
            ([0, 1, 2, 3], "1001", "1001", False),
            ([0, 1, 2, 3], "1010", "1011", False),
            ([0, 1, 2, 3], "1011", "1010", False),
            ([0, 1, 2, 3], "1100", "1111", False),
            ([0, 1, 2, 3], "1101", "1110", False),
            ([0, 1, 2, 3], "1110", "1101", False),
            ([0, 1, 2, 3], "1111", "1100", False),
            ([3, 1, 2, 0], "0110", "1100", False),
            ([3, 2, 0, 1], "1010", "0110", False),
        ],
    )
    def test_QubitCarry(self, wires, input_string, output_string, expand, mocker):
        """Test if ``QubitCarry`` produces the right output and is expandable."""
        dev = qml.device("default.qubit", wires=4)
        spy = mocker.spy(qml.QubitCarry, "decomposition")

        with qml.tape.QuantumTape() as tape:
            for i in range(len(input_string)):
                if input_string[i] == "1":
                    qml.PauliX(i)
            qml.QubitCarry(wires=wires)
            qml.probs(wires=[0, 1, 2, 3])

        if expand:
            tape = tape.expand()
        result = dev.execute(tape)
        result = np.argmax(result)
        result = format(result, "04b")
        assert result == output_string

        # checks that decomposition is only used when intended
        assert expand is (len(spy.call_args_list) != 0)

    def test_QubitCarry_superposition(self):
        """Test if ``QubitCarry`` works for superposition input states."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=1)
            qml.Hadamard(wires=2)
            qml.QubitCarry(wires=[0, 1, 2, 3])
            return qml.probs(wires=3)

        result = circuit()
        assert np.allclose(result, 0.5)

    # fmt: off
    @pytest.mark.parametrize(
        "wires,input_state,output_state,expand",
        [
            ([0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], True),
            ([2, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], True),
            ([1, 2, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], True),
            ([0, 1, 2], [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5, 0, 0.5, 0.5, 0], True),
            ([0, 1, 2], [np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8)],
            [np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8)], True),
            ([0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], False),
            ([2, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], False),
            ([1, 2, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], False),
            ([0, 1, 2], [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5, 0, 0.5, 0.5, 0], False),
            ([0, 1, 2], [np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8)],
            [np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8), np.sqrt(1/8)], False),
        ],
    )
    # fmt: on
    def test_QubitSum(self, wires, input_state, output_state, expand, mocker):
        """Test if ``QubitSum`` produces the correct output"""
        dev = qml.device("default.qubit", wires=3)
        spy = mocker.spy(qml.QubitSum, "decomposition")

        with qml.tape.QuantumTape() as tape:
            qml.QubitStateVector(input_state, wires=[0, 1, 2])

            if expand:
                qml.QubitSum(wires=wires).expand()
            else:
                qml.QubitSum(wires=wires)

            qml.state()

        result = dev.execute(tape)
        assert np.allclose(result, output_state)

        # checks that decomposition is only used when intended
        assert expand is (len(spy.call_args_list) != 0)

    def test_qubit_sum_adjoint(self):
        """Test the adjoint method of QubitSum by reconstructing the unitary matrix and checking
        if it is equal to qml.QubitSum.matrix (recall that the operation is self-adjoint)"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def f(state):
            qml.QubitStateVector(state, wires=range(3))
            qml.adjoint(qml.QubitSum)(wires=range(3))
            return qml.probs(wires=range(3))

        u = np.array([f(state) for state in np.eye(2 ** 3)]).T
        assert np.allclose(u, qml.QubitSum._matrix())

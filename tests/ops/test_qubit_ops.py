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
import pytest
import functools
import numpy as np
from numpy.linalg import multi_dot

import pennylane as qml
from pennylane.wires import Wires

from gate_data import I, X, Y, Z, H, CNOT, SWAP, CZ, S, T, CSWAP, Toffoli


# Standard observables, their matrix representation, and eigenvlaues
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

        assert np.allclose(
            qubit_unitary_2[0].data, observable_2_eigvecs.conj().T, atol=tol, rtol=0
        )
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
    @pytest.mark.parametrize("tapemode", [True, False])
    def test_CRot_decomposition(self, tol, phi, theta, omega, tapemode, monkeypatch):
        """Tests that the decomposition of the CRot gate is correct"""
        op = qml.CRot(phi, theta, omega, wires=[0, 1])

        with monkeypatch.context() as m:
            m.setattr(qml, "tape_mode_active", lambda: tapemode)
            res = op.decomposition(phi, theta, omega, op.wires)

        mats = []
        for i in reversed(res):
            if len(i.wires) == 1:
                mats.append(np.kron(np.eye(2), i.matrix))
            else:
                mats.append(i.matrix)

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


class TestPauliRot:
    """Test the PauliRot operation."""

    @pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 7))
    @pytest.mark.parametrize(
        "pauli_word,expected_matrix", PAULI_ROT_PARAMETRIC_MATRIX_TEST_DATA,
    )
    def test_PauliRot_matrix_parametric(self, theta, pauli_word, expected_matrix, tol):
        """Test parametrically that the PauliRot matrix is correct."""

        res = qml.PauliRot._matrix(theta, pauli_word)
        expected = expected_matrix(theta)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "theta,pauli_word,expected_matrix", PAULI_ROT_MATRIX_TEST_DATA,
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
    def test_differentiability(self, angle):
        """Test that differentiation of PauliRot works."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.PauliRot(theta, "XX", wires=[0, 1])

            return qml.expval(qml.PauliZ(0))

        res = circuit(angle)
        gradient = np.squeeze(circuit.jacobian(angle))

        assert gradient == 0.5 * (circuit(angle + np.pi / 2) - circuit(angle - np.pi / 2))

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
        assert np.squeeze(circuit.jacobian(angle)) == pytest.approx(
            np.squeeze(decomp_circuit.jacobian(angle)), abs=tol
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

    @pytest.mark.parametrize("pauli_word,wires", [("XYZ", [0, 1]), ("XYZ", [0, 1, 2, 3]),])
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

        if pauli_word[0] == 'I':
            # this is the identity
            expected_gen = qml.Identity(wires=0) 
        else:
            expected_gen = getattr(
                qml, 'Pauli{}'.format(pauli_word[0]))(wires=0)

        for i, pauli in enumerate(pauli_word[1:]):
            i += 1
            if pauli == 'I':
                expected_gen = expected_gen @  qml.Identity(
                    wires=i) 
            else:
                expected_gen = expected_gen @ getattr(
                    qml, 'Pauli{}'.format(pauli))(wires=i)

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
            ([0, 1], lambda theta: np.diag(np.exp(1j * np.array([-1, 1, 1, -1]) * theta / 2),),),
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
    def test_differentiability(self, angle):
        """Test that differentiation of MultiRZ works."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(0)
            qml.MultiRZ(theta, wires=[0, 1])

            return qml.expval(qml.PauliX(0))

        res = circuit(angle)
        gradient = np.squeeze(circuit.jacobian(angle))

        assert gradient == 0.5 * (circuit(angle + np.pi / 2) - circuit(angle - np.pi / 2))

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
        assert np.squeeze(circuit.jacobian(angle)) == pytest.approx(
            np.squeeze(decomp_circuit.jacobian(angle)), abs=tol
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

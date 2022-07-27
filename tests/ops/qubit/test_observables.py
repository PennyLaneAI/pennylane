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
"""Unit tests for qubit observables."""
import functools
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
from gate_data import H, I, X, Y, Z

import pennylane as qml

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

# Testing Projector observable with the basis states.
PROJECTOR_EIGVALS_TEST_DATA = [
    (np.array([0, 0])),
    (np.array([1, 0, 1])),
]


class TestSimpleObservables:
    """Tests for simple single-qubit observables"""

    @pytest.mark.parametrize("obs, mat, eigs", OBSERVABLES)
    def test_diagonalization(self, obs, mat, eigs, tol):
        """Test the method transforms standard observables into the Z-gate."""
        ob = obs(wires=0)
        A = ob.matrix()

        diag_gates = ob.diagonalizing_gates()
        U = np.eye(2)

        if diag_gates:
            mats = [i.matrix() for i in diag_gates]
            # Need to revert the order in which the matrices are applied such that they adhere to the order
            # of matrix multiplication
            # E.g. for PauliY: [PauliZ(wires=self.wires), S(wires=self.wires), Hadamard(wires=self.wires)]
            # becomes Hadamard @ S @ PauliZ, where @ stands for matrix multiplication
            mats = mats[::-1]
            U = np.linalg.multi_dot([np.eye(2)] + mats)

        res = U @ A @ U.conj().T
        expected = np.diag(eigs)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_diagonalization_static_identity(self):
        """Test the static compute_diagonalizing_gates method for the Identity observable."""
        assert qml.Identity.compute_diagonalizing_gates(wires=1) == []

    def test_diagonalization_static_hadamard(self):
        """Test the static compute_diagonalizing_gates method for the Hadamard observable."""
        res = qml.Hadamard.compute_diagonalizing_gates(wires=1)
        assert len(res) == 1
        assert res[0].name == "RY"
        assert res[0].parameters[0] == -np.pi / 4
        assert res[0].wires.tolist() == [1]

    def test_diagonalization_static_paulix(self):
        """Test the static compute_diagonalizing_gates method for the PauliX observable."""
        res = qml.PauliX.compute_diagonalizing_gates(wires=1)
        assert len(res) == 1
        assert res[0].name == "Hadamard"
        assert res[0].parameters == []
        assert res[0].wires.tolist() == [1]

    def test_diagonalization_static_pauliy(self):
        """Test the static compute_diagonalizing_gates method for the PauliY observable."""
        res = qml.PauliY.compute_diagonalizing_gates(wires=1)
        assert len(res) == 3
        assert res[0].name == "PauliZ"
        assert res[1].name == "S"
        assert res[2].name == "Hadamard"
        assert res[0].wires.tolist() == [1]
        assert res[1].wires.tolist() == [1]
        assert res[2].wires.tolist() == [1]

    def test_diagonalization_static_pauliz(self):
        """Test the static compute_diagonalizing_gates method for the PauliZ observable."""
        assert qml.PauliZ.compute_diagonalizing_gates(wires=1) == []

    @pytest.mark.parametrize("obs, mat, eigs", OBSERVABLES)
    def test_eigvals(self, obs, mat, eigs, tol):
        """Test eigenvalues of standard observables are correct"""
        obs = obs(wires=0)
        res = obs.eigvals()
        assert np.allclose(res, eigs, atol=tol, rtol=0)

    @pytest.mark.parametrize("obs, mat, eigs", OBSERVABLES)
    def test_matrices(self, obs, mat, eigs, tol):
        """Test matrices of standard observables are correct"""
        obs = obs(wires=0)
        res = obs.matrix()
        assert np.allclose(res, mat, atol=tol, rtol=0)


# run all tests in this class in the same thread.
# Prevents multiple threads from updating Hermitian._eigs at the same time
@pytest.mark.xdist_group(name="hermitian_cache_group")
class TestHermitian:
    """Test the Hermitian observable"""

    def setup_method(self):
        """Patch the _eigs class attribute of the Hermitian class before every test."""
        self.patched_eigs = patch(
            "pennylane.ops.qubit.observables.Hermitian._eigs", PropertyMock(return_value={})
        )
        self.patched_eigs.start()

    def tear_down_method(self):
        """Stop patch after every test."""
        self.patched_eigs.stop()

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

    @pytest.mark.parametrize("observable", EIGVALS_TEST_DATA_MULTI_WIRES)
    def test_hermitian_eigendecomposition_multiple_wires(self, observable, tol):
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

        qml.Hermitian(observable_1, 0).eigvals()
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

        qml.Hermitian(observable_2, 0).eigvals()
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

        qml.Hermitian(observable, 0).eigvals()
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

        qml.Hermitian(observable, 0).eigvals()
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

    def test_hermitian_compute_diagonalizing_gates(self, tol):
        """Tests that the compute_diagonalizing_gates method of the
        Hermitian class returns the correct results."""
        eigvecs = np.array([[0.38268343, -0.92387953], [-0.92387953, -0.38268343]])
        res = qml.Hermitian.compute_diagonalizing_gates(eigvecs, wires=[0])[0].data
        expected = eigvecs.conj().T
        assert np.allclose(res, expected, atol=tol, rtol=0)

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
        out = qml.Hermitian(H, wires=0).matrix()

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert np.allclose(out, H, atol=tol, rtol=0)

    def test_hermitian_exceptions(self):
        """Tests that the hermitian matrix method raises the proper errors."""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be a square matrix"):
            qml.Hermitian(H[1:], wires=0).matrix()

        # test non-Hermitian matrix
        H2 = H.copy()
        H2[0, 1] = 2
        with pytest.raises(ValueError, match="must be Hermitian"):
            qml.Hermitian(H2, wires=0).matrix()

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""
        A = np.array([[6 + 0j, 1 - 2j], [1 + 2j, -1]])
        res_static = qml.Hermitian.compute_matrix(A)
        res_dynamic = qml.Hermitian(A, wires=0).matrix()
        expected = np.array([[6.0 + 0.0j, 1.0 - 2.0j], [1.0 + 2.0j, -1.0 + 0.0j]])
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)


class TestProjector:
    """Tests for projector observable"""

    @pytest.mark.parametrize("basis_state", PROJECTOR_EIGVALS_TEST_DATA)
    def test_projector_eigvals(self, basis_state, tol):
        """Tests that the eigvals property of the Projector class returns the correct results."""
        num_wires = len(basis_state)
        eigvals = qml.Projector(basis_state, wires=range(num_wires)).eigvals()

        if basis_state[0] == 0:
            observable = np.array([[1, 0], [0, 0]])
        elif basis_state[0] == 1:
            observable = np.array([[0, 0], [0, 1]])
        for i in basis_state[1:]:
            if i == 0:
                observable = np.kron(observable, np.array([[1, 0], [0, 0]]))
            elif i == 1:
                observable = np.kron(observable, np.array([[0, 0], [0, 1]]))
        expected_eigvals, expected_eigvecs = np.linalg.eig(observable)

        assert np.allclose(np.sort(eigvals), np.sort(expected_eigvals), atol=tol, rtol=0)
        assert np.allclose(
            eigvals, expected_eigvecs[np.where(expected_eigvals == 1)[0][0]], atol=tol, rtol=0
        )

    @pytest.mark.parametrize("basis_state", PROJECTOR_EIGVALS_TEST_DATA)
    def test_projector_diagonalization(self, basis_state, tol):
        """Test that the projector has an empty list of diagonalizing gates."""
        num_wires = len(basis_state)
        diag_gates = qml.Projector(basis_state, wires=range(num_wires)).diagonalizing_gates()
        assert diag_gates == []

        diag_gates_static = qml.Projector.compute_diagonalizing_gates(
            basis_state, wires=range(num_wires)
        )
        assert diag_gates_static == []

    def test_projector_exceptions(self):
        """Tests that the projector construction raises the proper errors on incorrect inputs."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(basis_state):
            obs = qml.Projector(basis_state, wires=range(2))
            return qml.expval(obs)

        with pytest.raises(ValueError, match="Basis state must be one-dimensional"):
            basis_state = np.random.randint(2, size=(2, 4))
            circuit(basis_state)

        with pytest.raises(ValueError, match="Basis state must be of length"):
            basis_state = np.random.randint(2, size=(3))
            circuit(basis_state)

        with pytest.raises(ValueError, match="Basis state must only consist of 0s"):
            basis_state = np.array([0, 2])
            circuit(basis_state)

    @pytest.mark.parametrize(
        "basis_state,expected,n_wires",
        [
            ([0], np.array([[1, 0], [0, 0]]), 1),
            (
                [1, 0],
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                    ],
                ),
                2,
            ),
            (
                [1, 1],
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                    ],
                ),
                2,
            ),
        ],
    )
    def test_matrix_representation(self, basis_state, expected, n_wires, tol):
        """Test that the matrix representation is defined correctly"""
        res_dynamic = qml.Projector(basis_state, wires=range(n_wires)).matrix()
        res_static = qml.Projector.compute_matrix(basis_state)
        assert np.allclose(res_dynamic, expected, atol=tol)
        assert np.allclose(res_static, expected, atol=tol)

    def test_pow_zero(self):
        """Assert that the projector raised to zero is an empty list."""

        basis_state = np.array([0, 1])
        op = qml.Projector(basis_state, wires=(0, 1))
        assert len(op.pow(0)) == 0

    @pytest.mark.parametrize("n", (1, 3))
    def test_pow_non_zero_positive_int(self, n):
        """Test that the projector raised to a positive integer is just a copy."""
        basis_state = np.array([0, 1])
        op = qml.Projector(basis_state, wires=(0, 1))
        pow_op = op.pow(n)[0]
        assert pow_op.__class__ is qml.Projector
        assert qml.math.allclose(pow_op.data[0], op.data[0])


label_data = [
    (qml.Hermitian(np.eye(2), wires=1), "𝓗"),
    (qml.Projector([1, 0, 1], wires=(0, 1, 2)), "|101⟩⟨101|"),
]


@pytest.mark.parametrize("op, label", label_data)
def test_label_method(op, label):
    """Test non-cache label functionality."""
    assert op.label() == label
    assert op.label(decimals=5) == label
    assert op.label(base_label="obs") == "obs"


def test_hermitian_labelling_w_cache():
    """Test hermitian matrix interacts with matrix cache provided to label."""

    op = qml.Hermitian(X, wires=0)

    cache = {"matrices": [Z]}
    op.label(cache=cache) == "𝓗(M1)"
    assert qml.math.allclose(cache["matrices"][1], X)

    cache = {"matrices": [Z, Y, X]}
    op.label(cache=cache) == "𝓗(M2)"
    assert len(cache["matrices"]) == 3

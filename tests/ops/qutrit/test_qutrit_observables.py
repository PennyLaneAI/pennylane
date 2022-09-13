# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for qutrit observables."""
import functools
from unittest.mock import PropertyMock, patch

import pytest
import pennylane as qml
import numpy as np
from gate_data import GELL_MANN


# Hermitian matrices, their corresponding eigenvalues and eigenvectors.
EIGVALS_TEST_DATA = [
    (
        np.eye(3),
        np.array([1.0, 1.0, 1.0]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ),
    (
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        np.array([-1.0, 1.0, 1.0]),
        np.array(
            [[-0.70710678, -0.70710678, 0.0], [0.0, 0.0, -1.0], [0.70710678, -0.70710678, 0.0]]
        ),
    ),
    (
        np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]),
        np.array([-1.0, 0.0, 1.0]),
        np.array(
            [
                [-0.70710678 + 0.0j, 0.0 + 0.0j, -0.70710678 + 0.0j],
                [0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j],
                [0.0 + 0.70710678j, 0.0 + 0.0j, 0.0 - 0.70710678j],
            ]
        ),
    ),
    (
        np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]]),
        np.array([2.0, 3.0, 4.0]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ),
    (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / 2,
        np.array([-1.0, 0.5, 0.5]),
        np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    ),
]

X_12 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
Z_0 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
EIGVALS_TEST_DATA_MULTI_WIRES = [functools.reduce(np.kron, [X_12, np.eye(3), Z_0])]


# run all tests in this class in the same thread.
# Prevents multiple threads from updating THermitian._eigs at the same time
@pytest.mark.xdist_group(name="thermitian_cache_group")
@pytest.mark.usefixtures("tear_down_thermitian")
class TestTHermitian:
    """Test the THermitian observable"""

    def setup_method(self):
        """Patch the _eigs class attribute of the Hermitian class before every test."""
        self.patched_eigs = patch(
            "pennylane.ops.qutrit.observables.THermitian._eigs", PropertyMock(return_value={})
        )
        self.patched_eigs.start()

    def tear_down_method(self):
        """Stop patch after every test."""
        self.patched_eigs.stop()

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_thermitian_eigegendecomposition_single_wire(self, observable, eigvals, eigvecs, tol):
        """Tests that the eigendecomposition property of the THermitian class returns the correct results
        for a single wire."""

        eigendecomp = qml.THermitian(observable, wires=0).eigendecomposition
        assert np.allclose(eigendecomp["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(eigendecomp["eigvec"], eigvecs, atol=tol, rtol=0)

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.THermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.THermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

    @pytest.mark.parametrize("observable", EIGVALS_TEST_DATA_MULTI_WIRES)
    def test_thermitian_eigendecomposition_multiple_wires(self, observable, tol):
        """Tests that the eigendecomposition property of the THermitian class returns the correct results
        for multiple wires."""

        num_wires = int(np.log(len(observable)) / np.log(3))
        eigendecomp = qml.THermitian(observable, wires=list(range(num_wires))).eigendecomposition

        eigvals, eigvecs = np.linalg.eigh(observable)

        assert np.allclose(eigendecomp["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(eigendecomp["eigvec"], eigvecs, atol=tol, rtol=0)

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.THermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.THermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)
        assert len(qml.THermitian._eigs) == 1

    @pytest.mark.parametrize("obs1", EIGVALS_TEST_DATA)
    @pytest.mark.parametrize("obs2", EIGVALS_TEST_DATA)
    def test_thermitian_eigvals_eigvecs_two_different_observables(self, obs1, obs2, tol):
        """Tests that the eigvals method of the THermitian class returns the correct results
        for two observables."""
        if np.all(obs1[0] == obs2[0]):
            pytest.skip("Test only runs for pairs of differing observable")

        observable_1 = obs1[0]
        observable_1_eigvals = obs1[1]
        observable_1_eigvecs = obs1[2]

        key = tuple(observable_1.flatten().tolist())

        qml.THermitian(observable_1, 0).eigvals()
        assert np.allclose(
            qml.THermitian._eigs[key]["eigval"], observable_1_eigvals, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.THermitian._eigs[key]["eigvec"], observable_1_eigvecs, atol=tol, rtol=0
        )
        assert len(qml.THermitian._eigs) == 1

        observable_2 = obs2[0]
        observable_2_eigvals = obs2[1]
        observable_2_eigvecs = obs2[2]

        key_2 = tuple(observable_2.flatten().tolist())

        qml.THermitian(observable_2, 0).eigvals()
        assert np.allclose(
            qml.THermitian._eigs[key_2]["eigval"], observable_2_eigvals, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.THermitian._eigs[key_2]["eigvec"], observable_2_eigvecs, atol=tol, rtol=0
        )
        assert len(qml.THermitian._eigs) == 2

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_thermitian_eigvals_eigvecs_same_observable_twice(
        self, observable, eigvals, eigvecs, tol
    ):
        """Tests that the eigvals method of the THermitian class keeps the same dictionary entries upon multiple calls."""
        key = tuple(observable.flatten().tolist())

        qml.THermitian(observable, 0).eigvals()
        assert np.allclose(qml.THermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.THermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)
        assert len(qml.THermitian._eigs) == 1

        qml.THermitian(observable, 0).eigvals()
        assert np.allclose(qml.THermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.THermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)
        assert len(qml.THermitian._eigs) == 1

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_diagonalizing_gates(self, observable, eigvals, eigvecs, tol):
        """Tests that the diagonalizing_gates method of the THermitian class returns the correct results."""
        qutrit_unitary = qml.THermitian(observable, wires=[0]).diagonalizing_gates()

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.THermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.THermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

        assert np.allclose(qutrit_unitary[0].data, eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.THermitian._eigs) == 1

    def test_thermitian_compute_diagonalizing_gates(self, tol):
        """Tests that the compute_diagonalizing_gates method of the
        THermitian class returns the correct results."""
        eigvecs = np.array(
            [
                [0.38268343, -0.92387953, 0.70710678],
                [-0.92387953, -0.38268343, -0.70710678],
                [0.70710678, -0.38268343, 1.41421356],
            ]
        )
        res = qml.THermitian.compute_diagonalizing_gates(eigvecs, wires=[0])[0].data
        expected = eigvecs.conj().T
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("obs1", EIGVALS_TEST_DATA)
    @pytest.mark.parametrize("obs2", EIGVALS_TEST_DATA)
    def test_thermitian_diagonalizing_gates_two_different_observables(self, obs1, obs2, tol):
        """Tests that the diagonalizing_gates method of the THermitian class returns the correct results
        for two observables."""
        if np.all(obs1[0] == obs2[0]):
            pytest.skip("Test only runs for pairs of differing observable")

        observable_1 = obs1[0]
        observable_1_eigvals = obs1[1]
        observable_1_eigvecs = obs1[2]

        qutrit_unitary = qml.THermitian(observable_1, wires=[0]).diagonalizing_gates()

        key = tuple(observable_1.flatten().tolist())
        assert np.allclose(
            qml.THermitian._eigs[key]["eigval"], observable_1_eigvals, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.THermitian._eigs[key]["eigvec"], observable_1_eigvecs, atol=tol, rtol=0
        )

        assert np.allclose(qutrit_unitary[0].data, observable_1_eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.THermitian._eigs) == 1

        observable_2 = obs2[0]
        observable_2_eigvals = obs2[1]
        observable_2_eigvecs = obs2[2]

        qutrit_unitary_2 = qml.THermitian(observable_2, wires=[0]).diagonalizing_gates()

        key = tuple(observable_2.flatten().tolist())
        assert np.allclose(
            qml.THermitian._eigs[key]["eigval"], observable_2_eigvals, atol=tol, rtol=0
        )
        assert np.allclose(
            qml.THermitian._eigs[key]["eigvec"], observable_2_eigvecs, atol=tol, rtol=0
        )

        assert np.allclose(
            qutrit_unitary_2[0].data, observable_2_eigvecs.conj().T, atol=tol, rtol=0
        )
        assert len(qml.THermitian._eigs) == 2

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_thermitian_diagonalizing_gates_same_observable_twice(
        self, observable, eigvals, eigvecs, tol
    ):
        """Tests that the diagonalizing_gates method of the THermitian class keeps the same dictionary entries upon multiple calls."""
        qutrit_unitary = qml.THermitian(observable, wires=[0]).diagonalizing_gates()

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.THermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.THermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

        assert np.allclose(qutrit_unitary[0].data, eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.THermitian._eigs) == 1

        qutrit_unitary = qml.THermitian(observable, wires=[0]).diagonalizing_gates()

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.THermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.THermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

        assert np.allclose(qutrit_unitary[0].data, eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.THermitian._eigs) == 1

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_thermitian_diagonalizing_gates_integration(self, observable, eigvals, eigvecs, tol):
        """Tests that the diagonalizing_gates method of the THermitian class
        diagonalizes the given observable."""
        tensor_obs = np.kron(observable, observable)
        eigvals = np.kron(eigvals, eigvals)

        diag_gates = qml.THermitian(tensor_obs, wires=[0, 1]).diagonalizing_gates()

        assert len(diag_gates) == 1

        U = diag_gates[0].parameters[0]
        x = U @ tensor_obs @ U.conj().T
        assert np.allclose(np.diag(np.sort(eigvals)), x, atol=tol, rtol=0)

    def test_thermitian_matrix(self, tol):
        """Test that the Hermitian matrix method produces the correct output."""
        H_01 = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
        out = qml.THermitian(H_01, wires=0).matrix()

        # verify output type
        assert isinstance(out, np.ndarray)

        # verify equivalent to input state
        assert np.allclose(out, H_01, atol=tol, rtol=0)

    def test_thermitian_exceptions(self):
        """Tests that the Hermitian matrix method raises the proper errors."""
        H_01 = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)

        # test non-square matrix
        with pytest.raises(ValueError, match="must be a square matrix"):
            qml.THermitian(H_01[1:], wires=0).matrix()

        # test non-Hermitian matrix
        H2 = H_01.copy()
        H2[0, 1] = 2
        with pytest.raises(ValueError, match="must be Hermitian"):
            qml.THermitian(H2, wires=0).matrix()

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""
        A = np.array([[6 + 0j, 1 - 2j, 0], [1 + 2j, -1, 0], [0, 0, 1]])
        res_static = qml.THermitian.compute_matrix(A)
        res_dynamic = qml.THermitian(A, wires=0).matrix()
        expected = np.array(
            [
                [6.0 + 0.0j, 1.0 - 2.0j, 0.0 + 0.0j],
                [1.0 + 2.0j, -1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            ]
        )
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)


GM_OBSERVABLES = [
    (1, GELL_MANN[0], [1, -1, 0]),
    (2, GELL_MANN[1], [1, -1, 0]),
    (3, GELL_MANN[2], [1, -1, 0]),
    (4, GELL_MANN[3], [1, -1, 0]),
    (5, GELL_MANN[4], [1, -1, 0]),
    (6, GELL_MANN[5], [1, -1, 0]),
    (7, GELL_MANN[6], [1, -1, 0]),
    (8, GELL_MANN[7], np.array([1, 1, -2]) / np.sqrt(3)),
]


class TestGellMannervables:
    """Tests for simple single-qutrit observables"""

    @pytest.mark.parametrize("index, mat, eigs", GM_OBSERVABLES)
    def test_diagonalization(self, index, mat, eigs, tol):
        """Test the method transforms Gell-Mann observables appropriately."""
        ob = qml.GellMann(wires=0, index=index)
        A = ob.matrix()

        diag_gates = ob.diagonalizing_gates()
        U = np.eye(3)

        if index in {3, 8}:
            assert len(diag_gates) == 0
        else:
            assert len(diag_gates) == 1
            assert diag_gates[0].__class__ == qml.QutritUnitary
            mat = diag_gates[0].matrix()
            U = np.dot(np.eye(3), mat)

        res = U @ A @ U.conj().T
        expected = np.diag(eigs)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("index", [0, 9, 1.0, 2.5, "c"])
    def test_index_error(self, index):
        """Test that using bad index values throws the appropriate error"""
        with pytest.raises(
            ValueError,
            match="The index of a Gell-Mann observable must be an integer between 1 and 8 inclusive.",
        ):
            qml.GellMann(wires=0, index=index)

    @pytest.mark.parametrize("index, mat, eigs", GM_OBSERVABLES)
    def test_matrix(self, index, mat, eigs, tol):
        """Test that the Gell-Mann matrices are correct"""
        res_static = qml.GellMann.compute_matrix(index)
        res_dynamic = qml.GellMann(wires=0, index=index).matrix()

        assert np.allclose(res_static, mat)
        assert np.allclose(res_dynamic, mat)

    @pytest.mark.parametrize("index, mat, eigs", GM_OBSERVABLES)
    def test_eigvals(self, index, mat, eigs, tol):
        """Test that the Gell-Mann eigenvalues are correct"""
        res_static = qml.GellMann.compute_eigvals(index)
        res_dynamic = qml.GellMann(wires=0, index=index).eigvals()

        assert np.allclose(res_static, eigs)
        assert np.allclose(res_dynamic, eigs)

    @pytest.mark.parametrize("index", list(range(1, 9)))
    def test_label(self, index):
        """Test that the label is correct"""

        label = f"GellMann({index})"
        obs = qml.GellMann(wires=0, index=index)

        assert obs.label() == label
        assert obs.label(decimals=2) == label

        obs.inv()
        assert obs.label() == label

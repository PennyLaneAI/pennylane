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
Unit tests for the :mod:`pennylane.ops.qubit` operations.
"""
# pylint: disable=protected-access,cell-var-from-loop
import itertools
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.ops import qubit
from pennylane.templates.layers import StronglyEntanglingLayers

# EIGVALS_TEST_DATA is a list of tuples of Hermitian matrices, their corresponding eigenvalues and eigenvectors.
EIGVALS_TEST_DATA = [
        (np.array([[1, 0], [0, 1]]), np.array([1., 1.]), np.array([[1., 0.],[0., 1.]])),
        (np.array([[0, 1], [1, 0]]), np.array([-1., 1.]), np.array([[-0.70710678,  0.70710678],[ 0.70710678,  0.70710678]])),
        (np.array([[0, -1j], [1j, 0]]), np.array([-1., 1.]), np.array([[-0.70710678+0.j        , -0.70710678+0.j        ], [ 0.        +0.70710678j,  0.        -0.70710678j]])),
        (np.array([[1, 0], [0, -1]]), np.array([-1., 1.]), np.array([[0., 1.], [1., 0.]])),
        (1/np.sqrt(2)*np.array([[1, 1],[1, -1]]), np.array([-1., 1.]), np.array([[ 0.38268343, -0.92387953],[-0.92387953, -0.38268343]])),
    ]


@pytest.mark.usefixtures("tear_down_hermitian")
class TestQubit:
    """Tests the qubit based operations."""

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_eigvals_eigvecs(self, observable, eigvals, eigvecs, tol):
        """Tests that the eigvals method of the Hermitian class returns the correct results."""
        key = tuple(observable.flatten().tolist())
        qml.Hermitian.eigvals(observable)
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

        qml.Hermitian.eigvals(observable_1)
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], observable_1_eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], observable_1_eigvecs, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

        observable_2 = obs2[0]
        observable_2_eigvals = obs2[1]
        observable_2_eigvecs = obs2[2]

        key_2 = tuple(observable_2.flatten().tolist())

        qml.Hermitian.eigvals(observable_2)
        assert np.allclose(qml.Hermitian._eigs[key_2]["eigval"], observable_2_eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key_2]["eigvec"], observable_2_eigvecs, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 2

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_eigvals_eigvecs_same_observable_twice(self, observable, eigvals, eigvecs, tol):
        """Tests that the eigvals method of the Hermitian class keeps the same dictionary entries upon multiple calls."""
        key = tuple(observable.flatten().tolist())

        qml.Hermitian.eigvals(observable)
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

        qml.Hermitian.eigvals(observable)
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_diagonalizing_gates(self, observable, eigvals, eigvecs, tol):
        """Tests that the diagonalizing_gates method of the Hermitian class returns the correct results."""
        qubit_unitary = qml.Hermitian.diagonalizing_gates(observable, wires = [0])

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

        assert np.allclose(qubit_unitary[0].params, eigvecs.conj().T, atol=tol, rtol=0)
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

        qubit_unitary = qml.Hermitian.diagonalizing_gates(observable_1, wires = [0])

        key = tuple(observable_1.flatten().tolist())
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], observable_1_eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], observable_1_eigvecs, atol=tol, rtol=0)

        assert np.allclose(qubit_unitary[0].params, observable_1_eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

        observable_2 = obs2[0]
        observable_2_eigvals = obs2[1]
        observable_2_eigvecs = obs2[2]

        qubit_unitary_2 = qml.Hermitian.diagonalizing_gates(observable_2, wires = [0])

        key = tuple(observable_2.flatten().tolist())
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], observable_2_eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], observable_2_eigvecs, atol=tol, rtol=0)

        assert np.allclose(qubit_unitary_2[0].params, observable_2_eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 2

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_diagonalizing_gatesi_same_observable_twice(self, observable, eigvals, eigvecs, tol):
        """Tests that the diagonalizing_gates method of the Hermitian class keeps the same dictionary entries upon multiple calls."""
        qubit_unitary = qml.Hermitian.diagonalizing_gates(observable, wires = [0])

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

        assert np.allclose(qubit_unitary[0].params, eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

        qubit_unitary = qml.Hermitian.diagonalizing_gates(observable, wires = [0])

        key = tuple(observable.flatten().tolist())
        assert np.allclose(qml.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qml.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

        assert np.allclose(qubit_unitary[0].params, eigvecs.conj().T, atol=tol, rtol=0)
        assert len(qml.Hermitian._eigs) == 1

class TestQubitIntegration:
    """Integration for the qubit based operations."""

    @pytest.mark.parametrize("observable, eigvals, eigvecs", EIGVALS_TEST_DATA)
    def test_hermitian_diagonalizing_gates_integration(self, observable, eigvals, eigvecs, tol):
        """Tests that the diagonalizing_gates method of the Hermitian class contains contains a gate that diagonalizes the
        given observable."""
        num_wires = 2

        tensor_obs = np.kron(observable, observable)
        eigvals = np.kron(eigvals, eigvals)

        dev = qml.device('default.qubit', wires=num_wires)

        diag_gates = qml.Hermitian.diagonalizing_gates(tensor_obs, wires=list(range(num_wires)))

        assert len(diag_gates) == 1

        U = diag_gates[0].parameters[0]
        x = U @ tensor_obs @ U.conj().T
        assert np.allclose(np.diag(np.sort(eigvals)), x, atol=tol, rtol=0)


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
Unit tests for the :mod:`pennylane.plugin.DefaultGaussian` device.
"""
# pylint: disable=protected-access,cell-var-from-loop

import pytest

import pennylane
from pennylane import numpy as np
from pennylane.ops import qubit

class TestQubit:
    """Tests the qubit based operations."""

    identity = np.array([[1, 0], [0, 1]])
    paulix = np.array([[0, 1], [1, 0]])
    pauliy = np.array([[0, -1j], [1j, 0]])
    pauliz = np.array([[1, 0], [0, -1]])
    hadamard = 1/np.sqrt(2)*np.array([[1, 1],[1, -1]])

    observables = [identity, paulix, pauliy, pauliz, hadamard]


    @pytest.mark.parametrize("observable, eigvals, eigvecs", [
        (np.array([[1, 0], [0, 1]]), np.array([1., 1.]), np.array([[1., 0.],[0., 1.]])),
        (np.array([[0, 1], [1, 0]]), np.array([-1, 1]), np.array([[-0.70710678,  0.70710678],[ 0.70710678,  0.70710678]])),
        (np.array([[0, -1j], [1j, 0]]), np.array([-1, 1]), np.array([[-0.70710678+0.j        , -0.70710678+0.j        ], [ 0.        +0.70710678j,  0.        -0.70710678j]])),
        (np.array([[1, 0], [0, -1]]), np.array([-1, 1]), np.array([[0., 1.], [1., 0.]])),
        (1/np.sqrt(2)*np.array([[1, 1],[1, -1]]), np.array([-1, 1]), np.array([[ 0.38268343, -0.92387953],[-0.92387953, -0.38268343]])),
    ])
    def test_hermitian_eigvals_eigvecs(self, observable, eigvals, eigvecs, tol):
        """Tests that the eigvals method of the Hermitian class returns the correct results."""
        key = tuple(observable.flatten().tolist())

        qubit.Hermitian.eigvals(observable)
        assert np.allclose(qubit.Hermitian._eigs[key]["eigval"], eigvals, atol=tol, rtol=0)
        assert np.allclose(qubit.Hermitian._eigs[key]["eigvec"], eigvecs, atol=tol, rtol=0)

    @pytest.mark.parametrize("observable, eigvals, eigvecs", [
        (np.array([[1, 0], [0, 1]]), np.array([1., 1.]), np.array([[1., 0.],[0., 1.]])),
        (np.array([[0, 1], [1, 0]]), np.array([-1, 1]), np.array([[-0.70710678,  0.70710678],[ 0.70710678,  0.70710678]])),
        (np.array([[0, -1j], [1j, 0]]), np.array([-1, 1]), np.array([[-0.70710678+0.j        , -0.70710678+0.j        ], [ 0.        +0.70710678j,  0.        -0.70710678j]])),
        (np.array([[1, 0], [0, -1]]), np.array([-1, 1]), np.array([[0., 1.], [1., 0.]])),
        (1/np.sqrt(2)*np.array([[1, 1],[1, -1]]), np.array([-1, 1]), np.array([[ 0.38268343, -0.92387953],[-0.92387953, -0.38268343]])),
    ])
    def test_hermitian_diagonalizing_gates(self, observable, eigvals, eigvecs, tol):
        qubit_unitary = qubit.Hermitian.diagonalizing_gates(observable, wires = [0])
        assert np.allclose(qubit_unitary[0].params, eigvecs.conj().T, atol=tol, rtol=0)


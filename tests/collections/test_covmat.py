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
Unit tests for the :mod:`pennylane.collection.covmat` submodule.
"""
import pytest

import numpy as np
import pennylane as qml
from pennylane.collections.covmat import symmetric_product, CovarianceMatrix

H1 = np.array(
    [
        [-1.06804771 + 0.0j, -0.75838808 + 0.30230447j],
        [-0.75838808 - 0.30230447j, 0.37278698 + 0.0j],
    ]
)
H1I = np.kron(H1, np.eye(2))

H2 = np.array(
    [
        [2.34418375 + 0.0j, -0.01975143 - 0.59445146j],
        [-0.01975143 + 0.59445146j, -2.58303785 + 0.0j],
    ]
)
H3 = np.array(
    [
        [
            -1.41645546 + 0.0j,
            -1.04450946 + 0.74874704j,
            -0.47862897 + 0.20502963j,
            1.13931634 + 0.78861728j,
        ],
        [
            -1.04450946 - 0.74874704j,
            -0.00736014 + 0.0j,
            -1.45475026 + 0.76544868j,
            -0.82776401 + 0.9010034j,
        ],
        [
            -0.47862897 - 0.20502963j,
            -1.45475026 - 0.76544868j,
            1.72094365 + 0.0j,
            -0.11116566 - 1.67020271j,
        ],
        [
            1.13931634 - 0.78861728j,
            -0.82776401 - 0.9010034j,
            -0.11116566 + 1.67020271j,
            0.60507019 + 0.0j,
        ],
    ]
)
IH3 = np.kron(np.eye(2), H3)

H4 = np.array(
    [
        [
            0.96241883 + 0.0j,
            1.19671761 - 0.95830367j,
            -2.06704355 - 2.37562829j,
            0.4261865 - 0.52688039j,
        ],
        [
            1.19671761 + 0.95830367j,
            4.0468481 + 0.0j,
            -0.31784138 - 0.54805623j,
            1.39772101 - 2.02909327j,
        ],
        [
            -2.06704355 + 2.37562829j,
            -0.31784138 + 0.54805623j,
            0.88859588 + 0.0j,
            -1.10427527 - 0.25921907j,
        ],
        [
            0.4261865 + 0.52688039j,
            1.39772101 + 2.02909327j,
            -1.10427527 + 0.25921907j,
            1.39883545 + 0.0j,
        ],
    ]
)
H4I = np.kron(H4, np.eye(2))

XZ = (qml.PauliX(0) @ qml.PauliZ(1)).matrix

class TestSymmetricProduct:
    """Test the symmetric product of observables."""

    @pytest.mark.parametrize(
        "obs1,obs2,expected_product",
        [
            (qml.PauliX(0), qml.PauliX(0), qml.Identity(wires=[0])),
            (qml.PauliY(0), qml.PauliY(0), qml.Identity(wires=[0])),
            (qml.PauliZ(0), qml.PauliZ(0), qml.Identity(wires=[0])),
            (qml.Hadamard(0), qml.Hadamard(0), qml.Hermitian(np.eye(2), wires=[0])),
            (qml.Hermitian(H1, 0), qml.Hermitian(H1, 0), qml.Hermitian(H1 @ H1, wires=[0])),
            (qml.PauliX(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliZ(1), qml.Hermitian(XZ @ XZ, wires=[0, 1])),
            (qml.PauliX(0), qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)),
            (qml.PauliY(0), qml.PauliZ(1), qml.PauliY(0) @ qml.PauliZ(1)),
            (qml.PauliZ(0), qml.Hadamard(1), qml.PauliZ(0) @ qml.Hadamard(1)),
            (qml.Hadamard(0), qml.Hermitian(H1, wires=[1]), qml.Hadamard(0) @ qml.Hermitian(H1, wires=[1])),
            (qml.Hermitian(H1, 0), qml.PauliX(1), qml.Hermitian(H1, 0) @ qml.PauliX(1)),
            (qml.PauliX(0) @ qml.PauliZ(1), qml.PauliX(2) @ qml.PauliZ(3), qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliZ(3)),
            (qml.PauliX(0), qml.PauliY(0), qml.Hermitian(np.zeros((2, 2)), wires=[0])),
            (qml.PauliX(0), qml.PauliZ(0), qml.Hermitian(np.zeros((2, 2)), wires=[0])),
            (qml.PauliY(0), qml.PauliZ(0), qml.Hermitian(np.zeros((2, 2)), wires=[0])),
            (qml.Hermitian(H1, 0), qml.Hermitian(H2, 0), qml.Hermitian((H1 @ H2 + H2 @ H1)/2, wires=[0])),
            (qml.Hermitian(H1, wires=[0]), qml.Hermitian(H3, wires=[0, 1]), qml.Hermitian((H1I @ H3 + H3 @ H1I)/2, wires=[0, 1])),
            (qml.PauliX(0) @ qml.PauliZ(1), qml.Hermitian(H3, wires=[0, 1]), qml.Hermitian((XZ @ H3 + H3 @ XZ)/2, wires=[0, 1])),
            (qml.Hermitian(H3, wires=[1, 2]), qml.Hermitian(H4, wires=[0, 1]), qml.Hermitian((IH3 @ H4I + H4I @ IH3)/2, wires=[0, 1, 2])),
            (qml.Hermitian(H3, wires=[3, 6]), qml.Hermitian(H4, wires=[0, 3]), qml.Hermitian((IH3 @ H4I + H4I @ IH3)/2, wires=[0, 3, 6])),
        ],
    )
    def test_symmetric_product(self, obs1, obs2, expected_product, tol):
        """Test that the symmetric product yields the expected observable."""

        result = symmetric_product(obs1, obs2)

        assert result.name == expected_product.name
        assert result.wires == expected_product.wires
        for result_param, expected_param in zip(result.params, expected_product.params):
            assert np.allclose(result_param, expected_param, atol=tol, rtol=0)

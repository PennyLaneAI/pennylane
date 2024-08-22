# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for functions needed for computing a spin Hamiltonian.
"""

import pytest

import pennylane as qml
from pennylane import X, Z
from pennylane.spin import transverse_ising


def test_coupling_error():
    r"""Test that an error is raised when the provided coupling shape is wrong for
    transverse_ising Hamiltonian."""
    n_cells = [4, 4]
    lattice = "Square"
    with pytest.raises(
        ValueError, match="Coupling should be a number or an array of shape 1x1 or 16x16"
    ):
        transverse_ising(lattice=lattice, n_cells=n_cells, coupling=[1.0, 2.0], neighbour_order=1)


@pytest.mark.parametrize(
    # expected_ham here was obtained from datasets
    ("shape", "n_cells", "J", "h", "expected_ham"),
    [
        (
            "chain",
            [4, 0, 0],
            None,
            0,
            -1.0 * (Z(0) @ Z(1))
            + -1.0 * (Z(1) @ Z(2))
            + -1.0 * (Z(2) @ Z(3))
            + 0.0 * X(0)
            + 0.0 * X(1)
            + 0.0 * X(2)
            + 0.0 * X(3),
        ),
        (
            "chain",
            [4, 0, 0],
            [1.0],
            0,
            -1.0 * (Z(0) @ Z(1))
            + -1.0 * (Z(1) @ Z(2))
            + -1.0 * (Z(2) @ Z(3))
            + 0.0 * X(0)
            + 0.0 * X(1)
            + 0.0 * X(2)
            + 0.0 * X(3),
        ),
        (
            "chain",
            [8, 0, 0],
            [1.0],
            -0.17676768,
            -1.0 * (Z(0) @ Z(1))
            + -1.0 * (Z(1) @ Z(2))
            + -1.0 * (Z(2) @ Z(3))
            + -1.0 * (Z(3) @ Z(4))
            + -1.0 * (Z(4) @ Z(5))
            + -1.0 * (Z(5) @ Z(6))
            + -1.0 * (Z(6) @ Z(7))
            + 0.17676767676767677 * X(0)
            + 0.17676767676767677 * X(1)
            + 0.17676767676767677 * X(2)
            + 0.17676767676767677 * X(3)
            + 0.17676767676767677 * X(4)
            + 0.17676767676767677 * X(5)
            + 0.17676767676767677 * X(6)
            + 0.17676767676767677 * X(7),
        ),
        (
            "rectangle",
            [4, 2, 0],
            [1.0],
            -0.25252525,
            -1.0 * (Z(0) @ Z(1))
            + -1.0 * (Z(0) @ Z(2))
            + -1.0 * (Z(2) @ Z(3))
            + -1.0 * (Z(2) @ Z(4))
            + -1.0 * (Z(4) @ Z(5))
            + -1.0 * (Z(4) @ Z(6))
            + -1.0 * (Z(6) @ Z(7))
            + -1.0 * (Z(1) @ Z(3))
            + -1.0 * (Z(3) @ Z(5))
            + -1.0 * (Z(5) @ Z(7))
            + 0.25252525252525254 * X(0)
            + 0.25252525252525254 * X(1)
            + 0.25252525252525254 * X(2)
            + 0.25252525252525254 * X(3)
            + 0.25252525252525254 * X(4)
            + 0.25252525252525254 * X(5)
            + 0.25252525252525254 * X(6)
            + 0.25252525252525254 * X(7),
        ),
        (
            "rectangle",
            [8, 2, 0],
            [1.0],
            -0.44444444,
            -1.0 * (Z(0) @ Z(1))
            + -1.0 * (Z(0) @ Z(2))
            + -1.0 * (Z(2) @ Z(3))
            + -1.0 * (Z(2) @ Z(4))
            + -1.0 * (Z(4) @ Z(5))
            + -1.0 * (Z(4) @ Z(6))
            + -1.0 * (Z(6) @ Z(7))
            + -1.0 * (Z(6) @ Z(8))
            + -1.0 * (Z(8) @ Z(9))
            + -1.0 * (Z(8) @ Z(10))
            + -1.0 * (Z(10) @ Z(11))
            + -1.0 * (Z(10) @ Z(12))
            + -1.0 * (Z(12) @ Z(13))
            + -1.0 * (Z(12) @ Z(14))
            + -1.0 * (Z(14) @ Z(15))
            + -1.0 * (Z(1) @ Z(3))
            + -1.0 * (Z(3) @ Z(5))
            + -1.0 * (Z(5) @ Z(7))
            + -1.0 * (Z(7) @ Z(9))
            + -1.0 * (Z(9) @ Z(11))
            + -1.0 * (Z(11) @ Z(13))
            + -1.0 * (Z(13) @ Z(15))
            + 0.4444444444444444 * X(0)
            + 0.4444444444444444 * X(1)
            + 0.4444444444444444 * X(2)
            + 0.4444444444444444 * X(3)
            + 0.4444444444444444 * X(4)
            + 0.4444444444444444 * X(5)
            + 0.4444444444444444 * X(6)
            + 0.4444444444444444 * X(7)
            + 0.4444444444444444 * X(8)
            + 0.4444444444444444 * X(9)
            + 0.4444444444444444 * X(10)
            + 0.4444444444444444 * X(11)
            + 0.4444444444444444 * X(12)
            + 0.4444444444444444 * X(13)
            + 0.4444444444444444 * X(14)
            + 0.4444444444444444 * X(15),
        ),
    ],
)
def test_ising_hamiltonian(shape, n_cells, J, h, expected_ham):
    r"""Test that the correct Hamiltonian is generated compared to the datasets"""

    ising_ham = transverse_ising(lattice=shape, n_cells=n_cells, coupling=J, h=h, neighbour_order=1)

    qml.assert_equal(ising_ham, expected_ham)


@pytest.mark.parametrize(
    # expected_ham here was obtained manually.
    ("shape", "n_cells", "J", "h", "expected_ham"),
    [
        (
            "chain",
            [4, 0, 0],
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
            0,
            -1.0 * (Z(0) @ Z(1))
            + -1.0 * (Z(1) @ Z(2))
            + -1.0 * (Z(2) @ Z(3))
            + 0.0 * X(0)
            + 0.0 * X(1)
            + 0.0 * X(2)
            + 0.0 * X(3),
        ),
        (
            "square",
            [2, 2, 0],
            [[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]],
            -1.0,
            -0.5 * (Z(0) @ Z(1))
            + -0.5 * (Z(0) @ Z(2))
            + -0.5 * (Z(2) @ Z(3))
            + -0.5 * (Z(1) @ Z(3))
            + 1.0 * X(0)
            + 1.0 * X(1)
            + 1.0 * X(2)
            + 1.0 * X(3),
        ),
    ],
)
def test_ising_hamiltonian_matrix(shape, n_cells, J, h, expected_ham):
    r"""Test that the correct Hamiltonian is generated when coupling is provided as a matrix"""

    ising_ham = transverse_ising(lattice=shape, n_cells=n_cells, coupling=J, h=h, neighbour_order=1)

    qml.assert_equal(ising_ham, expected_ham)

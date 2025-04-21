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

import re

import pytest

import pennylane as qml
from pennylane import I, X, Y, Z
from pennylane.spin import (
    Lattice,
    emery,
    fermi_hubbard,
    haldane,
    heisenberg,
    kitaev,
    spin_hamiltonian,
    transverse_ising,
)

# pylint: disable=too-many-arguments


def test_coupling_error():
    r"""Test that an error is raised when the provided coupling shape is wrong for
    transverse_ising Hamiltonian."""
    n_cells = [4, 4]
    lattice = "Square"
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The coupling parameter should be a number or an array of shape (1,) or (16,16)"
        ),
    ):
        transverse_ising(lattice=lattice, n_cells=n_cells, coupling=[1.0, 2.0], neighbour_order=1)


@pytest.mark.parametrize(
    # expected_ham here was obtained from datasets
    ("shape", "n_cells", "j", "h", "expected_ham"),
    [
        (
            "chain",
            [4],
            1.0,
            0,
            -1.0 * (Z(0) @ Z(1)) + -1.0 * (Z(1) @ Z(2)) + -1.0 * (Z(2) @ Z(3)),
        ),
        (
            "chain",
            [4],
            1.0,
            0,
            -1.0 * (Z(0) @ Z(1)) + -1.0 * (Z(1) @ Z(2)) + -1.0 * (Z(2) @ Z(3)),
        ),
        (
            "chain",
            [8],
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
            [4, 2],
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
            [8, 2],
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
def test_ising_hamiltonian(shape, n_cells, j, h, expected_ham):
    r"""Test that the correct Hamiltonian is generated compared to the datasets"""

    ising_ham = transverse_ising(lattice=shape, n_cells=n_cells, coupling=j, h=h, neighbour_order=1)

    qml.assert_equal(ising_ham, expected_ham)


@pytest.mark.parametrize(
    # expected_ham here was obtained manually.
    ("shape", "n_cells", "j", "h", "expected_ham"),
    [
        (
            "chain",
            [4],
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
            0,
            -1.0 * (Z(0) @ Z(1)) + -1.0 * (Z(1) @ Z(2)) + -1.0 * (Z(2) @ Z(3)),
        ),
        (
            "square",
            [2, 2],
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
def test_ising_hamiltonian_matrix(shape, n_cells, j, h, expected_ham):
    r"""Test that the correct Hamiltonian is generated when coupling is provided as a matrix"""

    ising_ham = transverse_ising(lattice=shape, n_cells=n_cells, coupling=j, h=h, neighbour_order=1)

    qml.assert_equal(ising_ham, expected_ham)


def test_coupling_error_heisenberg():
    r"""Test that an error is raised when the provided coupling shape is wrong for
    Heisenberg Hamiltonian."""
    n_cells = [4, 4]
    lattice = "Square"
    with pytest.raises(
        ValueError,
        match=re.escape("The coupling parameter shape should be equal to (1,3) or (3,16,16)"),
    ):
        heisenberg(lattice=lattice, n_cells=n_cells, coupling=[1.0, 2.0], neighbour_order=1)


@pytest.mark.parametrize(
    # expected_ham here was obtained from datasets
    ("shape", "n_cells", "j", "expected_ham"),
    [
        (
            "chain",
            [4],
            None,
            1.0 * (Z(0) @ Z(1))
            + 1.0 * (Z(1) @ Z(2))
            + 1.0 * (Z(2) @ Z(3))
            + 1.0 * (X(0) @ X(1))
            + 1.0 * (X(1) @ X(2))
            + 1.0 * (X(2) @ X(3))
            + 1.0 * (Y(0) @ Y(1))
            + 1.0 * (Y(1) @ Y(2))
            + 1.0 * (Y(2) @ Y(3)),
        ),
        (
            "chain",
            [4],
            [[-1.0, -1.0, -0.16161616]],
            -0.16161616161616163 * (Z(0) @ Z(1))
            + -0.16161616161616163 * (Z(1) @ Z(2))
            + -0.16161616161616163 * (Z(2) @ Z(3))
            + -1.0 * (X(0) @ X(1))
            + -1.0 * (X(1) @ X(2))
            + -1.0 * (X(2) @ X(3))
            + -1.0 * (Y(0) @ Y(1))
            + -1.0 * (Y(1) @ Y(2))
            + -1.0 * (Y(2) @ Y(3)),
        ),
        (
            "chain",
            [8],
            [-1.0, -1.0, -0.08080808],
            -0.08080808080808081 * (Z(0) @ Z(1))
            + -0.08080808080808081 * (Z(1) @ Z(2))
            + -0.08080808080808081 * (Z(2) @ Z(3))
            + -0.08080808080808081 * (Z(3) @ Z(4))
            + -0.08080808080808081 * (Z(4) @ Z(5))
            + -0.08080808080808081 * (Z(5) @ Z(6))
            + -0.08080808080808081 * (Z(6) @ Z(7))
            + -1.0 * (X(0) @ X(1))
            + -1.0 * (X(1) @ X(2))
            + -1.0 * (X(2) @ X(3))
            + -1.0 * (X(3) @ X(4))
            + -1.0 * (X(4) @ X(5))
            + -1.0 * (X(5) @ X(6))
            + -1.0 * (X(6) @ X(7))
            + -1.0 * (Y(0) @ Y(1))
            + -1.0 * (Y(1) @ Y(2))
            + -1.0 * (Y(2) @ Y(3))
            + -1.0 * (Y(3) @ Y(4))
            + -1.0 * (Y(4) @ Y(5))
            + -1.0 * (Y(5) @ Y(6))
            + -1.0 * (Y(6) @ Y(7)),
        ),
        (
            "rectangle",
            [4, 2],
            [[-1.0, -1.0, -0.08080808]],
            -0.08080808080808081 * (Z(0) @ Z(1))
            + -0.08080808080808081 * (Z(0) @ Z(2))
            + -0.08080808080808081 * (Z(2) @ Z(3))
            + -0.08080808080808081 * (Z(2) @ Z(4))
            + -0.08080808080808081 * (Z(4) @ Z(5))
            + -0.08080808080808081 * (Z(4) @ Z(6))
            + -0.08080808080808081 * (Z(6) @ Z(7))
            + -0.08080808080808081 * (Z(1) @ Z(3))
            + -0.08080808080808081 * (Z(3) @ Z(5))
            + -0.08080808080808081 * (Z(5) @ Z(7))
            + -1.0 * (X(0) @ X(1))
            + -1.0 * (X(0) @ X(2))
            + -1.0 * (X(2) @ X(3))
            + -1.0 * (X(2) @ X(4))
            + -1.0 * (X(4) @ X(5))
            + -1.0 * (X(4) @ X(6))
            + -1.0 * (X(6) @ X(7))
            + -1.0 * (X(1) @ X(3))
            + -1.0 * (X(3) @ X(5))
            + -1.0 * (X(5) @ X(7))
            + -1.0 * (Y(0) @ Y(1))
            + -1.0 * (Y(0) @ Y(2))
            + -1.0 * (Y(2) @ Y(3))
            + -1.0 * (Y(2) @ Y(4))
            + -1.0 * (Y(4) @ Y(5))
            + -1.0 * (Y(4) @ Y(6))
            + -1.0 * (Y(6) @ Y(7))
            + -1.0 * (Y(1) @ Y(3))
            + -1.0 * (Y(3) @ Y(5))
            + -1.0 * (Y(5) @ Y(7)),
        ),
        (
            "rectangle",
            [8, 2],
            [[-1.0, -1.0, -0.12121212]],
            -0.12121212121212122 * (Z(0) @ Z(1))
            + -0.12121212121212122 * (Z(0) @ Z(2))
            + -0.12121212121212122 * (Z(2) @ Z(3))
            + -0.12121212121212122 * (Z(2) @ Z(4))
            + -0.12121212121212122 * (Z(4) @ Z(5))
            + -0.12121212121212122 * (Z(4) @ Z(6))
            + -0.12121212121212122 * (Z(6) @ Z(7))
            + -0.12121212121212122 * (Z(6) @ Z(8))
            + -0.12121212121212122 * (Z(8) @ Z(9))
            + -0.12121212121212122 * (Z(8) @ Z(10))
            + -0.12121212121212122 * (Z(10) @ Z(11))
            + -0.12121212121212122 * (Z(10) @ Z(12))
            + -0.12121212121212122 * (Z(12) @ Z(13))
            + -0.12121212121212122 * (Z(12) @ Z(14))
            + -0.12121212121212122 * (Z(14) @ Z(15))
            + -0.12121212121212122 * (Z(1) @ Z(3))
            + -0.12121212121212122 * (Z(3) @ Z(5))
            + -0.12121212121212122 * (Z(5) @ Z(7))
            + -0.12121212121212122 * (Z(7) @ Z(9))
            + -0.12121212121212122 * (Z(9) @ Z(11))
            + -0.12121212121212122 * (Z(11) @ Z(13))
            + -0.12121212121212122 * (Z(13) @ Z(15))
            + -1.0 * (X(0) @ X(1))
            + -1.0 * (X(0) @ X(2))
            + -1.0 * (X(2) @ X(3))
            + -1.0 * (X(2) @ X(4))
            + -1.0 * (X(4) @ X(5))
            + -1.0 * (X(4) @ X(6))
            + -1.0 * (X(6) @ X(7))
            + -1.0 * (X(6) @ X(8))
            + -1.0 * (X(8) @ X(9))
            + -1.0 * (X(8) @ X(10))
            + -1.0 * (X(10) @ X(11))
            + -1.0 * (X(10) @ X(12))
            + -1.0 * (X(12) @ X(13))
            + -1.0 * (X(12) @ X(14))
            + -1.0 * (X(14) @ X(15))
            + -1.0 * (X(1) @ X(3))
            + -1.0 * (X(3) @ X(5))
            + -1.0 * (X(5) @ X(7))
            + -1.0 * (X(7) @ X(9))
            + -1.0 * (X(9) @ X(11))
            + -1.0 * (X(11) @ X(13))
            + -1.0 * (X(13) @ X(15))
            + -1.0 * (Y(0) @ Y(1))
            + -1.0 * (Y(0) @ Y(2))
            + -1.0 * (Y(2) @ Y(3))
            + -1.0 * (Y(2) @ Y(4))
            + -1.0 * (Y(4) @ Y(5))
            + -1.0 * (Y(4) @ Y(6))
            + -1.0 * (Y(6) @ Y(7))
            + -1.0 * (Y(6) @ Y(8))
            + -1.0 * (Y(8) @ Y(9))
            + -1.0 * (Y(8) @ Y(10))
            + -1.0 * (Y(10) @ Y(11))
            + -1.0 * (Y(10) @ Y(12))
            + -1.0 * (Y(12) @ Y(13))
            + -1.0 * (Y(12) @ Y(14))
            + -1.0 * (Y(14) @ Y(15))
            + -1.0 * (Y(1) @ Y(3))
            + -1.0 * (Y(3) @ Y(5))
            + -1.0 * (Y(5) @ Y(7))
            + -1.0 * (Y(7) @ Y(9))
            + -1.0 * (Y(9) @ Y(11))
            + -1.0 * (Y(11) @ Y(13))
            + -1.0 * (Y(13) @ Y(15)),
        ),
    ],
)
def test_heisenberg_hamiltonian(shape, n_cells, j, expected_ham):
    r"""Test that the correct Hamiltonian is generated compared to the datasets"""
    heisenberg_ham = heisenberg(lattice=shape, n_cells=n_cells, coupling=j, neighbour_order=1)

    qml.assert_equal(heisenberg_ham, expected_ham)


@pytest.mark.parametrize(
    # expected_ham here was obtained manually.
    ("shape", "n_cells", "j", "expected_ham"),
    [
        (
            "chain",
            [4],
            [
                [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
                [[0, 2, 0, 0], [2, 0, 2, 0], [0, 2, 0, 2], [0, 0, 2, 0]],
                [[0, 3, 0, 0], [3, 0, 3, 0], [0, 3, 0, 3], [0, 0, 3, 0]],
            ],
            (1 * X(0)) @ X(1)
            + (2 * Y(0)) @ Y(1)
            + (3 * Z(0)) @ Z(1)
            + (1 * X(1)) @ X(2)
            + (2 * Y(1)) @ Y(2)
            + (3 * Z(1)) @ Z(2)
            + (1 * X(2)) @ X(3)
            + (2 * Y(2)) @ Y(3)
            + (3 * Z(2)) @ Z(3),
        ),
        (
            "square",
            [2, 2],
            [
                [[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]],
                [[0, 1.0, 1.0, 0], [1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0], [0, 1.0, 1.0, 0]],
                [[0, 1.0, 1.0, 0], [1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0], [0, 1.0, 1.0, 0]],
            ],
            (0.5 * X(0)) @ X(1)
            + (1.0 * Y(0)) @ Y(1)
            + (1.0 * Z(0)) @ Z(1)
            + (0.5 * X(0)) @ X(2)
            + (1.0 * Y(0)) @ Y(2)
            + (1.0 * Z(0)) @ Z(2)
            + (0.5 * X(1)) @ X(3)
            + (1.0 * Y(1)) @ Y(3)
            + (1.0 * Z(1)) @ Z(3)
            + (0.5 * X(2)) @ X(3)
            + (1.0 * Y(2)) @ Y(3)
            + (1.0 * Z(2)) @ Z(3),
        ),
    ],
)
def test_heisenberg_hamiltonian_matrix(shape, n_cells, j, expected_ham):
    r"""Test that the correct Hamiltonian is generated when coupling is provided as a matrix"""

    heisenberg_ham = heisenberg(lattice=shape, n_cells=n_cells, coupling=j)

    qml.assert_equal(heisenberg_ham, expected_ham)


def test_hopping_error_fermi_hubbard():
    r"""Test that an error is raised when the provided hopping shape is wrong for
    fermi_hubbard Hamiltonian."""
    n_cells = [4, 4]
    lattice = "Square"
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The hopping parameter should be a number or an array of shape (1,) or (16,16)"
        ),
    ):
        fermi_hubbard(lattice=lattice, n_cells=n_cells, hopping=[1.0, 2.0], neighbour_order=1)


def test_mapping_error_fermi_hubbard():
    r"""Test that an error is raised when unsupported mapping is provided"""
    n_cells = [4, 4]
    lattice = "Square"
    with pytest.raises(ValueError, match="The 'bk_sf' transformation is not available."):
        fermi_hubbard(lattice=lattice, n_cells=n_cells, mapping="bk_sf")


@pytest.mark.parametrize(
    # expected_ham in Jordan-Wigner transformation was obtained from openfermion and converted to
    # PennyLane format using from_openfermion
    # ham = openfermion.hamiltonians.fermi_hubbard(xaxis=n_cells[0], yaxis=n_cells[1], tunneling=hopping, coulomb=coulomb)
    # jw_ham = openfermion.transforms.jordan_wigner(ham)
    # pl_ham = qml.from_openfermion(jw_ham)
    ("shape", "n_cells", "hopping", "coulomb", "expected_ham"),
    [
        (
            "chain",
            [4],
            1.0,
            0.5,
            -0.5 * (Y(0) @ Z(1) @ Y(2))
            + -0.5 * (X(0) @ Z(1) @ X(2))
            + -0.5 * (Y(1) @ Z(2) @ Y(3))
            + -0.5 * (X(1) @ Z(2) @ X(3))
            + 0.5 * I(0)
            + -0.125 * Z(1)
            + -0.125 * Z(0)
            + 0.125 * (Z(0) @ Z(1))
            + -0.5 * (Y(2) @ Z(3) @ Y(4))
            + -0.5 * (X(2) @ Z(3) @ X(4))
            + -0.5 * (Y(3) @ Z(4) @ Y(5))
            + -0.5 * (X(3) @ Z(4) @ X(5))
            + -0.125 * Z(3)
            + -0.125 * Z(2)
            + 0.125 * (Z(2) @ Z(3))
            + -0.5 * (Y(4) @ Z(5) @ Y(6))
            + -0.5 * (X(4) @ Z(5) @ X(6))
            + -0.5 * (Y(5) @ Z(6) @ Y(7))
            + -0.5 * (X(5) @ Z(6) @ X(7))
            + -0.125 * Z(5)
            + -0.125 * Z(4)
            + 0.125 * (Z(4) @ Z(5))
            + -0.125 * Z(7)
            + -0.125 * Z(6)
            + 0.125 * (Z(6) @ Z(7)),
        ),
        (
            "chain",
            [8],
            [-1.0],
            0.0,
            0.5 * (Y(0) @ Z(1) @ Y(2))
            + 0.5 * (X(0) @ Z(1) @ X(2))
            + 0.5 * (Y(1) @ Z(2) @ Y(3))
            + 0.5 * (X(1) @ Z(2) @ X(3))
            + 0.5 * (Y(2) @ Z(3) @ Y(4))
            + 0.5 * (X(2) @ Z(3) @ X(4))
            + 0.5 * (Y(3) @ Z(4) @ Y(5))
            + 0.5 * (X(3) @ Z(4) @ X(5))
            + 0.5 * (Y(4) @ Z(5) @ Y(6))
            + 0.5 * (X(4) @ Z(5) @ X(6))
            + 0.5 * (Y(5) @ Z(6) @ Y(7))
            + 0.5 * (X(5) @ Z(6) @ X(7))
            + 0.5 * (Y(6) @ Z(7) @ Y(8))
            + 0.5 * (X(6) @ Z(7) @ X(8))
            + 0.5 * (Y(7) @ Z(8) @ Y(9))
            + 0.5 * (X(7) @ Z(8) @ X(9))
            + 0.5 * (Y(8) @ Z(9) @ Y(10))
            + 0.5 * (X(8) @ Z(9) @ X(10))
            + 0.5 * (Y(9) @ Z(10) @ Y(11))
            + 0.5 * (X(9) @ Z(10) @ X(11))
            + 0.5 * (Y(10) @ Z(11) @ Y(12))
            + 0.5 * (X(10) @ Z(11) @ X(12))
            + 0.5 * (Y(11) @ Z(12) @ Y(13))
            + 0.5 * (X(11) @ Z(12) @ X(13))
            + 0.5 * (Y(12) @ Z(13) @ Y(14))
            + 0.5 * (X(12) @ Z(13) @ X(14))
            + 0.5 * (Y(13) @ Z(14) @ Y(15))
            + 0.5 * (X(13) @ Z(14) @ X(15)),
        ),
        (
            "square",
            [2, 2],
            [1.0],
            3.0,
            -0.5 * (Y(0) @ Z(1) @ Y(2))
            + -0.5 * (X(0) @ Z(1) @ X(2))
            + -0.5 * (Y(1) @ Z(2) @ Y(3))
            + -0.5 * (X(1) @ Z(2) @ X(3))
            + -0.5 * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
            + -0.5 * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
            + -0.5 * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
            + -0.5 * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
            + 3.0 * I(0)
            + -0.75 * Z(1)
            + -0.75 * Z(0)
            + 0.75 * (Z(0) @ Z(1))
            + -0.5 * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + -0.5 * (X(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + -0.5 * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + -0.5 * (X(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + -0.75 * Z(3)
            + -0.75 * Z(2)
            + 0.75 * (Z(2) @ Z(3))
            + -0.5 * (Y(4) @ Z(5) @ Y(6))
            + -0.5 * (X(4) @ Z(5) @ X(6))
            + -0.5 * (Y(5) @ Z(6) @ Y(7))
            + -0.5 * (X(5) @ Z(6) @ X(7))
            + -0.75 * Z(5)
            + -0.75 * Z(4)
            + 0.75 * (Z(4) @ Z(5))
            + -0.75 * Z(7)
            + -0.75 * Z(6)
            + 0.75 * (Z(6) @ Z(7)),
        ),
        (
            "rectangle",
            [2, 3],
            [0.1],
            0.2,
            -0.05 * (Y(0) @ Z(1) @ Y(2))
            + -0.05 * (X(0) @ Z(1) @ X(2))
            + -0.05 * (Y(1) @ Z(2) @ Y(3))
            + -0.05 * (X(1) @ Z(2) @ X(3))
            + -0.05 * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + -0.05 * (X(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + -0.05 * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + -0.05 * (X(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + 0.3 * I(0)
            + -0.05 * Z(1)
            + -0.05 * Z(0)
            + 0.05 * (Z(0) @ Z(1))
            + -0.05 * (Y(2) @ Z(3) @ Y(4))
            + -0.05 * (X(2) @ Z(3) @ X(4))
            + -0.05 * (Y(3) @ Z(4) @ Y(5))
            + -0.05 * (X(3) @ Z(4) @ X(5))
            + -0.05 * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Z(7) @ Y(8))
            + -0.05 * (X(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Z(7) @ X(8))
            + -0.05 * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ Z(7) @ Z(8) @ Y(9))
            + -0.05 * (X(3) @ Z(4) @ Z(5) @ Z(6) @ Z(7) @ Z(8) @ X(9))
            + -0.05 * Z(3)
            + -0.05 * Z(2)
            + 0.05 * (Z(2) @ Z(3))
            + -0.05 * (Y(4) @ Z(5) @ Z(6) @ Z(7) @ Z(8) @ Z(9) @ Y(10))
            + -0.05 * (X(4) @ Z(5) @ Z(6) @ Z(7) @ Z(8) @ Z(9) @ X(10))
            + -0.05 * (Y(5) @ Z(6) @ Z(7) @ Z(8) @ Z(9) @ Z(10) @ Y(11))
            + -0.05 * (X(5) @ Z(6) @ Z(7) @ Z(8) @ Z(9) @ Z(10) @ X(11))
            + -0.05 * Z(5)
            + -0.05 * Z(4)
            + 0.05 * (Z(4) @ Z(5))
            + -0.05 * (Y(6) @ Z(7) @ Y(8))
            + -0.05 * (X(6) @ Z(7) @ X(8))
            + -0.05 * (Y(7) @ Z(8) @ Y(9))
            + -0.05 * (X(7) @ Z(8) @ X(9))
            + -0.05 * Z(7)
            + -0.05 * Z(6)
            + 0.05 * (Z(6) @ Z(7))
            + -0.05 * (Y(8) @ Z(9) @ Y(10))
            + -0.05 * (X(8) @ Z(9) @ X(10))
            + -0.05 * (Y(9) @ Z(10) @ Y(11))
            + -0.05 * (X(9) @ Z(10) @ X(11))
            + -0.05 * Z(9)
            + -0.05 * Z(8)
            + 0.05 * (Z(8) @ Z(9))
            + -0.05 * Z(11)
            + -0.05 * Z(10)
            + 0.05 * (Z(10) @ Z(11)),
        ),
    ],
)
def test_fermi_hubbard_hamiltonian(shape, n_cells, hopping, coulomb, expected_ham):
    r"""Test that the correct Hamiltonian is generated"""

    fermi_hubbard_ham = fermi_hubbard(
        lattice=shape, n_cells=n_cells, hopping=hopping, coulomb=coulomb, neighbour_order=1
    )

    qml.assert_equal(fermi_hubbard_ham, expected_ham)


@pytest.mark.parametrize(
    # expected_ham in Jordan-Wigner transformation was obtained from openfermion and converted to
    # PennyLane format using from_openfermion
    # ham = openfermion.hamiltonians.fermi_hubbard(xaxis=n_cells[0], yaxis=n_cells[1], tunneling=hopping, coulomb=coulomb)
    # jw_ham = openfermion.transforms.jordan_wigner(ham)
    # pl_ham = qml.from_openfermion(jw_ham)
    ("shape", "n_cells", "hopping", "mapping", "expected_ham"),
    [
        (
            "chain",
            [4],
            1.0,
            "parity",
            0.5 * (Y(0) @ Y(1))
            + 0.5 * (X(0) @ X(1) @ Z(2))
            + 0.5 * (Y(1) @ Y(2))
            + 0.5 * (Z(0) @ X(1) @ X(2) @ Z(3))
            + 1.0 * I(0)
            + -0.25 * Z(0)
            + -0.25 * (Z(0) @ Z(1))
            + 0.25 * Z(1)
            + 0.5 * (Y(2) @ Y(3))
            + 0.5 * (Z(1) @ X(2) @ X(3) @ Z(4))
            + 0.5 * (Y(3) @ Y(4))
            + 0.5 * (Z(2) @ X(3) @ X(4) @ Z(5))
            + -0.25 * (Z(1) @ Z(2))
            + -0.25 * (Z(2) @ Z(3))
            + 0.25 * (Z(1) @ Z(3))
            + 0.5 * (Y(4) @ Y(5))
            + 0.5 * (Z(3) @ X(4) @ X(5) @ Z(6))
            + 0.5 * (Y(5) @ Y(6))
            + 0.5 * (Z(4) @ X(5) @ X(6) @ Z(7))
            + -0.25 * (Z(3) @ Z(4))
            + -0.25 * (Z(4) @ Z(5))
            + 0.25 * (Z(3) @ Z(5))
            + -0.25 * (Z(5) @ Z(6))
            + -0.25 * (Z(6) @ Z(7))
            + 0.25 * (Z(5) @ Z(7)),
        ),
        (
            "square",
            [2, 2],
            2.0,
            "bravyi_kitaev",
            -1.0 * (X(0) @ Y(1) @ Y(2))
            + 1.0 * (Y(0) @ Y(1) @ X(2))
            + 1.0 * (Z(0) @ X(1) @ Z(3))
            + -1.0 * (X(1) @ Z(2))
            + -1.0 * (X(0) @ X(1) @ Y(3) @ Y(4) @ X(5))
            + 1.0 * (Y(0) @ X(1) @ Y(3) @ X(4) @ X(5))
            + -1.0 * (Z(0) @ X(1) @ Y(3) @ Y(5))
            + 1.0 * (Y(1) @ Y(3) @ Z(4) @ X(5))
            + 1.0 * I(0)
            + -0.25 * (Z(0) @ Z(1))
            + -0.25 * Z(0)
            + 0.25 * Z(1)
            + -1.0 * (Z(1) @ X(2) @ Y(3) @ Z(5) @ Y(6))
            + 1.0 * (Z(1) @ Y(2) @ Y(3) @ Z(5) @ X(6))
            + 1.0 * (Z(1) @ Z(2) @ X(3) @ Z(7))
            + -1.0 * (X(3) @ Z(5) @ Z(6))
            + -0.25 * (Z(1) @ Z(2) @ Z(3))
            + -0.25 * Z(2)
            + 0.25 * (Z(1) @ Z(3))
            + -1.0 * (X(4) @ Y(5) @ Y(6))
            + 1.0 * (Y(4) @ Y(5) @ X(6))
            + 1.0 * (Z(3) @ Z(4) @ X(5) @ Z(7))
            + -1.0 * (X(5) @ Z(6))
            + -0.25 * (Z(4) @ Z(5))
            + -0.25 * Z(4)
            + 0.25 * Z(5)
            + -0.25 * (Z(3) @ Z(5) @ Z(6) @ Z(7))
            + -0.25 * Z(6)
            + 0.25 * (Z(3) @ Z(5) @ Z(7)),
        ),
    ],
)
def test_fermi_hubbard_mapping(shape, n_cells, hopping, mapping, expected_ham):
    r"""Test that the correct Hamiltonian is generated with different mappings"""

    fermi_hubbard_ham = fermi_hubbard(
        lattice=shape, n_cells=n_cells, hopping=hopping, mapping=mapping
    )

    qml.assert_equal(fermi_hubbard_ham, expected_ham)


@pytest.mark.parametrize(
    # expected_ham here was obtained manually.
    ("shape", "n_cells", "t", "coulomb", "expected_ham"),
    [
        (
            "chain",
            [4],
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
            0.1,
            -0.5 * (Y(0) @ Z(1) @ Y(2))
            + -0.5 * (X(0) @ Z(1) @ X(2))
            + 0.1 * I(0)
            + -0.5 * (Y(1) @ Z(2) @ Y(3))
            + -0.5 * (X(1) @ Z(2) @ X(3))
            + -0.5 * (Y(2) @ Z(3) @ Y(4))
            + -0.5 * (X(2) @ Z(3) @ X(4))
            + -0.5 * (Y(3) @ Z(4) @ Y(5))
            + -0.5 * (X(3) @ Z(4) @ X(5))
            + -0.5 * (Y(4) @ Z(5) @ Y(6))
            + -0.5 * (X(4) @ Z(5) @ X(6))
            + -0.5 * (Y(5) @ Z(6) @ Y(7))
            + -0.5 * (X(5) @ Z(6) @ X(7))
            + -0.025 * Z(1)
            + -0.025 * Z(0)
            + 0.025 * (Z(0) @ Z(1))
            + -0.025 * Z(3)
            + -0.025 * Z(2)
            + 0.025 * (Z(2) @ Z(3))
            + -0.025 * Z(5)
            + -0.025 * Z(4)
            + 0.025 * (Z(4) @ Z(5))
            + -0.025 * Z(7)
            + -0.025 * Z(6)
            + 0.025 * (Z(6) @ Z(7)),
        ),
        (
            "square",
            [2, 2],
            [[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]],
            [-1.0, 0.0, 1.0, 0],
            -0.25 * (Y(0) @ Z(1) @ Y(2))
            + -0.25 * (X(0) @ Z(1) @ X(2))
            + -0.25 * (Y(1) @ Z(2) @ Y(3))
            + -0.25 * (X(1) @ Z(2) @ X(3))
            + -0.25 * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
            + -0.25 * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
            + -0.25 * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
            + -0.25 * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
            + -0.25 * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + -0.25 * (X(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + -0.25 * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + -0.25 * (X(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + -0.25 * (Y(4) @ Z(5) @ Y(6))
            + -0.25 * (X(4) @ Z(5) @ X(6))
            + -0.25 * (Y(5) @ Z(6) @ Y(7))
            + -0.25 * (X(5) @ Z(6) @ X(7))
            + 0.25 * Z(1)
            + 0.25 * Z(0)
            + -0.25 * (Z(0) @ Z(1))
            + -0.25 * Z(5)
            + -0.25 * Z(4)
            + 0.25 * (Z(4) @ Z(5)),
        ),
    ],
)
def test_fermi_hubbard_hamiltonian_matrix(shape, n_cells, t, coulomb, expected_ham):
    r"""Test that the correct fermi Hubbard Hamiltonian is generated when hopping or coulomb is provided as a matrix"""

    fermi_hub_ham = fermi_hubbard(lattice=shape, n_cells=n_cells, hopping=t, coulomb=coulomb)

    qml.assert_equal(fermi_hub_ham, expected_ham)


def test_interaction_parameter_error_emery():
    r"""Test that an error is raised when the provided interaction parameters are of wrong shape for
    emery Hamiltonian."""
    n_cells = [4, 4]
    lattice = "Square"
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The hopping parameter should be a number or an array of shape (1,) or (16,16)"
        ),
    ):
        emery(lattice=lattice, n_cells=n_cells, hopping=[1.0, 2.0], neighbour_order=1)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The intersite_coupling parameter should be a number or an array of shape (1,) or (16,16)"
        ),
    ):
        emery(
            lattice=lattice,
            n_cells=n_cells,
            hopping=[1.0],
            intersite_coupling=[1.0, 2.0],
            neighbour_order=1,
        )


def test_mapping_error_emery():
    r"""Test that an error is raised when unsupported mapping is provided"""
    n_cells = [4, 4]
    lattice = "Square"
    with pytest.raises(ValueError, match="The 'bk_sf' transformation is not available."):
        emery(lattice=lattice, n_cells=n_cells, mapping="bk_sf")


@pytest.mark.parametrize(
    # expected_ham here was obtained manually
    ("shape", "n_cells", "t", "u", "v", "boundary_condition", "expected_ham"),
    [
        (
            "square",
            [2, 2],
            -1.23,
            2.34,
            1.42,
            False,
            (0.615 + 0j) * (Y(0) @ Z(1) @ Y(2))
            + (0.615 + 0j) * (X(0) @ Z(1) @ X(2))
            + (0.615 + 0j) * (Y(1) @ Z(2) @ Y(3))
            + (0.615 + 0j) * (X(1) @ Z(2) @ X(3))
            + (0.615 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
            + (0.615 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
            + (0.615 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
            + (0.615 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
            + (0.615 + 0j) * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (0.615 + 0j) * (X(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (0.615 + 0j) * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (0.615 + 0j) * (X(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (0.615 + 0j) * (Y(4) @ Z(5) @ Y(6))
            + (0.615 + 0j) * (X(4) @ Z(5) @ X(6))
            + (0.615 + 0j) * (Y(5) @ Z(6) @ Y(7))
            + (0.615 + 0j) * (X(5) @ Z(6) @ X(7))
            + (8.020000000000005 + 0j) * I(0)
            + (-2.005 + 0j) * Z(1)
            + (-2.005 + 0j) * Z(0)
            + (0.585 + 0j) * (Z(0) @ Z(1))
            + (-2.005 + 0j) * Z(3)
            + (-2.005 + 0j) * Z(2)
            + (0.585 + 0j) * (Z(2) @ Z(3))
            + (-2.005 + 0j) * Z(5)
            + (-2.005 + 0j) * Z(4)
            + (0.585 + 0j) * (Z(4) @ Z(5))
            + (-2.005 + 0j) * Z(7)
            + (-2.005 + 0j) * Z(6)
            + (0.585 + 0j) * (Z(6) @ Z(7))
            + (0.355 + 0j) * (Z(0) @ Z(2))
            + (0.355 + 0j) * (Z(0) @ Z(3))
            + (0.355 + 0j) * (Z(1) @ Z(2))
            + (0.355 + 0j) * (Z(1) @ Z(3))
            + (0.355 + 0j) * (Z(0) @ Z(4))
            + (0.355 + 0j) * (Z(0) @ Z(5))
            + (0.355 + 0j) * (Z(1) @ Z(4))
            + (0.355 + 0j) * (Z(1) @ Z(5))
            + (0.355 + 0j) * (Z(2) @ Z(6))
            + (0.355 + 0j) * (Z(2) @ Z(7))
            + (0.355 + 0j) * (Z(3) @ Z(6))
            + (0.355 + 0j) * (Z(3) @ Z(7))
            + (0.355 + 0j) * (Z(4) @ Z(6))
            + (0.355 + 0j) * (Z(4) @ Z(7))
            + (0.355 + 0j) * (Z(5) @ Z(6))
            + (0.355 + 0j) * (Z(5) @ Z(7)),
        ),
        (
            "Chain",
            [4],
            -1.23,
            2.34,
            1.42,
            True,
            (0.615 + 0j) * (Y(0) @ Z(1) @ Y(2))
            + (0.615 + 0j) * (X(0) @ Z(1) @ X(2))
            + (0.615 + 0j) * (Y(1) @ Z(2) @ Y(3))
            + (0.615 + 0j) * (X(1) @ Z(2) @ X(3))
            + (0.615 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (0.615 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (0.615 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (0.615 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (0.615 + 0j) * (Y(2) @ Z(3) @ Y(4))
            + (0.615 + 0j) * (X(2) @ Z(3) @ X(4))
            + (0.615 + 0j) * (Y(3) @ Z(4) @ Y(5))
            + (0.615 + 0j) * (X(3) @ Z(4) @ X(5))
            + (0.615 + 0j) * (Y(4) @ Z(5) @ Y(6))
            + (0.615 + 0j) * (X(4) @ Z(5) @ X(6))
            + (0.615 + 0j) * (Y(5) @ Z(6) @ Y(7))
            + (0.615 + 0j) * (X(5) @ Z(6) @ X(7))
            + (8.020000000000005 + 0j) * I(0)
            + (-2.005 + 0j) * Z(1)
            + (-2.005 + 0j) * Z(0)
            + (0.585 + 0j) * (Z(0) @ Z(1))
            + (-2.005 + 0j) * Z(3)
            + (-2.005 + 0j) * Z(2)
            + (0.585 + 0j) * (Z(2) @ Z(3))
            + (-2.005 + 0j) * Z(5)
            + (-2.005 + 0j) * Z(4)
            + (0.585 + 0j) * (Z(4) @ Z(5))
            + (-2.005 + 0j) * Z(7)
            + (-2.005 + 0j) * Z(6)
            + (0.585 + 0j) * (Z(6) @ Z(7))
            + (0.355 + 0j) * (Z(0) @ Z(2))
            + (0.355 + 0j) * (Z(0) @ Z(3))
            + (0.355 + 0j) * (Z(1) @ Z(2))
            + (0.355 + 0j) * (Z(1) @ Z(3))
            + (0.355 + 0j) * (Z(0) @ Z(6))
            + (0.355 + 0j) * (Z(0) @ Z(7))
            + (0.355 + 0j) * (Z(1) @ Z(6))
            + (0.355 + 0j) * (Z(1) @ Z(7))
            + (0.355 + 0j) * (Z(2) @ Z(4))
            + (0.355 + 0j) * (Z(2) @ Z(5))
            + (0.355 + 0j) * (Z(3) @ Z(4))
            + (0.355 + 0j) * (Z(3) @ Z(5))
            + (0.355 + 0j) * (Z(4) @ Z(6))
            + (0.355 + 0j) * (Z(4) @ Z(7))
            + (0.355 + 0j) * (Z(5) @ Z(6))
            + (0.355 + 0j) * (Z(5) @ Z(7)),
        ),
    ],
)
def test_emery_hamiltonian(shape, n_cells, t, u, v, boundary_condition, expected_ham):
    r"""Test that the correct Emery Hamiltonian is generated."""

    emery_ham = emery(
        lattice=shape,
        n_cells=n_cells,
        hopping=t,
        coulomb=u,
        intersite_coupling=v,
        boundary_condition=boundary_condition,
    )

    qml.assert_equal(emery_ham, expected_ham)


@pytest.mark.parametrize(
    # expected_ham here was obtained manually.
    ("shape", "n_cells", "t", "u", "v", "expected_ham"),
    [
        (
            "chain",
            [4],
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
            0.1,
            0,
            -0.5 * (Y(0) @ Z(1) @ Y(2))
            + -0.5 * (X(0) @ Z(1) @ X(2))
            + 0.1 * I(0)
            + -0.5 * (Y(1) @ Z(2) @ Y(3))
            + -0.5 * (X(1) @ Z(2) @ X(3))
            + -0.5 * (Y(2) @ Z(3) @ Y(4))
            + -0.5 * (X(2) @ Z(3) @ X(4))
            + -0.5 * (Y(3) @ Z(4) @ Y(5))
            + -0.5 * (X(3) @ Z(4) @ X(5))
            + -0.5 * (Y(4) @ Z(5) @ Y(6))
            + -0.5 * (X(4) @ Z(5) @ X(6))
            + -0.5 * (Y(5) @ Z(6) @ Y(7))
            + -0.5 * (X(5) @ Z(6) @ X(7))
            + -0.025 * Z(1)
            + -0.025 * Z(0)
            + 0.025 * (Z(0) @ Z(1))
            + -0.025 * Z(3)
            + -0.025 * Z(2)
            + 0.025 * (Z(2) @ Z(3))
            + -0.025 * Z(5)
            + -0.025 * Z(4)
            + 0.025 * (Z(4) @ Z(5))
            + -0.025 * Z(7)
            + -0.025 * Z(6)
            + 0.025 * (Z(6) @ Z(7)),
        ),
        (
            "square",
            [2, 2],
            [[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]],
            [-1.0, 0.0, 1.0, 0],
            0,
            -0.25 * (Y(0) @ Z(1) @ Y(2))
            + -0.25 * (X(0) @ Z(1) @ X(2))
            + -0.25 * (Y(1) @ Z(2) @ Y(3))
            + -0.25 * (X(1) @ Z(2) @ X(3))
            + -0.25 * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
            + -0.25 * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
            + -0.25 * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
            + -0.25 * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
            + -0.25 * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + -0.25 * (X(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + -0.25 * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + -0.25 * (X(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + -0.25 * (Y(4) @ Z(5) @ Y(6))
            + -0.25 * (X(4) @ Z(5) @ X(6))
            + -0.25 * (Y(5) @ Z(6) @ Y(7))
            + -0.25 * (X(5) @ Z(6) @ X(7))
            + 0.25 * Z(1)
            + 0.25 * Z(0)
            + -0.25 * (Z(0) @ Z(1))
            + -0.25 * Z(5)
            + -0.25 * Z(4)
            + 0.25 * (Z(4) @ Z(5)),
        ),
        (
            "square",
            [2, 2],
            0.1,
            [-1.0, 0.0, 1.0, 0],
            [[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]],
            -0.05 * (Y(0) @ Z(1) @ Y(2))
            + -0.05 * (X(0) @ Z(1) @ X(2))
            + -0.05 * (Y(1) @ Z(2) @ Y(3))
            + -0.05 * (X(1) @ Z(2) @ X(3))
            + -0.05 * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
            + -0.05 * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
            + -0.05 * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
            + -0.05 * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
            + -0.05 * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + -0.05 * (X(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + -0.05 * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + -0.05 * (X(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + -0.05 * (Y(4) @ Z(5) @ Y(6))
            + -0.05 * (X(4) @ Z(5) @ X(6))
            + -0.05 * (Y(5) @ Z(6) @ Y(7))
            + -0.05 * (X(5) @ Z(6) @ X(7))
            + 2.0 * I(0)
            + -0.25 * Z(1)
            + -0.25 * Z(0)
            + -0.25 * (Z(0) @ Z(1))
            + -0.5 * Z(3)
            + -0.5 * Z(2)
            + -0.75 * Z(5)
            + -0.75 * Z(4)
            + 0.25 * (Z(4) @ Z(5))
            + -0.5 * Z(7)
            + -0.5 * Z(6)
            + 0.125 * (Z(0) @ Z(2))
            + 0.125 * (Z(0) @ Z(3))
            + 0.125 * (Z(1) @ Z(2))
            + 0.125 * (Z(1) @ Z(3))
            + 0.125 * (Z(0) @ Z(4))
            + 0.125 * (Z(0) @ Z(5))
            + 0.125 * (Z(1) @ Z(4))
            + 0.125 * (Z(1) @ Z(5))
            + 0.125 * (Z(2) @ Z(6))
            + 0.125 * (Z(2) @ Z(7))
            + 0.125 * (Z(3) @ Z(6))
            + 0.125 * (Z(3) @ Z(7))
            + 0.125 * (Z(4) @ Z(6))
            + 0.125 * (Z(4) @ Z(7))
            + 0.125 * (Z(5) @ Z(6))
            + 0.125 * (Z(5) @ Z(7)),
        ),
    ],
)
def test_emery_hamiltonian_matrix(shape, n_cells, t, u, v, expected_ham):
    r"""Test that the correct Emery Hamiltonian is generated when interaction parameters are provided as a matrix"""

    emery_ham = emery(lattice=shape, n_cells=n_cells, hopping=t, coulomb=u, intersite_coupling=v)

    qml.assert_equal(emery_ham, expected_ham)


def test_hopping_error_haldane():
    r"""Test that an error is raised when the shape of provided interaction parameters is wrong for
    Haldane Hamiltonian."""
    n_cells = [4, 4]
    lattice = "Square"
    with pytest.raises(
        ValueError,
        match=re.escape("The hopping parameter should be a constant or an array of shape (16,16)"),
    ):
        haldane(lattice=lattice, n_cells=n_cells, hopping=[1.0, 2.0])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The hopping_next parameter should be a constant or an array of shape (16,16)"
        ),
    ):
        haldane(lattice=lattice, n_cells=n_cells, hopping=1.0, hopping_next=[0.5, 0.6, 0.7])

    with pytest.raises(
        ValueError,
        match=re.escape("The phi parameter should be a constant or an array of shape (16,16)"),
    ):
        haldane(lattice=lattice, n_cells=n_cells, hopping=1.0, hopping_next=0.1, phi=[0.5, 0.6])


def test_mapping_error_haldane():
    r"""Test that an error is raised when unsupported mapping is provided"""
    n_cells = [4, 4]
    lattice = "Square"
    with pytest.raises(ValueError, match="The 'bk_sf' transformation is not available."):
        haldane(lattice=lattice, n_cells=n_cells, mapping="bk_sf")


@pytest.mark.parametrize(
    # expected_ham here was obtained manually.
    ("shape", "n_cells", "t1", "t2", "phi", "boundary_condition", "expected_ham"),
    [
        (
            "chain",
            [4],
            -1.23,
            2.34,
            1.0,
            True,
            (0.615 + 0j) * (Y(0) @ Z(1) @ Y(2))
            + (0.615 + 0j) * (X(0) @ Z(1) @ X(2))
            + (0.615 + 0j) * (Y(1) @ Z(2) @ Y(3))
            + (0.615 + 0j) * (X(1) @ Z(2) @ X(3))
            + (0.615 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (0.615 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (0.615 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (0.615 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (0.615 + 0j) * (Y(2) @ Z(3) @ Y(4))
            + (0.615 + 0j) * (X(2) @ Z(3) @ X(4))
            + (0.615 + 0j) * (Y(3) @ Z(4) @ Y(5))
            + (0.615 + 0j) * (X(3) @ Z(4) @ X(5))
            + (0.615 + 0j) * (Y(4) @ Z(5) @ Y(6))
            + (0.615 + 0j) * (X(4) @ Z(5) @ X(6))
            + (0.615 + 0j) * (Y(5) @ Z(6) @ Y(7))
            + (0.615 + 0j) * (X(5) @ Z(6) @ X(7))
            + (-0.9845210522252389 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
            + (-0.6321536978657235 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
            + (-0.6321536978657235 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
            + (0.9845210522252389 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
            + (-0.9845210522252389 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
            + (-0.6321536978657235 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
            + (-0.6321536978657235 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
            + (0.9845210522252389 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
            + (-0.9845210522252389 + 0j) * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (-0.6321536978657235 + 0j) * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (-0.6321536978657235 + 0j) * (X(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (0.9845210522252389 + 0j) * (X(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (-0.9845210522252389 + 0j) * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (-0.6321536978657235 + 0j) * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (-0.6321536978657235 + 0j) * (X(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (0.9845210522252389 + 0j) * (X(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7)),
        ),
        (
            "square",
            [2, 2],
            -1.23,
            2.34,
            [
                [0.0, 1.0, 1.0, 1.41421356],
                [1.0, 0.0, 1.41421356, 1.0],
                [1.0, 1.41421356, 0.0, 1.0],
                [1.41421356, 1.0, 1.0, 0.0],
            ],
            False,
            (0.615 + 0j) * (Y(0) @ Z(1) @ Y(2))
            + (0.615 + 0j) * (X(0) @ Z(1) @ X(2))
            + (0.615 + 0j) * (Y(1) @ Z(2) @ Y(3))
            + (0.615 + 0j) * (X(1) @ Z(2) @ X(3))
            + (0.615 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
            + (0.615 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
            + (0.615 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
            + (0.615 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
            + (0.615 + 0j) * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (0.615 + 0j) * (X(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (0.615 + 0j) * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (0.615 + 0j) * (X(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (0.615 + 0j) * (Y(4) @ Z(5) @ Y(6))
            + (0.615 + 0j) * (X(4) @ Z(5) @ X(6))
            + (0.615 + 0j) * (Y(5) @ Z(6) @ Y(7))
            + (0.615 + 0j) * (X(5) @ Z(6) @ X(7))
            + (-1.1556861568115007 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (-0.182454122875488 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (-0.182454122875488 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (1.1556861568115007 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (-1.1556861568115007 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (-0.182454122875488 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (-0.182454122875488 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (1.1556861568115007 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (-1.1556861568115007 + 0j) * (Y(2) @ Z(3) @ X(4))
            + (-0.182454122875488 + 0j) * (Y(2) @ Z(3) @ Y(4))
            + (-0.182454122875488 + 0j) * (X(2) @ Z(3) @ X(4))
            + (1.1556861568115007 + 0j) * (X(2) @ Z(3) @ Y(4))
            + (-1.1556861568115007 + 0j) * (Y(3) @ Z(4) @ X(5))
            + (-0.182454122875488 + 0j) * (Y(3) @ Z(4) @ Y(5))
            + (-0.182454122875488 + 0j) * (X(3) @ Z(4) @ X(5))
            + (1.1556861568115007 + 0j) * (X(3) @ Z(4) @ Y(5)),
        ),
        (
            "square",
            [2, 2],
            [[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]],
            1.0,
            [
                [0.0, 1.0, 1.0, 1.41421356],
                [1.0, 0.0, 1.41421356, 1.0],
                [1.0, 1.41421356, 0.0, 1.0],
                [1.41421356, 1.0, 1.0, 0.0],
            ],
            False,
            (-0.25 + 0j) * (Y(0) @ Z(1) @ Y(2))
            + (-0.25 + 0j) * (X(0) @ Z(1) @ X(2))
            + (-0.25 + 0j) * (Y(1) @ Z(2) @ Y(3))
            + (-0.25 + 0j) * (X(1) @ Z(2) @ X(3))
            + (-0.25 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
            + (-0.25 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
            + (-0.25 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
            + (-0.25 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
            + (-0.25 + 0j) * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (-0.25 + 0j) * (X(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (-0.25 + 0j) * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (-0.25 + 0j) * (X(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (-0.25 + 0j) * (Y(4) @ Z(5) @ Y(6))
            + (-0.25 + 0j) * (X(4) @ Z(5) @ X(6))
            + (-0.25 + 0j) * (Y(5) @ Z(6) @ Y(7))
            + (-0.25 + 0j) * (X(5) @ Z(6) @ X(7))
            + (-0.4938829729963678 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (-0.07797184738268718 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (-0.07797184738268718 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (0.4938829729963678 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (-0.4938829729963678 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (-0.07797184738268718 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (-0.07797184738268718 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (0.4938829729963678 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (-0.4938829729963678 + 0j) * (Y(2) @ Z(3) @ X(4))
            + (-0.07797184738268718 + 0j) * (Y(2) @ Z(3) @ Y(4))
            + (-0.07797184738268718 + 0j) * (X(2) @ Z(3) @ X(4))
            + (0.4938829729963678 + 0j) * (X(2) @ Z(3) @ Y(4))
            + (-0.4938829729963678 + 0j) * (Y(3) @ Z(4) @ X(5))
            + (-0.07797184738268718 + 0j) * (Y(3) @ Z(4) @ Y(5))
            + (-0.07797184738268718 + 0j) * (X(3) @ Z(4) @ X(5))
            + (0.4938829729963678 + 0j) * (X(3) @ Z(4) @ Y(5)),
        ),
        (
            "square",
            [2, 2],
            [[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]],
            [[0, 0, 0, 0.5], [0, 0, 0.5, 0], [0, 0.5, 0, 0], [0.5, 0.0, 0.0, 0]],
            [
                [0.0, 1.0, 1.0, 1.41421356],
                [1.0, 0.0, 1.41421356, 1.0],
                [1.0, 1.41421356, 0.0, 1.0],
                [1.41421356, 1.0, 1.0, 0.0],
            ],
            False,
            (-0.25 + 0j) * (Y(0) @ Z(1) @ Y(2))
            + (-0.25 + 0j) * (X(0) @ Z(1) @ X(2))
            + (-0.25 + 0j) * (Y(1) @ Z(2) @ Y(3))
            + (-0.25 + 0j) * (X(1) @ Z(2) @ X(3))
            + (-0.25 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
            + (-0.25 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
            + (-0.25 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
            + (-0.25 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
            + (-0.25 + 0j) * (Y(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (-0.25 + 0j) * (X(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (-0.25 + 0j) * (Y(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (-0.25 + 0j) * (X(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (-0.25 + 0j) * (Y(4) @ Z(5) @ Y(6))
            + (-0.25 + 0j) * (X(4) @ Z(5) @ X(6))
            + (-0.25 + 0j) * (Y(5) @ Z(6) @ Y(7))
            + (-0.25 + 0j) * (X(5) @ Z(6) @ X(7))
            + (-0.2469414864981839 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (-0.03898592369134359 + 0j) * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (-0.03898592369134359 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ X(6))
            + (0.2469414864981839 + 0j) * (X(0) @ Z(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Y(6))
            + (-0.2469414864981839 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (-0.03898592369134359 + 0j) * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (-0.03898592369134359 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ X(7))
            + (0.2469414864981839 + 0j) * (X(1) @ Z(2) @ Z(3) @ Z(4) @ Z(5) @ Z(6) @ Y(7))
            + (-0.2469414864981839 + 0j) * (Y(2) @ Z(3) @ X(4))
            + (-0.03898592369134359 + 0j) * (Y(2) @ Z(3) @ Y(4))
            + (-0.03898592369134359 + 0j) * (X(2) @ Z(3) @ X(4))
            + (0.2469414864981839 + 0j) * (X(2) @ Z(3) @ Y(4))
            + (-0.2469414864981839 + 0j) * (Y(3) @ Z(4) @ X(5))
            + (-0.03898592369134359 + 0j) * (Y(3) @ Z(4) @ Y(5))
            + (-0.03898592369134359 + 0j) * (X(3) @ Z(4) @ X(5))
            + (0.2469414864981839 + 0j) * (X(3) @ Z(4) @ Y(5)),
        ),
    ],
)
def test_haldane_hamiltonian_matrix(shape, n_cells, t1, t2, phi, boundary_condition, expected_ham):
    r"""Test that the correct Haldane Hamiltonian is generated."""

    haldane_ham = haldane(
        lattice=shape,
        n_cells=n_cells,
        hopping=t1,
        hopping_next=t2,
        phi=phi,
        boundary_condition=boundary_condition,
    )

    qml.assert_equal(haldane_ham, expected_ham)


def test_coupling_error_kitaev():
    r"""Test that an error is raised when the provided coupling shape is wrong for
    Kitaev Hamiltonian."""
    with pytest.raises(
        ValueError,
        match=re.escape("The coupling parameter should be a list of length 3."),
    ):
        kitaev(n_cells=[3, 4], coupling=[1.0, 2.0])


@pytest.mark.parametrize(
    # expected_ham here was obtained manually
    ("n_cells", "j", "boundary_condition", "expected_ham"),
    [
        (
            [2, 2, 1],
            None,
            False,
            1.0 * (Z(1) @ Z(4))
            + 1.0 * (Z(3) @ Z(6))
            + 1.0 * (X(0) @ X(1))
            + 1.0 * (X(2) @ X(3))
            + 1.0 * (X(4) @ X(5))
            + 1.0 * (X(6) @ X(7))
            + 1.0 * (Y(1) @ Y(2))
            + 1.0 * (Y(5) @ Y(6)),
        ),
        (
            [2, 2],
            [0.5, 0.6, 0.7],
            False,
            0.7 * (Z(1) @ Z(4))
            + 0.7 * (Z(3) @ Z(6))
            + 0.5 * (X(0) @ X(1))
            + 0.5 * (X(2) @ X(3))
            + 0.5 * (X(4) @ X(5))
            + 0.5 * (X(6) @ X(7))
            + 0.6 * (Y(1) @ Y(2))
            + 0.6 * (Y(5) @ Y(6)),
        ),
        (
            [2, 3],
            [0.1, 0.2, 0.3],
            True,
            0.3 * (Z(1) @ Z(6))
            + 0.3 * (Z(3) @ Z(8))
            + 0.3 * (Z(5) @ Z(10))
            + 0.3 * (Z(0) @ Z(7))
            + 0.3 * (Z(2) @ Z(9))
            + 0.3 * (Z(4) @ Z(11))
            + 0.1 * (X(0) @ X(1))
            + 0.1 * (X(2) @ X(3))
            + 0.1 * (X(4) @ X(5))
            + 0.1 * (X(6) @ X(7))
            + 0.1 * (X(8) @ X(9))
            + 0.1 * (X(10) @ X(11))
            + 0.2 * (Y(1) @ Y(2))
            + 0.2 * (Y(3) @ Y(4))
            + 0.2 * (Y(0) @ Y(5))
            + 0.2 * (Y(7) @ Y(8))
            + 0.2 * (Y(9) @ Y(10))
            + 0.2 * (Y(11) @ Y(6)),
        ),
        (
            [2, 3],
            [0.1, 0.2, 0.3],
            [True, False],
            0.3 * (Z(1) @ Z(6))
            + 0.3 * (Z(3) @ Z(8))
            + 0.3 * (Z(5) @ Z(10))
            + 0.3 * (Z(0) @ Z(7))
            + 0.3 * (Z(2) @ Z(9))
            + 0.3 * (Z(4) @ Z(11))
            + 0.1 * (X(0) @ X(1))
            + 0.1 * (X(2) @ X(3))
            + 0.1 * (X(4) @ X(5))
            + 0.1 * (X(6) @ X(7))
            + 0.1 * (X(8) @ X(9))
            + 0.1 * (X(10) @ X(11))
            + 0.2 * (Y(1) @ Y(2))
            + 0.2 * (Y(3) @ Y(4))
            + 0.2 * (Y(7) @ Y(8))
            + 0.2 * (Y(9) @ Y(10)),
        ),
    ],
)
def test_kitaev_hamiltonian(n_cells, j, boundary_condition, expected_ham):
    r"""Test that the correct Hamiltonian is generated"""
    kitaev_ham = kitaev(n_cells=n_cells, coupling=j, boundary_condition=boundary_condition)

    qml.assert_equal(kitaev_ham, expected_ham)


@pytest.mark.parametrize(
    ("lattice", "expected_ham"),
    [
        # This is the Hamiltonian for the Kitaev model on the Honeycomb lattice
        (
            Lattice(
                n_cells=[2, 2],
                vectors=[[1, 0], [0, 1]],
                positions=[[0, 0], [1, 5]],
                boundary_condition=False,
                custom_edges=[[(0, 1), ("XX", 0.5)], [(1, 2), ("YY", 0.6)], [(1, 4), ("ZZ", 0.7)]],
            ),
            (
                0.5 * (X(0) @ X(1))
                + 0.5 * (X(2) @ X(3))
                + 0.5 * (X(4) @ X(5))
                + 0.5 * (X(6) @ X(7))
                + 0.6 * (Y(1) @ Y(2))
                + 0.6 * (Y(5) @ Y(6))
                + 0.7 * (Z(1) @ Z(4))
                + 0.7 * (Z(3) @ Z(6))
            ),
        ),
        (
            Lattice(
                n_cells=[2, 2],
                vectors=[[1, 0], [0, 1]],
                positions=[[0, 0], [1, 5]],
                boundary_condition=False,
                custom_edges=[[(0, 1), ("XX", 0.5)], [(1, 2), ("YY", 0.6)], [(1, 4), ("ZZ", 0.7)]],
                custom_nodes=[[0, ("X", 0.3)], [7, ("Y", 0.9)]],
            ),
            (
                0.5 * (X(0) @ X(1))
                + 0.5 * (X(2) @ X(3))
                + 0.5 * (X(4) @ X(5))
                + 0.5 * (X(6) @ X(7))
                + 0.6 * (Y(1) @ Y(2))
                + 0.6 * (Y(5) @ Y(6))
                + 0.7 * (Z(1) @ Z(4))
                + 0.7 * (Z(3) @ Z(6))
                + 0.3 * X(0)
                + 0.9 * Y(7)
            ),
        ),
        (
            Lattice(
                n_cells=[2, 2],
                vectors=[[1, 0], [0, 1]],
                positions=[[0, 0], [1, 5]],
                boundary_condition=False,
                custom_edges=[[(0, 1), ("XX", 0.5)], [(1, 2), ("YY", 0.6)], [(1, 4), ("ZZ", 0.7)]],
                custom_nodes=[[0, ("X", 0.3)], [7, ("Y", 0.9)], [0, ("X", 0.5)]],
            ),
            (
                0.5 * (X(0) @ X(1))
                + 0.5 * (X(2) @ X(3))
                + 0.5 * (X(4) @ X(5))
                + 0.5 * (X(6) @ X(7))
                + 0.6 * (Y(1) @ Y(2))
                + 0.6 * (Y(5) @ Y(6))
                + 0.7 * (Z(1) @ Z(4))
                + 0.7 * (Z(3) @ Z(6))
                + 0.8 * X(0)
                + 0.9 * Y(7)
            ),
        ),
    ],
)
def test_spin_hamiltonian(lattice, expected_ham):
    r"""Test that the correct Hamiltonian is generated from a given Lattice"""
    spin_ham = spin_hamiltonian(lattice=lattice)

    qml.assert_equal(spin_ham, expected_ham)


def test_spin_hamiltonian_error():
    r"""Test that the correct error is raised Hamiltonian with incompatible Lattice"""
    lattice = Lattice(n_cells=[2, 2], vectors=[[1, 0], [0, 1]], positions=[[0, 0], [1, 1]])
    with pytest.raises(
        ValueError,
        match="Custom edges need to be defined and should have an operator defined as a `str`",
    ):
        spin_hamiltonian(lattice=lattice)

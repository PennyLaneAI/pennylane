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
Unit tests for functions needed for computing the spin Hamiltonians.
"""

import pytest

import pennylane as qml
from pennylane.spin import transverse_ising


def test_coupling_error():
    r"""Test that an error is raised when the provided coupling shape is wrong"""
    n_cells = [4, 4]
    lattice = "Square"
    with pytest.raises(ValueError, match="Coupling shape should be 1 or 16x16"):
        transverse_ising(lattice=lattice, n_cells=n_cells, coupling=1.0)


def test_shape_error():
    r"""Test that an error is raised if wrong shape is provided"""
    n_cells = [5, 5, 5]
    lattice = "Octagon"
    with pytest.raises(ValueError, match="Lattice shape, 'Octagon' is not supported."):
        transverse_ising(lattice=lattice, n_cells=n_cells, coupling=1.0)


@pytest.mark.parametrize(
    ("shape_ds", "shape_lattice", "layout", "n_cells"),
    [
        ("chain", "chain", "1x4", [4, 0, 0]),
        ("chain", "chain", "1x8", [8, 0, 0]),
        ("rectangular", "rectangle", "2x4", [4, 2, 0]),
        ("rectangular", "rectangle", "2x8", [8, 2, 0]),
    ],
)
def test_ising_hamiltonian(shape_ds, shape_lattice, layout, n_cells):
    r"""Test that the correct Hamiltonian is generated compared to the datasets"""
    spin_dataset = qml.data.load("qspin", sysname="Ising", lattice=shape_ds, layout=layout)
    dataset_ham = spin_dataset[0].hamiltonians[0]

    J = [-spin_dataset[0].parameters["J"]]
    h = -spin_dataset[0].parameters["h"][0]

    ising_ham = transverse_ising(
        lattice=shape_lattice, n_cells=n_cells, coupling=J, h=h, neighbour_order=1
    )

    assert qml.equal(ising_ham, dataset_ham)

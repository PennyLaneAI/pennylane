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
"""
Unit tests for functions needed for computing the dipole.
"""
import autograd
import pennylane as qml
import pytest
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import numpy as np
from pennylane.hf.dipole import (
    generate_dipole,
    generate_dipole_integrals,
    generate_fermionic_dipole,
    one_particle,
    qubit_operator,
)
from pennylane.hf.molecule import Molecule


@pytest.mark.parametrize(
    ("symbols", "geometry", "charge", "core", "active", "core_ref", "int_ref"),
    [
        (
            ["H", "H", "H"],
            np.array(
                [[0.028, 0.054, 0.0], [0.986, 1.610, 0.0], [1.855, 0.002, 0.0]], requires_grad=False
            ),
            1,
            None,
            None,
            [2.869, 1.666, 0.000],  # computed with PL-QChem dipole function
            # computed with PL-QChem dipole function using OpenFermion and PySCF
            np.array(
                [
                    [
                        [0.95622463, 0.7827277, -0.53222294],
                        [0.7827277, 1.42895581, 0.23469918],
                        [-0.53222294, 0.23469918, 0.48381955],
                    ],
                    [
                        [0.55538736, -0.53229398, -0.78262324],
                        [-0.53229398, 0.3203965, 0.47233426],
                        [-0.78262324, 0.47233426, 0.79021614],
                    ],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
        ),
        (
            ["H", "H", "H"],
            np.array(
                [[0.028, 0.054, 0.0], [0.986, 1.610, 0.0], [1.855, 0.002, 0.0]], requires_grad=True
            ),
            1,
            [],
            [0, 1, 2],
            [2.869, 1.666, 0.000],  # computed with PL-QChem dipole function
            # computed with PL-QChem dipole function using OpenFermion and PySCF
            np.array(
                [
                    [
                        [0.95622463, 0.7827277, -0.53222294],
                        [0.7827277, 1.42895581, 0.23469918],
                        [-0.53222294, 0.23469918, 0.48381955],
                    ],
                    [
                        [0.55538736, -0.53229398, -0.78262324],
                        [-0.53229398, 0.3203965, 0.47233426],
                        [-0.78262324, 0.47233426, 0.79021614],
                    ],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
        ),
    ],
)
def test_generate_dipole_integrals(symbols, geometry, core, charge, active, core_ref, int_ref):
    r"""Test that generate_electron_integrals returns the correct values."""
    mol = Molecule(symbols, geometry, charge=charge)
    args = [p for p in [geometry] if p.requires_grad]
    constants, integrals = generate_dipole_integrals(mol, core=core, active=active)(*args)

    for i in range(3):  # loop on x, y, z components
        assert np.allclose(constants[i], core_ref[i])
        assert np.allclose(integrals[i], int_ref[i])


@pytest.mark.parametrize(
    ("symbols", "geometry", "charge", "core", "active", "f_ref"),
    [
        (
            ["H", "H", "H"],
            np.array(
                [[0.028, 0.054, 0.0], [0.986, 1.610, 0.0], [1.855, 0.002, 0.0]], requires_grad=False
            ),
            1,
            None,
            None,
            # x component of fermionic dipole computed with PL-QChem dipole (format is modified)
            (
                np.array(
                    [
                        -2.869,
                        0.956224634652776,
                        0.782727697897828,
                        -0.532222940905614,
                        0.956224634652776,
                        0.782727697897828,
                        -0.532222940905614,
                        0.782727697897828,
                        1.42895581236226,
                        0.234699175620383,
                        0.782727697897828,
                        1.42895581236226,
                        0.234699175620383,
                        -0.532222940905614,
                        0.234699175620383,
                        0.483819552892797,
                        -0.532222940905614,
                        0.234699175620383,
                        0.483819552892797,
                    ]
                ),
                [
                    [],
                    [0, 0],
                    [0, 2],
                    [0, 4],
                    [1, 1],
                    [1, 3],
                    [1, 5],
                    [2, 0],
                    [2, 2],
                    [2, 4],
                    [3, 1],
                    [3, 3],
                    [3, 5],
                    [4, 0],
                    [4, 2],
                    [4, 4],
                    [5, 1],
                    [5, 3],
                    [5, 5],
                ],
            ),
        ),
    ],
)
def test_generate_fermionic_dipole(symbols, geometry, core, charge, active, f_ref):
    r"""Test that generate_electron_integrals returns the correct values."""
    mol = Molecule(symbols, geometry, charge=charge)
    args = [p for p in [geometry] if p.requires_grad]
    f = generate_fermionic_dipole(mol, core=core, active=active)(*args)[0]

    assert np.allclose(f[0], f_ref[0])  # fermionic coefficients
    assert np.allclose(f[0], f_ref[0])  # fermionic operators


@pytest.mark.parametrize(
    ("symbols", "geometry", "charge", "core", "active", "n_terms"),
    [
        (
            ["H", "H", "H"],
            np.array(
                [[0.028, 0.054, 0.0], [0.986, 1.610, 0.0], [1.855, 0.002, 0.0]], requires_grad=False
            ),
            1,
            None,
            None,
            # number of terms in the x component of the fermionic dipole computed with PL-QChem
            18,
        ),
    ],
)
def test_generate_dipole(symbols, geometry, core, charge, active, n_terms):
    r"""Test that generate_electron_integrals returns the correct values."""
    mol = Molecule(symbols, geometry, charge=charge)
    args = [p for p in [geometry] if p.requires_grad]
    f = generate_dipole(mol, core=core, active=active, cutoff=1.0e-8)(*args)[0]

    assert len(f.terms[0]) == n_terms

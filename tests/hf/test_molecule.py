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
"""
Unit tests for the molecule object.
"""
# pylint: disable=no-self-use
import pytest
from pennylane import numpy as np
from pennylane.hf.molecule import Molecule, generate_basis_set, generate_nuclear_charges
from pennylane.hf.basis_set import BasisFunction


class TestMolecule:
    """Tests for generating a molecule object."""

    @pytest.mark.parametrize(
        ("symbols", "geometry"),
        [
            (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
        ],
    )
    def test_build_molecule(self, symbols, geometry):
        r"""Test that the generated molecule object has the correct type."""
        mol = Molecule(symbols, geometry)
        assert isinstance(mol, Molecule)

    @pytest.mark.parametrize(
        ("symbols", "geometry"),
        [
            (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
        ],
    )
    def test_basis_error(self, symbols, geometry):
        r"""Test that an error is raised if a wrong basis set name is entered."""
        with pytest.raises(ValueError, match="The only supported basis set is"):
            Molecule(symbols, geometry, basis_name="6-31g")

    @pytest.mark.parametrize(
        ("symbols", "geometry"),
        [
            (["H", "Og"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
        ],
    )
    def test_symbol_error(self, symbols, geometry):
        r"""Test that an error is raised if a wrong/not-supported atomic symbol is entered."""
        with pytest.raises(ValueError, match="are not supported"):
            Molecule(symbols, geometry)

    @pytest.mark.parametrize(
        ("symbols", "geometry", "charge", "mult", "basis_name"),
        [
            (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 0, 1, "sto-3g"),
        ],
    )
    def test_default_inputs(self, symbols, geometry, charge, mult, basis_name):
        r"""Test that the molecule object contains correct default molecular input data."""
        mol = Molecule(symbols, geometry)

        assert mol.symbols == symbols
        assert np.allclose(mol.coordinates, geometry)
        assert mol.charge == charge
        assert mol.mult == mult
        assert mol.basis_name == basis_name

    @pytest.mark.parametrize(
        ("symbols", "geometry", "n_electrons", "n_orbitals", "nuclear_charges", "core", "active"),
        [
            (
                ["H", "F"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                10,
                6,
                [1, 9],
                [],
                [0, 1, 2, 3, 4, 5],
            ),
        ],
    )
    def test_mol_data(
        self, symbols, geometry, n_electrons, n_orbitals, nuclear_charges, core, active
    ):
        r"""Test that the molecule object computes correct molecular data."""
        mol = Molecule(symbols, geometry)

        assert mol.n_electrons == n_electrons
        assert mol.n_orbitals == n_orbitals
        assert mol.nuclear_charges == nuclear_charges
        assert mol.core == core
        assert mol.active == active

    data_H2 = [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, -0.694349], [0.0, 0.0, 0.694349]]),
            [1, 1],
            (
                (
                    (0, 0, 0),
                    [3.42525091, 0.62391373, 0.1688554],
                    [0.15432897, 0.53532814, 0.44463454],
                ),
                (
                    (0, 0, 0),
                    [3.42525091, 0.62391373, 0.1688554],
                    [0.15432897, 0.53532814, 0.44463454],
                ),
            ),
        )
    ]

    @pytest.mark.parametrize("molecular_data", data_H2)
    def test_molecule_basisdata(self, molecular_data):
        r"""Test that correct atomic nuclear charges are generated for a given molecule."""
        symbols, geometry = molecular_data[0], molecular_data[1]
        n_basis_H2 = molecular_data[2]
        basis_data_H2 = molecular_data[3]

        mol = Molecule(symbols, geometry)

        assert mol.n_basis == n_basis_H2
        assert np.allclose(mol.basis_data, basis_data_H2)

    basis_data_H2 = [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, -0.694349], [0.0, 0.0, 0.694349]]),
            [(0, 0, 0), (0, 0, 0)],
            [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
            [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
            [[0.0, 0.0, -0.694349], [0.0, 0.0, 0.694349]],
        )
    ]

    @pytest.mark.parametrize("molecular_data", basis_data_H2)
    def test_molecule_basisset(self, molecular_data):
        r"""Test that correct basis set is generated for a given molecule."""
        symbols, geometry = molecular_data[0], molecular_data[1]
        mol = Molecule(symbols, geometry)

        assert mol.l == molecular_data[2]
        assert np.allclose(mol.alpha, molecular_data[3])
        assert np.allclose(mol.coeff, molecular_data[4])
        assert np.allclose(mol.r, molecular_data[5])

        assert isinstance(mol.basis_set[0], BasisFunction)
        assert isinstance(mol.basis_set[1], BasisFunction)

    molecular_symbols = [(["H", "H"], [1, 1]), (["H", "F"], [1, 9]), (["F", "C", "N"], [9, 6, 7])]

    @pytest.mark.parametrize("molecular_symbols", molecular_symbols)
    def test_generate_nuclear_charges(self, molecular_symbols):
        r"""Test that correct atomic nuclear charges are generated for a given molecule."""
        symbols, charges = molecular_symbols

        assert generate_nuclear_charges(symbols) == charges

    basis_data_H2 = [
        (
            [(0, 0, 0), (0, 0, 0)],
            [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
            [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
            [[0.0, 0.0, -0.694349], [0.0, 0.0, 0.694349]],
        )
    ]

    @pytest.mark.parametrize("basis_data", basis_data_H2)
    def test_generate_basis_set(self, basis_data):
        r"""Test that correct atomic nuclear charges are generated for a given molecule."""
        l, alpha, coeff, r = basis_data
        basis_set = generate_basis_set(l, alpha, coeff, r)

        assert np.allclose(basis_set[0].l, l[0])
        assert np.allclose(basis_set[1].l, l[1])

        assert np.allclose(basis_set[0].alpha, alpha[0])
        assert np.allclose(basis_set[1].alpha, alpha[1])

        assert np.allclose(basis_set[0].coeff, coeff[0])
        assert np.allclose(basis_set[1].coeff, coeff[1])

        assert np.allclose(basis_set[0].r, r[0])
        assert np.allclose(basis_set[1].r, r[1])

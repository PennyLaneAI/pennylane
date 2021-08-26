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
from molecule import Molecule, generate_basis_set, generate_nuclear_charges


molecular_data_ref = [(["H", "H"], np.array([[0.0, 0.0, -0.694349], [0.0, 0.0, 0.694349]]))]


class TestMolecule:
    """Tests for generating a molecule object."""

    @pytest.mark.parametrize("molecular_data", molecular_data_ref)
    def test_build_molecule(self, molecular_data):
        r"""Test that the generated molecule object has the correct type."""
        symbols, geometry = molecular_data[0], molecular_data[1]
        mol = Molecule(symbols, geometry)

        assert isinstance(mol, Molecule)

    @pytest.mark.parametrize("molecular_data", molecular_data_ref)
    def test_molecular_prop(self, molecular_data):
        r"""Test that the molecule object contains correct molecular properties."""
        symbols, geometry = molecular_data[0], molecular_data[1]
        mol = Molecule(symbols, geometry)

        assert mol.symbols == symbols
        assert np.allclose(mol.coordinates, geometry)
        assert mol.charge == 0
        assert mol.mult == 1
        assert mol.basis_name == "sto-3g"

    @pytest.mark.parametrize("molecular_data", molecular_data_ref)
    def test_molecular_error(self, molecular_data):
        r"""Test that an error is raised if a wrong basis set name is entered."""
        symbols, geometry = molecular_data[0], molecular_data[1]

        with pytest.raises(ValueError, match="The only supported basis set is"):
            Molecule(symbols, geometry, basis_name="6-31g")

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
    def test_generate_nuclear_charges(self, basis_data):
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

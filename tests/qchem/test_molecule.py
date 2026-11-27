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
"""Tests for molecular metadata."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.qchem.molecule import BOHR_TO_ANG


def test_molecule_metadata():
    """Store the information required by retained external backends."""
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    mol = qp.qchem.Molecule(["He", "H"], geometry, charge=1, basis_name="def2-svp", name="HeH")
    assert mol.symbols == ["He", "H"]
    assert mol.coordinates is geometry
    assert mol.charge == 1
    assert mol.mult == 1
    assert mol.basis_name == "def2-svp"
    assert mol.name == "HeH"
    assert mol.nuclear_charges == [2, 1]
    assert mol.n_electrons == 2
    assert repr(mol) == "<Molecule = HHe, Charge: 1, Basis: DEF2-SVP, Electrons: 2>"


def test_angstrom_coordinates():
    """Convert input coordinates to the atomic units used by backends."""
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    mol = qp.qchem.Molecule(["H", "H"], geometry, unit=" Angstrom ")
    np.testing.assert_allclose(mol.coordinates, geometry / BOHR_TO_ANG)


def test_invalid_unit():
    """Reject unsupported coordinate units."""
    with pytest.raises(ValueError, match="unit 'nm' is not supported"):
        qp.qchem.Molecule(["H"], np.zeros((1, 3)), unit="nm")


def test_invalid_symbol():
    """Reject unknown atomic symbols."""
    with pytest.raises(ValueError, match="are not supported"):
        qp.qchem.Molecule(["Unknown"], np.zeros((1, 3)))


def test_differentiable_basis_parameters_removed():
    """Custom Gaussian exponents are no longer constructor arguments."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'alpha'"):
        qp.qchem.Molecule(["H"], np.zeros((1, 3)), alpha=np.array([[1.0]]))

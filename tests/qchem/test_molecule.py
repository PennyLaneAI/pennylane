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
# pylint: disable=too-many-arguments
import pytest

from pennylane import numpy as np
from pennylane import qchem


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
        mol = qchem.Molecule(symbols, geometry)
        assert isinstance(mol, qchem.Molecule)

    @pytest.mark.parametrize(
        ("symbols", "geometry"),
        [
            (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
        ],
    )
    def test_basis_error(self, symbols, geometry):
        r"""Test that an error is raised if a wrong basis set name is entered."""
        with pytest.raises(ValueError, match="Currently, the only supported basis sets"):
            qchem.Molecule(symbols, geometry, basis_name="6-3_1_g")

    @pytest.mark.parametrize(
        ("symbols", "geometry"),
        [
            (["H", "Ox"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
        ],
    )
    def test_symbol_error(self, symbols, geometry):
        r"""Test that an error is raised if a wrong/not-supported atomic symbol is entered."""
        with pytest.raises(ValueError, match="are not supported"):
            qchem.Molecule(symbols, geometry)

    @pytest.mark.parametrize(
        ("symbols", "geometry", "charge", "mult", "basis_name"),
        [
            (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 0, 1, "sto-3g"),
        ],
    )
    def test_default_inputs(self, symbols, geometry, charge, mult, basis_name):
        r"""Test that the molecule object contains correct default molecular input data."""
        mol = qchem.Molecule(symbols, geometry)

        assert mol.symbols == symbols
        assert np.allclose(mol.coordinates, geometry)
        assert mol.charge == charge
        assert mol.mult == mult
        assert mol.basis_name == basis_name

    @pytest.mark.parametrize(
        ("symbols", "geometry", "n_electrons", "n_orbitals", "nuclear_charges"),
        [
            (
                ["H", "F"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                10,
                6,
                [1, 9],
            ),
        ],
    )
    def test_molecule_data(self, symbols, geometry, n_electrons, n_orbitals, nuclear_charges):
        r"""Test that the molecule object contains correct molecular data."""
        mol = qchem.Molecule(symbols, geometry)

        assert mol.n_electrons == n_electrons
        assert mol.n_orbitals == n_orbitals
        assert mol.nuclear_charges == nuclear_charges

    @pytest.mark.parametrize(
        ("symbols", "geometry", "n_basis", "basis_data"),
        [
            (
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
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
            ),
        ],
    )
    def test_default_basisdata(self, symbols, geometry, n_basis, basis_data):
        r"""Test that the molecule object contains correct default basis data for a given molecule."""
        mol = qchem.Molecule(symbols, geometry)

        assert mol.n_basis == n_basis
        assert np.allclose(mol.basis_data, basis_data)

    @pytest.mark.parametrize(
        ("symbols", "geometry", "l", "alpha", "coeff", "r", "load_data"),
        [
            (
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                [(0, 0, 0), (0, 0, 0)],
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                False,
            ),
            (
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                [(0, 0, 0), (0, 0, 0)],
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                True,
            ),
            (
                ["H", "F"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                [(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                [
                    [0.3425250914e01, 0.6239137298e00, 0.1688554040e00],
                    [0.1666791340e03, 0.3036081233e02, 0.8216820672e01],
                    [0.6464803249e01, 0.1502281245e01, 0.4885884864e00],
                    [0.6464803249e01, 0.1502281245e01, 0.4885884864e00],
                    [0.6464803249e01, 0.1502281245e01, 0.4885884864e00],
                    [0.6464803249e01, 0.1502281245e01, 0.4885884864e00],
                ],
                [
                    [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
                    [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
                    [-0.9996722919e-01, 0.3995128261e00, 0.7001154689e00],
                    [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
                    [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
                    [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
                ],
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                False,
            ),
        ],
    )
    def test_basisset(self, symbols, geometry, l, alpha, coeff, r, load_data):
        r"""Test that the molecule object contains the correct basis set and non-default basis data
        for a given molecule.
        """
        pytest.importorskip("basis_set_exchange")
        mol = qchem.Molecule(symbols, geometry, load_data=load_data, normalize=False)

        assert set(map(type, mol.basis_set)) == {qchem.BasisFunction}
        assert mol.l == l
        assert np.allclose(mol.alpha, alpha)
        assert np.allclose(mol.coeff, coeff)
        assert np.allclose(mol.r, r)

    @pytest.mark.parametrize(
        ("symbols", "geometry", "alpha", "coeff", "index", "position", "ref_value"),
        [
            (
                # normalized primitive Gaussians centered at 0, G(0, 0, 0) = coeff * exp(alpha * 0)
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array(
                    [
                        [3.425250914, 3.425250914, 3.425250914],
                        [3.425250914, 3.425250914, 3.425250914],
                    ],
                    requires_grad=False,
                ),
                np.array(
                    [[1.79444183, 1.79444183, 1.79444183], [1.79444183, 1.79444183, 1.79444183]],
                    requires_grad=False,
                ),
                0,
                (0.0, 0.0, 0.0),
                1.79444183,
            ),
            (
                # normalized primitive Gaussians centered at z=1, G(0, 0, 0) = coeff * exp(alpha *1)
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array(
                    [
                        [3.425250914, 3.425250914, 3.425250914],
                        [3.425250914, 3.425250914, 3.425250914],
                    ],
                    requires_grad=False,
                ),
                np.array(
                    [[1.79444183, 1.79444183, 1.79444183], [1.79444183, 1.79444183, 1.79444183]],
                    requires_grad=False,
                ),
                1,
                (0.0, 0.0, 0.0),
                0.05839313784917416,
            ),
        ],
    )
    def test_atomic_orbital(self, symbols, geometry, alpha, coeff, index, position, ref_value):
        r"""Test that the computed atomic orbital value is correct."""
        mol = qchem.Molecule(symbols, geometry, alpha=alpha, coeff=coeff)

        x, y, z = position
        ao = mol.atomic_orbital(index)
        ao_value = ao(x, y, z)

        assert np.allclose(ao_value, ref_value)

    @pytest.mark.parametrize(
        ("symbols", "geometry", "index", "position", "ref_value"),
        [
            (
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
                1,
                (0.0, 0.0, 0.0),
                0.01825128,
            ),
        ],
    )
    def test_molecular_orbital(self, symbols, geometry, index, position, ref_value):
        r"""Test that the computed atomic orbital value is correct."""
        mol = qchem.Molecule(symbols, geometry)

        x, y, z = position
        _ = qchem.scf(mol)()
        mo = mol.molecular_orbital(index)
        mo_value = mo(x, y, z)

        assert np.allclose(mo_value, ref_value)

    def test_repr(self):
        """Test that __repr__ for Molecule returns correct representation."""

        symbols, geometry = (
            ["H", "H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        )
        mol = qchem.Molecule(symbols, geometry, 1)
        assert repr(mol) == "<Molecule = H3, Charge: 1, Basis: STO-3G, Orbitals: 3, Electrons: 2>"

        symbols, geometry = (
            ["H", "C", "O"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        )
        mol = qchem.Molecule(symbols, geometry, 1)
        assert (
            repr(mol) == "<Molecule = CHO, Charge: 1, Basis: STO-3G, Orbitals: 11, Electrons: 14>"
        )

        symbols, geometry = (["C", "O"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
        mol = qchem.Molecule(symbols, geometry, 0)
        assert repr(mol) == "<Molecule = CO, Charge: 0, Basis: STO-3G, Orbitals: 10, Electrons: 14>"

    @pytest.mark.parametrize(
        ("symbols", "geometry", "basis_name", "params_ref"),
        [  # data manually copied from https://www.basissetexchange.org/
            (
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                "6-31+g",
                (
                    (
                        (0, 0, 0),  # l
                        [0.1873113696e02, 0.2825394365e01, 0.6401216923e00],  # alpha
                        [0.3349460434e-01, 0.2347269535e00, 0.8137573261e00],  # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [0.1612777588e00],  # alpha
                        [1.0000000],  # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [0.1873113696e02, 0.2825394365e01, 0.6401216923e00],  # alpha
                        [0.3349460434e-01, 0.2347269535e00, 0.8137573261e00],  # coeff
                    ),
                    (
                        (0, 0, 0),  # l
                        [0.1612777588e00],  # alpha
                        [1.0000000],  # coeff
                    ),
                ),
            ),
        ],
    )
    def test_load_basis(self, symbols, geometry, basis_name, params_ref):
        """Test that correct basis set parameters are loaded from the basis_set_exchange library."""

        pytest.importorskip("basis_set_exchange")

        mol = qchem.Molecule(symbols, geometry, basis_name=basis_name, load_data=True)

        params = mol.basis_data

        l = [p[0] for p in params]
        l_ref = [p[0] for p in params_ref]

        alpha = [p[1] for p in params]
        alpha_ref = [p[1] for p in params_ref]

        coeff = [p[2] for p in params]
        coeff_ref = [p[2] for p in params_ref]

        assert l == l_ref
        assert alpha == alpha_ref
        assert coeff == coeff_ref

    def test_unit_error(self):
        r"""Test that an error is raised if a wrong/not-supported unit for coordinates is entered."""

        symbols = ["H", "H"]
        geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        with pytest.raises(ValueError, match="The provided unit 'degrees' is not supported."):
            qchem.Molecule(symbols, geometry, unit="degrees")

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
This module contains tests for functions needed to compute PES object.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

import pennylane as qml
from pennylane.qchem import vibrational
from pennylane.qchem.vibrational import vibrational_class

h5py = pytest.importorskip("h5py")
# pylint: disable=too-many-arguments, protected-access


def test_import_geometric(monkeypatch):
    """Test if an ImportError is raised by _import_geometric function."""
    # pylint: disable=protected-access

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "geometric", None)

        with pytest.raises(ImportError, match="This feature requires geometric"):
            vibrational.vibrational_class._import_geometric()


@pytest.mark.usefixtures("skip_if_no_pyscf_support", "skip_if_no_geometric_support")
def test_optimize_geometry_methoderror():
    r"""Test that an error is raised if wrong method is provided for
    geometry optimization."""

    symbols = ["H", "H"]
    geom = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    mol = qml.qchem.Molecule(symbols, geom)

    with pytest.raises(
        ValueError, match="Specified electronic structure method, ccsd, is not available."
    ):
        vibrational.optimize_geometry(mol, method="ccsd")


@pytest.mark.parametrize(
    ("sym", "geom", "unit", "method", "basis", "expected_energy"),
    # Expected energy was obtained using pyscf
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            "Angstrom",
            "RHF",
            "6-31g",
            -99.97763667852357,
        ),
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            "Bohr",
            "UHF",
            "6-31g",
            -99.43441545109692,
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            "Angstrom",
            "RHF",
            "6-31+g",
            -1.094807962860512,
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_single_point_energy(sym, geom, unit, method, basis, expected_energy):
    r"""Test that correct energy is produced for a given molecule."""

    mol = qml.qchem.Molecule(sym, geom, unit=unit, basis_name=basis, load_data=True)
    scf_obj = qml.qchem.vibrational.vibrational_class._single_point(mol, method=method)

    assert np.isclose(scf_obj.e_tot, expected_energy)


@pytest.mark.parametrize(
    ("sym", "geom", "unit", "expected_geom"),
    # Expected geometry was obtained using pyscf
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            "Angstrom",
            np.array([[0.0, 0.0, 0.03967348], [0.0, 0.0, 0.96032612]]),
        ),
        (
            ["C", "O"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.88972613]]),
            "Bohr",
            np.array([[0.0, 0.0, -0.12346543], [0.0, 0.0, 2.0131908]]),
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support", "skip_if_no_geometric_support")
def test_optimize_geometry(sym, geom, unit, expected_geom):
    r"""Test that correct optimized geometry is obtained."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit=unit)
    coordinates = vibrational.optimize_geometry(mol)
    print(coordinates, geom)
    assert np.allclose(coordinates, expected_geom)


@pytest.mark.parametrize(
    ("sym", "geom", "expected_vecs"),
    # Expected displacement vectors were obtained using vibrant code
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[[0.0, 0.0, -0.9706078], [0.0, 0.0, 0.05149763]]]),
        ),
        (
            ["C", "O"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[[0.0, 0.0, -0.21807219], [0.0, 0.0, 0.1637143]]]),
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support", "skip_if_no_geometric_support")
def test_harmonic_analysis(sym, geom, expected_vecs):
    r"""Test that the correct displacement vectors are obtained after harmonic analysis."""
    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom")
    geom_eq = vibrational.optimize_geometry(mol)
    mol_eq = qml.qchem.Molecule(
        mol.symbols,
        geom_eq,
        unit=mol.unit,
        basis_name=mol.basis_name,
        load_data=mol.load_data,
    )

    scf_result = vibrational_class._single_point(mol_eq)

    _, displ_vecs = vibrational_class._harmonic_analysis(scf_result)
    assert np.allclose(displ_vecs, expected_vecs) or np.allclose(
        displ_vecs, -1 * np.array(expected_vecs)
    )


@pytest.mark.parametrize(
    ("freqs", "vecs", "bins", "exp_results"),
    # Expected results were obtained using vibrant code
    [
        (
            # HF Molecule with Bond distance 1.0 A.
            np.array([4137.96875377]),
            np.array([[[0.0, 0.0, 0.9706078], [0.0, 0.0, -0.05149763]]]),
            [2600],
            {
                "vecs": [[[0.0, 0.0, 0.9706078], [0.0, 0.0, -0.05149763]]],
                "freqs": [4137.96875377],
                "uloc": [[1.0]],
            },
        ),
        (
            # H2S Molecule with geometry [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]
            np.array([1294.21950382, 2691.27147004, 2718.40232678]),
            np.array(
                [
                    [
                        [5.04812381e-17, -4.56823323e-01, 5.19946514e-01],
                        [1.86137414e-16, 4.56823322e-01, 5.19946514e-01],
                        [1.35223494e-17, 1.15509844e-11, -3.26953266e-02],
                    ],
                    [
                        [-9.48719985e-18, -5.36045204e-01, -4.43104273e-01],
                        [1.58761035e-16, 5.36044714e-01, -4.43103833e-01],
                        [5.31102499e-17, 1.54185981e-08, 2.78633116e-02],
                    ],
                    [
                        [6.52265582e-17, -5.15178735e-01, -4.62528551e-01],
                        [3.12480469e-16, -5.15179245e-01, 4.62528972e-01],
                        [1.63797372e-17, 3.23955347e-02, -1.32498366e-08],
                    ],
                ]
            ),
            [1000, 2000],
            {
                "vecs": [
                    [
                        [8.73825764e-18, 4.56823325e-01, -5.19946511e-01],
                        [2.70251290e-17, -4.56823323e-01, -5.19946513e-01],
                        [-4.06626509e-17, -7.26343693e-11, 3.26953265e-02],
                    ],
                    [
                        [-3.76513979e-17, -7.43327581e-01, -6.40379106e-01],
                        [7.06164585e-17, 1.47544755e-02, 1.37353335e-02],
                        [1.43329806e-17, 2.29071020e-02, 1.97023370e-02],
                    ],
                    [
                        [-3.30668012e-17, 1.47544588e-02, -1.37353509e-02],
                        [3.85908620e-18, -7.43327582e-01, 6.40379105e-01],
                        [-1.26315618e-17, 2.29071026e-02, -1.97023364e-02],
                    ],
                ],
                "freqs": [
                    1294.21950382,
                    2704.87090197,
                    2704.87092841,
                ],  # [0.00589689, 0.01232428, 0.01232428],
                "uloc": [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.70710715, -0.70710641],
                    [0.0, 0.70710641, 0.70710715],
                ],
            },
        ),
        (
            # H2S Molecule with geometry [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]
            np.array([1294.21950382, 2691.27147004, 2718.40232678]),
            np.array(
                [
                    [
                        [5.04812381e-17, -4.56823323e-01, 5.19946514e-01],
                        [1.86137414e-16, 4.56823322e-01, 5.19946514e-01],
                        [1.35223494e-17, 1.15509844e-11, -3.26953266e-02],
                    ],
                    [
                        [-9.48719985e-18, -5.36045204e-01, -4.43104273e-01],
                        [1.58761035e-16, 5.36044714e-01, -4.43103833e-01],
                        [5.31102499e-17, 1.54185981e-08, 2.78633116e-02],
                    ],
                    [
                        [6.52265582e-17, -5.15178735e-01, -4.62528551e-01],
                        [3.12480469e-16, -5.15179245e-01, 4.62528972e-01],
                        [1.63797372e-17, 3.23955347e-02, -1.32498366e-08],
                    ],
                ]
            ),
            [2600],
            {
                "vecs": [
                    [
                        [8.73825764e-18, 4.56823325e-01, -5.19946511e-01],
                        [2.70251290e-17, -4.56823323e-01, -5.19946513e-01],
                        [-4.06626509e-17, -7.26343693e-11, 3.26953265e-02],
                    ],
                    [
                        [-3.76513979e-17, -7.43327581e-01, -6.40379106e-01],
                        [7.06164585e-17, 1.47544755e-02, 1.37353335e-02],
                        [1.43329806e-17, 2.29071020e-02, 1.97023370e-02],
                    ],
                    [
                        [-3.30668012e-17, 1.47544588e-02, -1.37353509e-02],
                        [3.85908620e-18, -7.43327582e-01, 6.40379105e-01],
                        [-1.26315618e-17, 2.29071026e-02, -1.97023364e-02],
                    ],
                ],
                "freqs": [1294.21950382, 2704.87090197, 2704.87092841],
                "uloc": [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.70710715, -0.70710641],
                    [0.0, 0.70710641, 0.70710715],
                ],
            },
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support", "skip_if_no_geometric_support")
def test_mode_localization(freqs, vecs, bins, exp_results):
    r"""Test that mode localization returns correct results."""

    freqs_loc, vecs_loc, uloc = vibrational.localize_normal_modes(freqs, vecs, bins=bins)
    nmodes = len(freqs)
    for i in range(nmodes):
        res_in_expvecs = any(
            (
                np.allclose(vecs_loc[i], vec, atol=1e-6)
                or np.allclose(vecs_loc[i], -1.0 * np.array(vec), atol=1e-6)
            )
            for vec in exp_results["vecs"]
        )
        exp_in_resvecs = any(
            (
                np.allclose(exp_results["vecs"][i], vec, atol=1e-6)
                or np.allclose(exp_results["vecs"][i], -1.0 * np.array(vec), atol=1e-6)
            )
            for vec in vecs_loc
        )

        res_in_expuloc = any(
            (
                np.allclose(uloc[i], u, atol=1e-6)
                or np.allclose(uloc[i], -1.0 * np.array(u), atol=1e-6)
            )
            for u in exp_results["uloc"]
        )
        exp_in_resuloc = any(
            (
                np.allclose(exp_results["uloc"][i], u, atol=1e-6)
                or np.allclose(exp_results["uloc"][i], -1.0 * np.array(u), atol=1e-6)
            )
            for u in uloc
        )
        assert res_in_expvecs and exp_in_resvecs
        assert res_in_expuloc and exp_in_resuloc

    assert np.allclose(freqs_loc, exp_results["freqs"])


@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_hess_methoderror():
    r"""Test that an error is raised if wrong method is provided for
    harmonic analysis."""

    symbols = ["H", "H"]
    geom = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    mol = qml.qchem.Molecule(symbols, geom)
    mol_scf = qml.qchem.vibrational.vibrational_class._single_point(mol)

    with pytest.raises(
        ValueError, match="Specified electronic structure method, ccsd is not available."
    ):
        vibrational_class._harmonic_analysis(mol_scf, method="ccsd")


@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_error_mode_localization():
    r"""Test that an error is raised if empty list of frequencies is provided for localization"""

    sym = ["H", "F"]
    geom = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_scf = qml.qchem.vibrational.vibrational_class._single_point(mol)
    freqs, vecs = vibrational_class._harmonic_analysis(mol_scf)
    with pytest.raises(ValueError, match="The `bins` list cannot be empty."):
        vibrational.localize_normal_modes(freqs, vecs, bins=[])


@pytest.mark.parametrize(
    ("sym", "geom", "method", "mult", "charge", "expected_dipole"),
    # Expected dipole was obtained using vibrant code
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.03967368], [0.0, 0.0, 0.96032632]]),
            "RHF",
            1,
            0,
            [-3.78176692e-16, -3.50274735e-17, -9.05219767e-01],
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            "RHF",
            3,
            1,
            [1.10150593e-15, -1.68930482e-16, -1.60982339e-15],
        ),
        (
            ["H", "H", "S"],
            np.array(
                [
                    [0.0, -1.00688408, -0.9679942],
                    [0.0, 1.00688408, -0.9679942],
                    [0.0, 0.0, -0.0640116],
                ]
            ),
            "UHF",
            1,
            0,
            [1.95258747e-16, 5.62355462e-15, -7.34149703e-01],
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_get_dipole(sym, geom, mult, charge, method, expected_dipole):
    r"""Test that the get_dipole function produces correct results."""
    mol = qml.qchem.Molecule(
        sym, geom, mult=mult, charge=charge, basis_name="6-31g", unit="Angstrom", load_data=True
    )
    mol_scf = qml.qchem.vibrational.vibrational_class._single_point(mol, method=method)
    dipole = vibrational_class._get_dipole(mol_scf, method=method)
    assert np.allclose(dipole, expected_dipole)


def test_VibrationalPES():
    r"""Test that VibrationalPES class stores the correct results."""

    pes_file = Path(__file__).resolve().parent / "test_ref_files" / "HF.hdf5"
    with h5py.File(pes_file, "r+") as f:
        pes_onebody = np.array(f["V1_PES"][()])
        dip_onebody = np.array(f["D1_DMS"][()])
        pes_twobody = np.array(f["V2_PES"][()])
        dip_twobody = np.array(f["D2_DMS"][()])
        pes_threebody = np.array(f["V3_PES"][()])
        dip_threebody = np.array(f["D3_DMS"][()])
        freqs = np.array(f["freqs"][()])
        uloc = np.array(f["Uloc"][()])

    pes_data = [pes_onebody, pes_twobody, pes_threebody]
    dipole_data = [dip_onebody, dip_twobody, dip_threebody]

    grid, gauss_weights = np.polynomial.hermite.hermgauss(pes_onebody.shape[1])

    vib_obj = vibrational.VibrationalPES(
        freqs, grid, gauss_weights, uloc, pes_data, dipole_data, localized=True, dipole_level=3
    )

    assert np.allclose(vib_obj.freqs, freqs)
    assert np.allclose(vib_obj.pes_onemode, pes_onebody)
    assert np.allclose(vib_obj.pes_twomode, pes_twobody)
    assert np.allclose(vib_obj.pes_threemode, pes_threebody)
    assert np.allclose(vib_obj.dipole_onemode, dip_onebody)
    assert np.allclose(vib_obj.dipole_twomode, dip_twobody)
    assert np.allclose(vib_obj.dipole_threemode, dip_threebody)
    assert np.allclose(vib_obj.grid, grid)
    assert np.allclose(vib_obj.uloc, uloc)
    assert np.allclose(vib_obj.gauss_weights, gauss_weights)

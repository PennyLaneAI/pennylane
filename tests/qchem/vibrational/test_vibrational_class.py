"""
This module contains tests for functions needed to compute PES object."""

import pytest

import numpy as np
import pennylane as qml
from pennylane.qchem import vibrational

AU_TO_CM = 219475

# pylint: disable=too-many-arguments


def test_es_methoderror():
    r"""Test that an error is raised if wrong method is provided for
    geometry optimization."""

    symbols = ["H", "H"]
    geom = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    mol = qml.qchem.Molecule(symbols, geom)

    with pytest.raises(
        ValueError, match="Specified electronic structure method, ccsd is not available."
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
def test_scf_energy(sym, geom, unit, method, basis, expected_energy):
    r"""Test that correct energy is produced for a given molecule."""

    mol = qml.qchem.Molecule(sym, geom, unit=unit, basis_name=basis, load_data=True)
    scf_obj = vibrational.single_point(mol, method=method)

    assert np.isclose(scf_obj.e_tot, expected_energy)


@pytest.mark.parametrize(
    ("sym", "geom", "expected_geom"),
    # Expected geometry was obtained using pyscf
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 0.07497201], [0.0, 0.0, 1.81475336]]),
        ),
        (
            ["C", "O"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, -0.12346543], [0.0, 0.0, 2.0131908]]),
        ),
    ],
)
def test_optimize_geometry(sym, geom, expected_geom):
    r"""Test that correct optimized geometry is obtained."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom")
    mol_eq = vibrational.optimize_geometry(mol)
    assert np.allclose(mol_eq[0].coordinates, expected_geom)


@pytest.mark.parametrize(
    ("sym", "geom", "expected_vecs"),
    # Expected displacement vectors were obtained using vibrant code
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[[0.0, 0.0, 0.9706078], [0.0, 0.0, -0.05149763]]]),
        ),
        (
            ["C", "O"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[[0.0, 0.0, -0.21807219], [0.0, 0.0, 0.1637143]]]),
        ),
    ],
)
def test_harmonic_analysis(sym, geom, expected_vecs):
    r"""Test that the correct displacement vectors are obtained after harmonic analysis."""
    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom")
    mol_eq = vibrational.optimize_geometry(mol)
    harmonic_res = vibrational.harmonic_analysis(mol_eq[1])
    assert np.allclose(harmonic_res["norm_mode"], expected_vecs)


@pytest.mark.parametrize(
    ("sym", "geom", "loc_freqs", "exp_results"),
    # Expected results were obtained using vibrant code
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            [2600],
            {
                "vecs": [[[0.0, 0.0, 0.9706078], [0.0, 0.0, -0.05149763]]],
                "freqs": [0.01885394],
                "uloc": [[1.0]],
            },
        ),
        (
            ["H", "H", "S"],
            np.array([[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]),
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
                "freqs": [0.00589689, 0.01232428, 0.01232428],
                "uloc": [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.70710715, -0.70710641],
                    [0.0, 0.70710641, 0.70710715],
                ],
            },
        ),
    ],
)
def test_mode_localization(sym, geom, loc_freqs, exp_results):
    r"""Test that mode localization returns correct results."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = vibrational.optimize_geometry(mol)

    harmonic_res = vibrational.harmonic_analysis(mol_eq[1])
    loc_res, uloc = vibrational.localize_normal_modes(harmonic_res, freq_separation=loc_freqs)
    freqs = loc_res["freq_wavenumber"] / AU_TO_CM

    nmodes = len(freqs)
    for i in range(nmodes):
        res_in_expvecs = any(
            (
                np.allclose(loc_res["norm_mode"][i], vec, atol=1e-6)
                or np.allclose(loc_res["norm_mode"][i], -1.0 * np.array(vec), atol=1e-6)
            )
            for vec in exp_results["vecs"]
        )
        exp_in_resvecs = any(
            (
                np.allclose(exp_results["vecs"][i], vec, atol=1e-6)
                or np.allclose(exp_results["vecs"][i], -1.0 * np.array(vec), atol=1e-6)
            )
            for vec in loc_res["norm_mode"]
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

    assert np.allclose(freqs, exp_results["freqs"])


def test_hess_methoderror():
    r"""Test that an error is raised if wrong method is provided for
    harmonic analysis."""

    symbols = ["H", "H"]
    geom = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    mol = qml.qchem.Molecule(symbols, geom)
    mol_scf = vibrational.single_point(mol)

    with pytest.raises(
        ValueError, match="Specified electronic structure method, ccsd is not available."
    ):
        vibrational.harmonic_analysis(mol_scf, method="ccsd")


def test_error_mode_localization():
    r"""Test that an error is raised if empty list of frequencies is provided for localization"""

    sym = ["H", "F"]
    geom = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = vibrational.optimize_geometry(mol)

    harmonic_res = vibrational.harmonic_analysis(mol_eq[1])
    with pytest.raises(ValueError, match="The `freq_separation` list cannot be empty."):
        vibrational.localize_normal_modes(harmonic_res, freq_separation=[])

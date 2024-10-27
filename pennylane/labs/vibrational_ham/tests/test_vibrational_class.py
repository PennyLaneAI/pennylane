"""
This module contains tests for functions needed to compute PES object."""

import pytest

import numpy as np
import pennylane as qml
from pennylane.labs import vibrational_ham

au_to_cm = 219475


def test_es_methoderror():
    r"""Test that an error is raised if wrong method is provided for
    geometry optimization."""

    symbols = ["H", "H"]
    geom = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    mol = qml.qchem.Molecule(symbols, geom)

    with pytest.raises(
        ValueError, match="Specified electronic structure method, ccsd is not available."
    ):
        vibrational_ham.optimize_geometry(mol, method="ccsd")


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
    scf_obj = vibrational_ham.single_point(mol, method=method)

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
    mol_eq = vibrational_ham.optimize_geometry(mol)
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
    mol_eq = vibrational_ham.optimize_geometry(mol)
    harmonic_res = vibrational_ham.harmonic_analysis(mol_eq[1])
    assert np.allclose(harmonic_res["norm_mode"], expected_vecs)


@pytest.mark.parametrize(
    ("sym", "geom", "loc_freqs", "exp_vecs", "exp_freqs", "exp_uloc"),
    # Expected results were obtained using vibrant code
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            [2600],
            [[[0.0, 0.0, 0.9706078], [0.0, 0.0, -0.05149763]]],
            [0.01885394],
            [[1.0]],
        ),
        (
            ["H", "H", "S"],
            np.array([[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]),
            [2600],
            [
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
            [0.00589689, 0.01232428, 0.01232428],
            [[1.0, 0.0, 0.0], [0.0, 0.70710715, -0.70710641], [0.0, 0.70710641, 0.70710715]],
        ),
    ],
)
def test_mode_localization(sym, geom, loc_freqs, exp_vecs, exp_freqs, exp_uloc):
    r"""Test that mode localization returns correct results."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = vibrational_ham.optimize_geometry(mol)

    harmonic_res = vibrational_ham.harmonic_analysis(mol_eq[1])
    loc_res, uloc = vibrational_ham.localize_normal_modes(harmonic_res, freq_separation=loc_freqs)
    freqs = loc_res["freq_wavenumber"] / au_to_cm

    assert np.allclose(loc_res["norm_mode"], exp_vecs)
    assert np.allclose(freqs, exp_freqs)
    assert np.allclose(uloc, exp_uloc)


def test_hess_methoderror():
    r"""Test that an error is raised if wrong method is provided for
    harmonic analysis."""

    symbols = ["H", "H"]
    geom = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    mol = qml.qchem.Molecule(symbols, geom)
    mol_scf = vibrational_ham.single_point(mol)

    with pytest.raises(
        ValueError, match="Specified electronic structure method, ccsd is not available."
    ):
        vibrational_ham.harmonic_analysis(mol_scf, method="ccsd")

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
import os
import numpy as np
import pytest

import pennylane as qml
from pennylane.qchem import *
import h5py
AU_TO_CM = 219475

# pylint: disable=too-many-arguments

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")

@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_es_methoderror():
    r"""Test that an error is raised if wrong method is provided for
    geometry optimization."""

    symbols = ["H", "H"]
    geom = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    mol = qml.qchem.Molecule(symbols, geom)

    with pytest.raises(
        ValueError, match="Specified electronic structure method, ccsd, is not available."
    ):
        optimize_geometry(mol, method="ccsd")


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
def test_scf_energy(sym, geom, unit, method, basis, expected_energy):
    r"""Test that correct energy is produced for a given molecule."""

    mol = qml.qchem.Molecule(sym, geom, unit=unit, basis_name=basis, load_data=True)
    scf_obj = single_point(mol, method=method)

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
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_optimize_geometry(sym, geom, expected_geom):
    r"""Test that correct optimized geometry is obtained."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom")
    mol_eq = optimize_geometry(mol)
    assert np.allclose(mol_eq[0].coordinates, expected_geom)


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
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_harmonic_analysis(sym, geom, expected_vecs):
    r"""Test that the correct displacement vectors are obtained after harmonic analysis."""
    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom")
    mol_eq = optimize_geometry(mol)
    harmonic_res = harmonic_analysis(mol_eq[1])

    assert np.allclose(harmonic_res["norm_mode"], expected_vecs) or np.allclose(
        harmonic_res["norm_mode"], -1 * np.array(expected_vecs)
    )


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
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_mode_localization(sym, geom, loc_freqs, exp_results):
    r"""Test that mode localization returns correct results."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = optimize_geometry(mol)

    harmonic_res = harmonic_analysis(mol_eq[1])
    loc_res, uloc = localize_normal_modes(harmonic_res, freq_separation=loc_freqs)
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


@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_hess_methoderror():
    r"""Test that an error is raised if wrong method is provided for
    harmonic analysis."""

    symbols = ["H", "H"]
    geom = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    mol = qml.qchem.Molecule(symbols, geom)
    mol_scf = single_point(mol)

    with pytest.raises(
        ValueError, match="Specified electronic structure method, ccsd is not available."
    ):
        harmonic_analysis(mol_scf, method="ccsd")


@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_error_mode_localization():
    r"""Test that an error is raised if empty list of frequencies is provided for localization"""

    sym = ["H", "F"]
    geom = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = optimize_geometry(mol)

    harmonic_res = harmonic_analysis(mol_eq[1])
    with pytest.raises(ValueError, match="The `freq_separation` list cannot be empty."):
        localize_normal_modes(harmonic_res, freq_separation=[])


@pytest.mark.parametrize(
    ("sym", "geom", "harmonic_res", "exp_pes_onemode", "exp_dip_onemode"),
    # Expected results were obtained using vibrant code
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.03967368], [0.0, 0.0, 0.96032632]]),
            {'freq_wavenumber': np.array([4137.96877864]), 'norm_mode': np.array([[[-1.93180788e-17, -2.79300482e-17, -9.70607797e-01],
                                                                             [-3.49162986e-17,  8.71500605e-17,  5.14976259e-02]]])},
            np.array(
                [
                    [
                        0.22015546,
                        0.08479983,
                        0.02880759,
                        0.00584308,
                        0.0,
                        0.00420657,
                        0.01501477,
                        0.03092302,
                        0.05234379,
                    ]
                ]
            ),
            np.array(
                [
                    [
                        [-1.64944938e-16, 1.23788916e-16, 1.63989194e-01],
                        [-1.83439598e-16, 3.39162428e-16, 1.17708005e-01],
                        [1.05778290e-16, 2.68365451e-16, 7.60769136e-02],
                        [2.12937257e-16, 2.88191645e-16, 3.69897289e-02],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00],
                        [-6.69962210e-17, 2.53813366e-16, -3.52192473e-02],
                        [-1.49136505e-17, 2.63832036e-16, -6.92725488e-02],
                        [1.40888251e-16, 1.36994405e-16, -1.03362529e-01],
                        [1.09576098e-16, 1.60885404e-16, -1.40432255e-01],
                    ]
                ]
            ),
        ),
        (
            ["H", "H", "S"],
            np.array([[0.0, -1.00688408, -0.9679942], [0.0, 1.00688408, -0.9679942 ], [0.0, 0.0, -0.06401159]]),
            {'freq_wavenumber': np.array([1294.2195371 , 2691.27147945, 2718.4023196 ]),
             'norm_mode': np.array([[[ 5.04812379e-17, -4.56823333e-01,  5.19946505e-01],
                                     [ 1.86137417e-16,  4.56823334e-01,  5.19946504e-01],
                                     [ 1.35223505e-17, -1.52311695e-11, -3.26953260e-02]],
                                    
                                    [[-9.48723219e-18, -5.36044948e-01, -4.43104062e-01],
                                     [ 1.58760881e-16,  5.36044952e-01, -4.43104065e-01],
                                     [ 5.31102418e-17, -1.25299135e-10,  2.78633123e-02]],

                                    [[ 6.52265536e-17, -5.15178992e-01, -4.62528763e-01],
                                     [ 3.12480546e-16, -5.15178988e-01,  4.62528760e-01],
                                     [ 1.63797627e-17,  3.23955347e-02,  9.23972875e-11]]])},
            np.array([[0.03038184, 0.0155299,  0.00650523, 0.00156346, 0.0, 0.00151837,
                       0.00613061, 0.01428086, 0.02755765],
                      [0.10140357, 0.04412527, 0.01637343, 0.00355301, 0.0, 0.00290997,
                       0.01083774, 0.02329395, 0.04121951],
                      [0.07198428, 0.0340414,  0.01374172, 0.00326402, 0.0, 0.00326402,
                       0.01374172, 0.0340414,  0.07198428]]),
            np.array([[[-5.36105111e-16,  1.33592345e-10,  9.41527336e-02],
                       [-6.24887513e-16,  8.82111475e-11,  7.79937948e-02],
                       [-5.19872629e-16,  5.05100438e-11,  5.33232767e-02],
                       [-5.39642354e-17,  2.12469860e-11,  2.60979423e-02],
                       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                       [ 5.11986070e-16, -1.42688961e-11, -2.28485454e-02],
                       [-7.19161722e-16, -2.15452303e-11, -4.12130918e-02],
                       [-5.12026491e-16, -2.06124746e-11, -5.38435444e-02],
                       [ 9.35736172e-16, -5.75791709e-12, -5.77604214e-02]],

                      [[-2.22066607e-16,  2.95745910e-10, -9.70203759e-02],
                       [-6.58015868e-16,  2.54133013e-10, -7.42203663e-02],
                       [-5.08918190e-16,  1.83813440e-10, -5.04545990e-02],
                       [ 1.10366883e-15,  9.69697526e-11, -2.56971336e-02],
                       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                       [-2.85241742e-16, -1.02915824e-10,  2.66325967e-02],
                       [-1.39431150e-17, -2.08111968e-10,  5.43391161e-02],
                       [-1.46190173e-17, -3.11964836e-10,  8.35802521e-02],
                       [ 1.71204651e-16, -4.11259828e-10,  1.15968472e-01]],

                      [[-1.04883276e-16, -1.03296190e-01,  1.35696490e-03],
                       [ 4.87263795e-18, -7.73177418e-02,  8.19496674e-04],
                       [-4.97201877e-16, -5.16458875e-02,  3.84521980e-04],
                       [-5.75774440e-16, -2.58720612e-02,  1.00242797e-04],
                       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                       [ 3.02208598e-16,  2.58720612e-02,  1.00242982e-04],
                       [ 9.52676251e-16,  5.16458875e-02,  3.84522344e-04],
                       [ 4.48188933e-16,  7.73177419e-02,  8.19497209e-04],
                       [-1.04202762e-15,  1.03296190e-01,  1.35696559e-03]]])
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_onemode_pes(sym, geom, harmonic_res, exp_pes_onemode, exp_dip_onemode):
    r"""Test that the correct onemode PES is obtained."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = single_point(mol)

    gauss_grid, gauss_weights = np.polynomial.hermite.hermgauss(9)

    freqs_au = harmonic_res["freq_wavenumber"] / AU_TO_CM
    displ_vecs = harmonic_res["norm_mode"]
    pes_onebody, dipole_onebody = pes_onemode(
        mol, mol_eq, freqs_au, displ_vecs, gauss_grid, method="RHF", do_dipole=True
    )


    assert np.allclose(pes_onebody, exp_pes_onemode, atol=1e-6)
    assert np.allclose(dipole_onebody, exp_dip_onemode, atol=1e-6)

@pytest.mark.parametrize(
    ("sym", "geom", "harmonic_res", "ref_file"),
    # Expected results were obtained using vibrant code
    [
        (
            ["H", "H", "S"],
            np.array([[0.0, -1.00688408, -0.9679942], [0.0, 1.00688408, -0.9679942 ], [0.0, 0.0, -0.06401159]]),
            {'freq_wavenumber': np.array([1294.2195371 , 2691.27147945, 2718.4023196 ]),
             'norm_mode': np.array([[[ 5.04812379e-17, -4.56823333e-01,  5.19946505e-01],
                                     [ 1.86137417e-16,  4.56823334e-01,  5.19946504e-01],
                                     [ 1.35223505e-17, -1.52311695e-11, -3.26953260e-02]],
                                    
                                    [[-9.48723219e-18, -5.36044948e-01, -4.43104062e-01],
                                     [ 1.58760881e-16,  5.36044952e-01, -4.43104065e-01],
                                     [ 5.31102418e-17, -1.25299135e-10,  2.78633123e-02]],

                                    [[ 6.52265536e-17, -5.15178992e-01, -4.62528763e-01],
                                     [ 3.12480546e-16, -5.15178988e-01,  4.62528760e-01],
                                     [ 1.63797627e-17,  3.23955347e-02,  9.23972875e-11]]])},
            "H2S.hdf5"
        )
    ]
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_twomode_pes(sym, geom, harmonic_res, ref_file):
    r"""Test that the correct onemode PES is obtained."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = single_point(mol)

    gauss_grid, gauss_weights = np.polynomial.hermite.hermgauss(9)

    freqs_au = harmonic_res["freq_wavenumber"] / AU_TO_CM
    displ_vecs = harmonic_res["norm_mode"]
    pes_onebody, dipole_onebody = pes_onemode(
        mol, mol_eq, freqs_au, displ_vecs, gauss_grid, method="RHF", do_dipole=True
    )

    pes_twobody, dipole_twobody = pes_twomode(mol, mol_eq, freqs_au, displ_vecs, gauss_grid, pes_onebody, dipole_onebody, method="rhf", do_dipole=True)

    pes_file = os.path.join(ref_dir, ref_file)
    f = h5py.File(pes_file, 'r+')
    exp_pes_twomode = np.array(f['V2_PES'][()])
    exp_dip_twomode = np.array(f['D2_DMS'][()])


    assert np.allclose(pes_twobody, exp_pes_twomode, atol=1e-6) 
    assert np.allclose(dipole_twobody, exp_dip_twomode, atol=1e-6)


# @pytest.mark.parametrize(
#     ("sym", "geom", "harmonic_res", "ref_file"),
#     # Expected results were obtained using vibrant code
#     [
#         (
#             ["H", "H", "S"],
#             np.array([[0.0, -1.00688408, -0.9679942], [0.0, 1.00688408, -0.9679942 ], [0.0, 0.0, -0.06401159]]),
#             {'freq_wavenumber': np.array([1294.2195371 , 2691.27147945, 2718.4023196 ]),
#              'norm_mode': np.array([[[ 5.04812379e-17, -4.56823333e-01,  5.19946505e-01],
#                                      [ 1.86137417e-16,  4.56823334e-01,  5.19946504e-01],
#                                      [ 1.35223505e-17, -1.52311695e-11, -3.26953260e-02]],
                                    
#                                     [[-9.48723219e-18, -5.36044948e-01, -4.43104062e-01],
#                                      [ 1.58760881e-16,  5.36044952e-01, -4.43104065e-01],
#                                      [ 5.31102418e-17, -1.25299135e-10,  2.78633123e-02]],

#                                     [[ 6.52265536e-17, -5.15178992e-01, -4.62528763e-01],
#                                      [ 3.12480546e-16, -5.15178988e-01,  4.62528760e-01],
#                                      [ 1.63797627e-17,  3.23955347e-02,  9.23972875e-11]]])},
#             "H2S.hdf5"
#         )
#     ]
# )
# @pytest.mark.usefixtures("skip_if_no_pyscf_support")
# def test_threemode_pes(sym, geom, harmonic_res, ref_file):
#     r"""Test that the correct onemode PES is obtained."""

#     mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
#     mol_eq = single_point(mol)

#     gauss_grid, gauss_weights = np.polynomial.hermite.hermgauss(9)

#     freqs_au = harmonic_res["freq_wavenumber"] / AU_TO_CM
#     displ_vecs = harmonic_res["norm_mode"]
#     pes_onebody, dipole_onebody = pes_onemode(
#         mol, mol_eq, freqs_au, displ_vecs, gauss_grid, method="RHF", do_dipole=True
#     )

#     pes_twobody, dipole_twobody = pes_twomode(mol, mol_eq, freqs_au, displ_vecs, gauss_grid, pes_onebody, dipole_onebody, method="rhf", do_dipole=True)

#     pes_threebody, dipole_threebody = pes_threemode(mol, mol_eq, freqs_au, displ_vecs, gauss_grid, pes_onebody, pes_twobody, dipole_onebody,
#                                                     dipole_twobody, method="rhf", do_dipole=True)
#     pes_file = os.path.join(ref_dir, ref_file)
#     f = h5py.File(pes_file, 'r+')
#     exp_pes_threemode = np.array(f['V3_PES'][()])
#     exp_dip_threemode = np.array(f['D3_DMS'][()])

#     assert np.allclose(pes_threebody, exp_pes_threemode, atol=1e-6) 
#     assert np.allclose(dipole_threebody, exp_dip_threemode, atol=1e-6)

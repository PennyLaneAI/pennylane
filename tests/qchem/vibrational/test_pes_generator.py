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
import sys

import numpy as np
import pytest

import pennylane as qml
from pennylane.qchem import vibrational
from pennylane.qchem.vibrational import pes_generator, vibrational_class

h5py = pytest.importorskip("h5py")

AU_TO_CM = 219475

# pylint: disable=too-many-arguments, protected-access

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")


def test_import_mpi4py(monkeypatch):
    """Test if an ImportError is raised by _import_mpi4py function."""
    # pylint: disable=protected-access

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "mpi4py", None)

        with pytest.raises(ImportError, match="This feature requires mpi4py"):
            pes_generator._import_mpi4py()


@pytest.mark.parametrize(
    ("sym", "geom", "harmonic_res", "do_dipole", "exp_pes_onemode", "exp_dip_onemode"),
    # Expected results were obtained using vibrant code
    [
        (
            ["H", "F"],
            np.array([[0.0, 0.0, 0.03967368], [0.0, 0.0, 0.96032632]]),
            {
                "freq_wavenumber": np.array([4137.96877864]),
                "norm_mode": np.array(
                    [
                        [
                            [-1.93180788e-17, -2.79300482e-17, -9.70607797e-01],
                            [-3.49162986e-17, 8.71500605e-17, 5.14976259e-02],
                        ]
                    ]
                ),
            },
            False,
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
            None,
        ),
        (
            ["H", "H", "S"],
            np.array(
                [
                    [0.0, -1.00688408, -0.9679942],
                    [0.0, 1.00688408, -0.9679942],
                    [0.0, 0.0, -0.06401159],
                ]
            ),
            {
                "freq_wavenumber": np.array([1294.2195371, 2691.27147945, 2718.4023196]),
                "norm_mode": np.array(
                    [
                        [
                            [5.04812379e-17, -4.56823333e-01, 5.19946505e-01],
                            [1.86137417e-16, 4.56823334e-01, 5.19946504e-01],
                            [1.35223505e-17, -1.52311695e-11, -3.26953260e-02],
                        ],
                        [
                            [-9.48723219e-18, -5.36044948e-01, -4.43104062e-01],
                            [1.58760881e-16, 5.36044952e-01, -4.43104065e-01],
                            [5.31102418e-17, -1.25299135e-10, 2.78633123e-02],
                        ],
                        [
                            [6.52265536e-17, -5.15178992e-01, -4.62528763e-01],
                            [3.12480546e-16, -5.15178988e-01, 4.62528760e-01],
                            [1.63797627e-17, 3.23955347e-02, 9.23972875e-11],
                        ],
                    ]
                ),
            },
            True,
            np.array(
                [
                    [
                        0.03038184,
                        0.0155299,
                        0.00650523,
                        0.00156346,
                        0.0,
                        0.00151837,
                        0.00613061,
                        0.01428086,
                        0.02755765,
                    ],
                    [
                        0.10140357,
                        0.04412527,
                        0.01637343,
                        0.00355301,
                        0.0,
                        0.00290997,
                        0.01083774,
                        0.02329395,
                        0.04121951,
                    ],
                    [
                        0.07198428,
                        0.0340414,
                        0.01374172,
                        0.00326402,
                        0.0,
                        0.00326402,
                        0.01374172,
                        0.0340414,
                        0.07198428,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [-5.36105111e-16, 1.33592345e-10, 9.41527336e-02],
                        [-6.24887513e-16, 8.82111475e-11, 7.79937948e-02],
                        [-5.19872629e-16, 5.05100438e-11, 5.33232767e-02],
                        [-5.39642354e-17, 2.12469860e-11, 2.60979423e-02],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00],
                        [5.11986070e-16, -1.42688961e-11, -2.28485454e-02],
                        [-7.19161722e-16, -2.15452303e-11, -4.12130918e-02],
                        [-5.12026491e-16, -2.06124746e-11, -5.38435444e-02],
                        [9.35736172e-16, -5.75791709e-12, -5.77604214e-02],
                    ],
                    [
                        [-2.22066607e-16, 2.95745910e-10, -9.70203759e-02],
                        [-6.58015868e-16, 2.54133013e-10, -7.42203663e-02],
                        [-5.08918190e-16, 1.83813440e-10, -5.04545990e-02],
                        [1.10366883e-15, 9.69697526e-11, -2.56971336e-02],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00],
                        [-2.85241742e-16, -1.02915824e-10, 2.66325967e-02],
                        [-1.39431150e-17, -2.08111968e-10, 5.43391161e-02],
                        [-1.46190173e-17, -3.11964836e-10, 8.35802521e-02],
                        [1.71204651e-16, -4.11259828e-10, 1.15968472e-01],
                    ],
                    [
                        [-1.04883276e-16, -1.03296190e-01, 1.35696490e-03],
                        [4.87263795e-18, -7.73177418e-02, 8.19496674e-04],
                        [-4.97201877e-16, -5.16458875e-02, 3.84521980e-04],
                        [-5.75774440e-16, -2.58720612e-02, 1.00242797e-04],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00],
                        [3.02208598e-16, 2.58720612e-02, 1.00242982e-04],
                        [9.52676251e-16, 5.16458875e-02, 3.84522344e-04],
                        [4.48188933e-16, 7.73177419e-02, 8.19497209e-04],
                        [-1.04202762e-15, 1.03296190e-01, 1.35696559e-03],
                    ],
                ]
            ),
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support", "skip_if_no_mpi4py_support")
def test_onemode_pes(sym, geom, harmonic_res, do_dipole, exp_pes_onemode, exp_dip_onemode):
    r"""Test that the correct onemode PES is obtained."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = vibrational_class._single_point(mol)

    gauss_grid, _ = np.polynomial.hermite.hermgauss(9)

    freqs = harmonic_res["freq_wavenumber"]
    displ_vecs = harmonic_res["norm_mode"]
    pes_onebody, dipole_onebody = pes_generator._pes_onemode(
        mol, mol_eq, freqs, displ_vecs, gauss_grid, method="RHF", dipole=do_dipole
    )

    assert np.allclose(pes_onebody, exp_pes_onemode, atol=1e-6)
    if do_dipole:
        assert np.allclose(dipole_onebody, exp_dip_onemode, atol=1e-6)
    else:
        assert dipole_onebody is None


@pytest.mark.parametrize(
    ("sym", "geom", "freqs", "vectors", "ref_file"),
    # Expected results were obtained using vibrant code
    [
        (
            ["H", "H", "S"],
            np.array(
                [
                    [0.0, -1.00688408, -0.9679942],
                    [0.0, 1.00688408, -0.9679942],
                    [0.0, 0.0, -0.06401159],
                ]
            ),
            np.array([1294.2195371, 2691.27147945, 2718.4023196]),
            np.array(
                [
                    [
                        [5.04812379e-17, -4.56823333e-01, 5.19946505e-01],
                        [1.86137417e-16, 4.56823334e-01, 5.19946504e-01],
                        [1.35223505e-17, -1.52311695e-11, -3.26953260e-02],
                    ],
                    [
                        [-9.48723219e-18, -5.36044948e-01, -4.43104062e-01],
                        [1.58760881e-16, 5.36044952e-01, -4.43104065e-01],
                        [5.31102418e-17, -1.25299135e-10, 2.78633123e-02],
                    ],
                    [
                        [6.52265536e-17, -5.15178992e-01, -4.62528763e-01],
                        [3.12480546e-16, -5.15178988e-01, 4.62528760e-01],
                        [1.63797627e-17, 3.23955347e-02, 9.23972875e-11],
                    ],
                ]
            ),
            "H2S.hdf5",
        )
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support", "skip_if_no_mpi4py_support")
def test_twomode_pes(sym, geom, freqs, vectors, ref_file):
    r"""Test that the correct twomode PES is obtained."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = vibrational_class._single_point(mol)

    gauss_grid, _ = np.polynomial.hermite.hermgauss(9)

    pes_file = os.path.join(ref_dir, ref_file)
    with h5py.File(pes_file, "r+") as f:
        exp_pes_onebody = np.array(f["V1_PES"][()])
        exp_dip_onebody = np.array(f["D1_DMS"][()])
        exp_pes_twobody = np.array(f["V2_PES"][()])
        exp_dip_twobody = np.array(f["D2_DMS"][()])

    pes_twobody, dipole_twobody = pes_generator._pes_twomode(
        mol,
        mol_eq,
        freqs,
        vectors,
        gauss_grid,
        exp_pes_onebody,
        exp_dip_onebody,
        method="rhf",
        dipole=True,
    )

    assert np.allclose(pes_twobody, exp_pes_twobody, atol=1e-6)
    assert np.allclose(dipole_twobody, exp_dip_twobody, atol=1e-6)


@pytest.mark.parametrize(
    ("sym", "geom", "freqs", "vectors", "ref_file"),
    # Expected results were obtained using vibrant code
    [
        (
            ["H", "H", "S"],
            np.array(
                [
                    [0.0, -1.00688408, -0.9679942],
                    [0.0, 1.00688408, -0.9679942],
                    [0.0, 0.0, -0.06401159],
                ]
            ),
            np.array([1294.2195371, 2691.27147945, 2718.4023196]),
            np.array(
                [
                    [
                        [5.04812379e-17, -4.56823333e-01, 5.19946505e-01],
                        [1.86137417e-16, 4.56823334e-01, 5.19946504e-01],
                        [1.35223505e-17, -1.52311695e-11, -3.26953260e-02],
                    ],
                    [
                        [-9.48723219e-18, -5.36044948e-01, -4.43104062e-01],
                        [1.58760881e-16, 5.36044952e-01, -4.43104065e-01],
                        [5.31102418e-17, -1.25299135e-10, 2.78633123e-02],
                    ],
                    [
                        [6.52265536e-17, -5.15178992e-01, -4.62528763e-01],
                        [3.12480546e-16, -5.15178988e-01, 4.62528760e-01],
                        [1.63797627e-17, 3.23955347e-02, 9.23972875e-11],
                    ],
                ]
            ),
            "H2S.hdf5",
        )
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support", "skip_if_no_mpi4py_support")
def test_threemode_pes(sym, geom, freqs, vectors, ref_file):
    r"""Test that the correct threemode PES is obtained."""

    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = vibrational_class._single_point(mol)

    gauss_grid, _ = np.polynomial.hermite.hermgauss(9)

    pes_file = os.path.join(ref_dir, ref_file)
    with h5py.File(pes_file, "r+") as f:
        exp_pes_onebody = np.array(f["V1_PES"][()])
        exp_dip_onebody = np.array(f["D1_DMS"][()])
        exp_pes_twobody = np.array(f["V2_PES"][()])
        exp_dip_twobody = np.array(f["D2_DMS"][()])
        exp_pes_threebody = np.array(f["V3_PES"][()])
        exp_dip_threebody = np.array(f["D3_DMS"][()])

    pes_threebody, dipole_threebody = pes_generator._pes_threemode(
        mol,
        mol_eq,
        freqs,
        vectors,
        gauss_grid,
        exp_pes_onebody,
        exp_pes_twobody,
        exp_dip_onebody,
        exp_dip_twobody,
        method="rhf",
        dipole=True,
    )

    assert np.allclose(pes_threebody, exp_pes_threebody, atol=1e-6)
    assert np.allclose(dipole_threebody, exp_dip_threebody, atol=1e-6)


def test_quad_order_error():
    r"""Test that an error is raised if invalid value of quad_order is provided."""

    sym = ["H", "F"]
    geom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)

    with pytest.raises(ValueError, match="Number of sample points cannot be less than 1."):
        vibrational.vibrational_pes(mol, quad_order=-1)


def test_dipole_order_error():
    r"""Test that an error is raised if invalid value of dipole_level is provided."""

    sym = ["H", "F"]
    geom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)

    with pytest.raises(
        ValueError,
        match="Currently, one-mode, two-mode and three-mode dipole calculations are supported.",
    ):
        vibrational.vibrational_pes(mol, dipole_level=4)


@pytest.mark.parametrize(
    ("sym", "geom", "dipole_level", "result_file"),
    # Expected results were obtained using vibrant code
    [
        (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 3, "HF.hdf5"),
        (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 1, "HF.hdf5"),
    ],
)
def test_vibrational_pes(sym, geom, dipole_level, result_file):
    r"""Test that vibrational_pes returns correct object."""
    mol = qml.qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)

    vib_obj = vibrational.vibrational_pes(mol, dipole_level=dipole_level, cubic=True)

    pes_file = os.path.join(ref_dir, result_file)
    with h5py.File(pes_file, "r") as f:
        exp_pes_onemode = np.array(f["V1_PES"][()])
        exp_dip_onemode = np.array(f["D1_DMS"][()])
        exp_pes_twomode = np.array(f["V2_PES"][()])
        exp_dip_twomode = np.array(f["D2_DMS"][()])
        exp_pes_threemode = np.array(f["V3_PES"][()])
        exp_dip_threemode = np.array(f["D3_DMS"][()])
        nmodes_expected = len(f["freqs"][()])

    nmodes = len(vib_obj.freqs)

    assert nmodes == nmodes_expected

    for i in range(nmodes):
        assert np.allclose(vib_obj.pes_onemode[i], exp_pes_onemode[i], atol=1e-5) or np.allclose(
            vib_obj.pes_onemode[i], exp_pes_onemode[i][::-1], atol=1e-5
        )
        assert np.allclose(vib_obj.dipole_onemode[i], exp_dip_onemode[i], atol=1e-5) or np.allclose(
            vib_obj.dipole_onemode[i], exp_dip_onemode[i][::-1, :], atol=1e-5
        )
        for j in range(nmodes):
            assert np.allclose(vib_obj.pes_twomode[i, j], exp_pes_twomode[i, j], atol=1e-5)
            if dipole_level > 1:
                assert np.allclose(vib_obj.dipole_twomode[i, j], exp_dip_twomode[i, j], atol=1e-5)
            else:
                assert vib_obj.dipole_twomode is None
            for k in range(nmodes):
                assert np.allclose(
                    vib_obj.pes_threemode[i, j, k], exp_pes_threemode[i, j, k], atol=1e-5
                )
                if dipole_level > 2:
                    assert np.allclose(
                        vib_obj.dipole_threemode[i, j, k], exp_dip_threemode[i, j, k], atol=1e-5
                    )
                else:
                    assert vib_obj.dipole_threemode is None

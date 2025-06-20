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
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from pennylane import qchem
from pennylane.qchem import vibrational
from pennylane.qchem.vibrational import pes_generator
from pennylane.qchem.vibrational.vibrational_class import _single_point

h5py = pytest.importorskip("h5py")

# pylint: disable=too-many-arguments, protected-access, too-many-positional-arguments, unsubscriptable-object

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")


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
                            [-1.93180788e-17, -2.79300482e-17, 9.70607797e-01],
                            [-3.49162986e-17, 8.71500605e-17, -5.14976259e-02],
                        ]
                    ]
                ),
            },
            False,
            np.array(
                [
                    [
                        0.05235573,
                        0.03093067,
                        0.01501878,
                        0.00420778,
                        0.0,
                        0.00584504,
                        0.02881817,
                        0.08483433,
                        0.22025702,
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
                        0.03039086,
                        0.01553468,
                        0.00650726,
                        0.00156395,
                        0.0,
                        0.00151883,
                        0.00613247,
                        0.01428514,
                        0.02756578,
                    ],
                    [
                        0.10144312,
                        0.04414136,
                        0.01637907,
                        0.00355417,
                        0.0,
                        0.00291082,
                        0.01084077,
                        0.02330013,
                        0.04122984,
                    ],
                    [
                        0.07200939,
                        0.03405263,
                        0.01374609,
                        0.00326503,
                        0.0,
                        0.00326503,
                        0.0137461,
                        0.03405266,
                        0.07200947,
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [-6.78930174e-16, -1.01245409e-10, 9.41568263e-02],
                        [6.01201132e-17, -6.68733709e-11, 7.80028335e-02],
                        [-4.30141004e-17, -3.82773113e-11, 5.33312174e-02],
                        [-3.71789109e-16, -1.62100894e-11, 2.61021033e-02],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00],
                        [-5.85471753e-17, 1.07225534e-11, -2.28517326e-02],
                        [2.01004792e-16, 1.63063665e-11, -4.12177809e-02],
                        [-9.23333645e-16, 1.55660206e-11, -5.38473277e-02],
                        [-8.54694312e-17, 4.27542438e-12, -5.77590817e-02],
                    ],
                    [
                        [5.87707846e-16, -3.63943212e-08, -9.70309622e-02],
                        [-3.87358464e-16, -3.12726065e-08, -7.42300553e-02],
                        [1.42378971e-16, -2.26104689e-08, -5.04618120e-02],
                        [1.23018666e-16, -1.19384577e-08, -2.57010020e-02],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00],
                        [-4.58645168e-16, 1.26721705e-08, 2.66367562e-02],
                        [-1.60659081e-16, 2.56128867e-08, 5.43475419e-02],
                        [1.74895673e-17, 3.83835753e-08, 8.35928980e-02],
                        [-3.45285283e-18, 5.06060763e-08, 1.15985191e-01],
                    ],
                    [
                        [1.06933714e-15, -1.03308767e-01, 1.35728940e-03],
                        [-1.18946608e-16, -7.73284468e-02, 8.19737833e-04],
                        [-4.56874373e-16, -5.16535304e-02, 3.84656298e-04],
                        [-5.35146567e-16, -2.58760228e-02, 1.00285612e-04],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00],
                        [-2.91851154e-16, 2.58760221e-02, 1.00260934e-04],
                        [-6.59129376e-16, 5.16535272e-02, 3.84607485e-04],
                        [2.79961294e-16, 7.73284385e-02, 8.19665999e-04],
                        [4.74399424e-17, 1.03308749e-01, 1.35719641e-03],
                    ],
                ]
            ),
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_onemode_pes(sym, geom, harmonic_res, do_dipole, exp_pes_onemode, exp_dip_onemode):
    r"""Test that the correct onemode PES is obtained."""

    mol = qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = _single_point(mol)

    gauss_grid, _ = np.polynomial.hermite.hermgauss(9)

    freqs = harmonic_res["freq_wavenumber"]
    displ_vecs = harmonic_res["norm_mode"]
    with TemporaryDirectory() as tmpdir:
        pes_onebody, dipole_onebody = pes_generator._pes_onemode(
            mol, mol_eq, freqs, displ_vecs, gauss_grid, method="RHF", dipole=do_dipole, path=tmpdir
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
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_twomode_pes(sym, geom, freqs, vectors, ref_file):
    r"""Test that the correct twomode PES is obtained."""

    mol = qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = _single_point(mol)

    gauss_grid, _ = np.polynomial.hermite.hermgauss(9)

    pes_file = os.path.join(ref_dir, ref_file)
    with h5py.File(pes_file, "r+") as f:
        exp_pes_onebody = np.array(f["V1_PES"][()])
        exp_dip_onebody = np.array(f["D1_DMS"][()])
        exp_pes_twobody = np.array(f["V2_PES"][()])
        exp_dip_twobody = np.array(f["D2_DMS"][()])
    with TemporaryDirectory() as tmpdir:
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
            path=tmpdir,
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
@pytest.mark.usefixtures("skip_if_no_pyscf_support")
def test_threemode_pes(sym, geom, freqs, vectors, ref_file):
    r"""Test that the correct threemode PES is obtained."""

    mol = qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)
    mol_eq = _single_point(mol)

    gauss_grid, _ = np.polynomial.hermite.hermgauss(9)

    pes_file = os.path.join(ref_dir, ref_file)
    with h5py.File(pes_file, "r+") as f:
        exp_pes_onebody = np.array(f["V1_PES"][()])
        exp_dip_onebody = np.array(f["D1_DMS"][()])
        exp_pes_twobody = np.array(f["V2_PES"][()])
        exp_dip_twobody = np.array(f["D2_DMS"][()])
        exp_pes_threebody = np.array(f["V3_PES"][()])
        exp_dip_threebody = np.array(f["D3_DMS"][()])
    with TemporaryDirectory() as tmpdir:
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
            path=tmpdir,
        )

        assert np.allclose(pes_threebody, exp_pes_threebody, atol=1e-6)
        assert np.allclose(dipole_threebody, exp_dip_threebody, atol=1e-6)


def test_quad_order_error():
    r"""Test that an error is raised if invalid value of n_points is provided."""

    sym = ["H", "F"]
    geom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mol = qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)

    with pytest.raises(ValueError, match="Number of sample points cannot be less than 1."):
        vibrational.vibrational_pes(mol, n_points=-1)


def test_dipole_order_error():
    r"""Test that an error is raised if invalid value of dipole_level is provided."""

    sym = ["H", "F"]
    geom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mol = qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)

    with pytest.raises(
        ValueError,
        match="Currently, one-mode, two-mode and three-mode dipole calculations are supported.",
    ):
        vibrational.vibrational_pes(mol, dipole_level=4)


@pytest.mark.parametrize(
    ("sym", "geom", "dipole_level", "result_file", "max_workers", "backend"),
    # Expected results were obtained using vibrant code
    [
        (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 3, "HF.hdf5", 1, "serial"),
        (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 1, "HF.hdf5", 2, "mp_pool"),
        (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 3, "HF.hdf5", 2, "cf_procpool"),
        (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 1, "HF.hdf5", 2, "mpi4py_pool"),
        (["H", "F"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), 3, "HF.hdf5", 2, "mpi4py_comm"),
    ],
)
def test_vibrational_pes(
    sym, geom, dipole_level, result_file, backend, max_workers, mpi4py_support
):
    r"""Test that vibrational_pes returns correct object."""

    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")

    mol = qchem.Molecule(sym, geom, basis_name="6-31g", unit="Angstrom", load_data=True)

    vib_obj = vibrational.vibrational_pes(
        mol, dipole_level=dipole_level, cubic=True, num_workers=max_workers, backend=backend
    )

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


def test_optimize_false():
    r"""Test that VibrationalPES is constructed when geometry optimization is not requested."""

    symbols = ["H", "F"]
    geometry = np.array([[0.0, 0.0, -0.40277116], [0.0, 0.0, 1.40277116]])
    mol = qchem.Molecule(symbols, geometry)
    pes = qchem.vibrational_pes(mol, optimize=False)

    assert len(pes.freqs) == 1

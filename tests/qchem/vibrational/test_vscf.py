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
This module contains tests for functions needed to do vscf calculations.
"""
from pathlib import Path

import numpy as np
import pytest

from pennylane.qchem import vibrational
from pennylane.qchem.vibrational import vscf

# pylint: disable=protected-access

h5py = pytest.importorskip("h5py")

cform_file = Path(__file__).resolve().parent / "test_ref_files" / "H2S.hdf5"

with h5py.File(cform_file, "r") as f:
    h1_h2s = f["H1"][()]
    h2_h2s = f["H2"][()]
    h3_h2s = f["H3"][()]
    d1_h2s = f["D1"][()]
    d2_h2s = f["D2"][()]
    d3_h2s = f["D3"][()]
h_data_h2s = [h1_h2s, h2_h2s, h3_h2s]
dip_data_h2s = [d1_h2s, d2_h2s, d3_h2s]

result_file = Path(__file__).resolve().parent / "test_ref_files" / "H2S_vscf_result.hdf5"

with h5py.File(result_file, "r") as f:
    h1_h2s = f["H1_rot"][()]
    h2_h2s = f["H2_rot"][()]
    h3_h2s = f["H3_rot"][()]
    d1_h2s = f["D1_rot"][()]
    d2_h2s = f["D2_rot"][()]
    d3_h2s = f["D3_rot"][()]
    energy = f["energy"][()]
    u_mat = f["u_mat"][()]

h2s_exp_result = {
    "h_data": [h1_h2s, h2_h2s, h3_h2s],
    "dip_data": [d1_h2s, d2_h2s, d3_h2s],
    "energy": energy,
    "u_mat": u_mat,
}


def test_error_vscf_hamiltonian():
    r"""Test that an error is raised if wrong shape is provided for Hamiltonian"""
    with pytest.raises(
        ValueError, match="Building n-mode Hamiltonian is not implemented for n equal to 4"
    ):
        vibrational.vscf_hamiltonian(ham_data=[1, 2, 3, 4])


def test_error_vscf_dipole():
    r"""Test that an error is raised if wrong shape is provided for dipole."""
    with pytest.raises(
        ValueError, match="Building n-mode dipole is not implemented for n equal to 4"
    ):
        vibrational.vscf_hamiltonian(ham_data=[1, 2, 3], dipole_data=[1, 2, 3, 4])


@pytest.mark.parametrize(
    ("h_data"),
    [
        (h_data_h2s),
    ],
)
def test_modal_error(h_data):
    r"""Test that an error is raised if number of modals provided is incorrect"""

    with pytest.raises(
        ValueError,
        match="Number of maximum modals cannot be greater than the modals for unrotated integrals.",
    ):
        vibrational.vscf_hamiltonian(ham_data=h_data, modals=[5, 5, 5])


@pytest.mark.parametrize(
    ("h_data", "h2s_result"),
    [
        (h_data_h2s, h2s_exp_result),
    ],
)
def test_vscf_calculation(h_data, h2s_result):
    r"""Test that vscf calculation produces correct energy and rotation matrices"""

    vib_energy, rot_matrix = vscf._vscf(h_data, modals=[4, 4, 4], cutoff=1e-8)
    assert np.isclose(vib_energy, h2s_result["energy"])
    assert np.allclose(rot_matrix, h2s_result["u_mat"])


@pytest.mark.parametrize(
    ("h_data", "dip_data", "h2s_result"),
    [
        (h_data_h2s, dip_data_h2s, h2s_exp_result),
    ],
)
def test_vscf_hamiltonian_dipole(h_data, dip_data, h2s_result):
    r"""Test that correct rotated Hamiltonian and dipole is produced."""
    result_h, result_dip = vscf.vscf_hamiltonian(h_data, dip_data, modals=[3, 3, 3], cutoff=1e-8)

    expected_h = h2s_result["h_data"]
    expected_dip = h2s_result["dip_data"]
    assert np.allclose(result_h[0], expected_h[0])
    assert np.allclose(result_h[1], expected_h[1])
    assert np.allclose(result_h[2], expected_h[2])
    assert np.allclose(result_dip[0], expected_dip[0])
    assert np.allclose(result_dip[1], expected_dip[1])
    assert np.allclose(result_dip[2], expected_dip[2])


@pytest.mark.parametrize(
    ("h_data", "h2s_result"),
    [
        (h_data_h2s, h2s_exp_result),
    ],
)
def test_vscf_hamiltonian(h_data, h2s_result):
    r"""Test that correct rotated Hamiltonian is produced."""
    result_h, result_dip = vscf.vscf_hamiltonian(h_data, modals=[3, 3, 3], cutoff=1e-8)

    expected_h = h2s_result["h_data"]

    assert np.allclose(result_h[0], expected_h[0])
    assert np.allclose(result_h[1], expected_h[1])
    assert np.allclose(result_h[2], expected_h[2])

    assert result_dip is None

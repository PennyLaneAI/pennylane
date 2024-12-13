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
"""Unit Tests for the Christiansen Hamiltonian construction functions."""

from pathlib import Path

import pytest
import numpy as np
from pennylane.bose.bosonic import BoseWord
from pennylane.qchem.vibrational.vibrational_class import VibrationalPES
from tests.qchem.vibrational.test_ref_files.cform_ops_data import (
    cform_ham_ref,
    cform_ops_ref,
    cform_coeffs_ref,
    cform_dipole_ref_x,
)


from pennylane.labs.vibrational.christiansen_ham import (
    christiansen_bosonic,
    christiansen_hamiltonian,
    christiansen_dipole,
)

from pennylane.labs.vibrational.christiansen_utils import (
    christiansen_integrals,
    christiansen_integrals_dipole,
    _cform_onemode,
    _cform_onemode_dipole,
    _cform_threemode,
    _cform_threemode_dipole,
    _cform_twomode,
    _cform_twomode_dipole,
    _load_cform_onemode,
)

# Path is pennylane/tests/qchem/vibrational/test_ref_files/H2S.hdf5
cform_file = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "tests"
    / "qchem"
    / "vibrational"
    / "test_ref_files"
    / "H2S.hdf5"
)

# Path is pennylane/tests/qchem/vibrational/test_ref_files/H2S_3D_PES.hdf5
pes_file = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "tests"
    / "qchem"
    / "vibrational"
    / "test_ref_files"
    / "H2S_3D_PES.hdf5"
)

h5py = pytest.importorskip("h5py")

with h5py.File(pes_file, "r") as f:
    pes_onemode = np.array(f["pes_onemode"][()])
    pes_twomode = np.array(f["pes_twomode"][()])
    pes_threemode = np.array(f["pes_threemode"][()])

    dipole_onemode = np.array(f["dipole_onemode"][()])
    dipole_twomode = np.array(f["dipole_twomode"][()])
    dipole_threemode = np.array(f["dipole_threemode"][()])
    pes_object_3D = VibrationalPES(
        freqs=np.array(f["freqs"][()]),
        grid=np.array(f["grid"][()]),
        uloc=np.array(f["uloc"][()]),
        gauss_weights=np.array(f["gauss_weights"][()]),
        pes_data=[pes_onemode, pes_twomode, pes_threemode],
        dipole_data=[dipole_onemode, dipole_twomode, dipole_threemode],
        localized=f["localized"][()],
        dipole_level=f["dipole_level"][()],
    )

    pes_object_2D = VibrationalPES(
        freqs=np.array(f["freqs"][()]),
        grid=np.array(f["grid"][()]),
        uloc=np.array(f["uloc"][()]),
        gauss_weights=np.array(f["gauss_weights"][()]),
        pes_data=[pes_onemode, pes_twomode],
        dipole_data=[dipole_onemode, dipole_twomode],
        localized=f["localized"][()],
        dipole_level=2,
    )

with h5py.File(cform_file, "r") as f:
    H1 = f["H1"][()]
    H2 = f["H2"][()]
    H3 = f["H3"][()]
    D1 = f["D1"][()]
    D2 = f["D2"][()]
    D3 = f["D3"][()]


def test_christiansen_bosonic():
    """Test that christiansen_bosonic produces the correct bosonic operator."""
    christiansen_bos_op = christiansen_bosonic(one=H1, two=H2, three=H3)
    christiansen_bos_op.simplify()

    ops, coeffs = zip(*list(christiansen_bos_op.items()))

    for i, ele in enumerate(cform_ops_ref):
        cform_ops_ref[i] = BoseWord(ele)

    assert list(ops) == cform_ops_ref
    assert np.allclose(coeffs, cform_coeffs_ref, atol=1e-10)
    assert len(christiansen_bos_op) == len(cform_ops_ref)


def test_christiansen_hamiltonian():
    """Test that christiansen_hamiltonian produces the expected hamiltonian."""
    cform_ham = christiansen_hamiltonian(pes=pes_object_3D, n_states=4, cubic=True)
    cform_ham.simplify()
    assert len(cform_ham.pauli_rep) == len(cform_ham_ref)
    assert all(
        np.allclose(abs(cform_ham_ref.pauli_rep[term]), abs(coeff), atol=1e-8)
        for term, coeff in cform_ham.pauli_rep.items()
    )


def test_christiansen_dipole():
    """Test that christiansen_dipole produces the expected dipole operator coefficients."""
    cform_dipole_x, _, _ = christiansen_dipole(pes=pes_object_2D, n_states=4)
    assert len(cform_dipole_x.pauli_rep) == len(cform_dipole_ref_x)
    assert all(
        np.allclose(abs(cform_dipole_ref_x.pauli_rep[term]), abs(coeff), atol=1e-8)
        for term, coeff in cform_dipole_x.pauli_rep.items()
    )


def test_christiansen_integrals():
    one, two, three = christiansen_integrals(pes=pes_object_3D, n_states=4, cubic=True)
    assert np.allclose(abs(one), abs(H1), atol=1e-8)
    assert np.allclose(abs(two), abs(H2), atol=1e-8)
    assert np.allclose(abs(three), abs(H3), atol=1e-8)


def test_christiansen_integrals_dipole():
    one, two, three = christiansen_integrals_dipole(pes=pes_object_3D, n_states=4)
    assert np.allclose(abs(one), abs(D1), atol=1e-8)
    assert np.allclose(abs(two), abs(D2), atol=1e-8)
    assert np.allclose(abs(three), abs(D3), atol=1e-8)


def test_cform_onemode():
    flattened_H1 = H1.ravel()
    assert np.allclose(
        abs(flattened_H1), abs(_cform_onemode(pes=pes_object_3D, n_states=4)), atol=1e-8
    )


def test_cform_onemode_dipole():
    flattened_D1 = D1.transpose(1, 2, 3, 0).ravel()
    assert np.allclose(
        abs(flattened_D1),
        abs(_cform_onemode_dipole(pes=pes_object_3D, n_states=4).ravel()),
        atol=1e-8,
    )

def test_cform_threemode():
    flattened_H3 = H3.ravel()
    assert np.allclose(
        abs(flattened_H3), abs(_cform_threemode(pes=pes_object_3D, n_states=4)), atol=1e-8
    )


def test_cform_threemode_dipole():
    flattened_D3 = D3.transpose(1, 2, 3, 4, 5, 6, 7, 8, 9, 0).ravel()

    assert np.allclose(
        abs(flattened_D3),
        abs(_cform_threemode_dipole(pes=pes_object_3D, n_states=4).ravel()),
        atol=1e-8,
    )


def test_cform_twomode():
    flattened_H2 = H2.ravel()
    assert np.allclose(
        abs(flattened_H2), abs(_cform_twomode(pes=pes_object_3D, n_states=4)), atol=1e-8
    )


def test_cform_twomode_dipole():
    flattened_D2 = D2.transpose(1, 2, 3, 4, 5, 6, 0).ravel()

    assert np.allclose(
        abs(flattened_D2),
        abs(_cform_twomode_dipole(pes=pes_object_3D, n_states=4).ravel()),
        atol=1e-8,
    )

def test_load_cform_onemode():
    data = H1.ravel()
    # We have to create a file to test the loader
    with h5py.File("cform_H1data" + f"_{0}" + ".hdf5", "w") as f:
        f.create_dataset("H1", data=data)
    assert np.allclose(
        abs(H1), (abs(_load_cform_onemode(num_proc=1, nmodes=3, quad_order=4))), atol=1e-8
    )

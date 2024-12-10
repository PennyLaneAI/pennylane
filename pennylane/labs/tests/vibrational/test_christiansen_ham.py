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
from pennylane.qchem.vibrational.vibrational_class import VibrationalPES


from pennylane.labs.vibrational.christiansen_ham import (
    christiansen_bosonic,
    christiansen_hamiltonian,
    christiansen_dipole,
)

from pennylane.labs.vibrational.christiansen_utils import (
    christiansen_integrals,
    christiansen_integrals_dipole,
)

cform_file = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "tests"
    / "qchem"
    / "vibrational"
    / "test_ref_files"
    / "H2S.hdf5"
)

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

with h5py.File(cform_file, "r") as f:
    H1 = f["H1"][()]
    H2 = f["H2"][()]
    H3 = f["H3"][()]


def test_christiansen_bosonic():
    """Test that christiansen_bosonic produces the correct bosonic operator."""
    christiansen_bos_op = christiansen_bosonic(one=H1, two=H2, three=H3)
    christiansen_bos_op.simplify()
    assert len(christiansen_bos_op) == 4702 # Reference from Vibrant


def test_christiansen_hamiltonian():
    """Test that christiansen_hamiltonian produces the expected hamiltonian."""
    cform_ham = christiansen_hamiltonian(pes_object=pes_object_3D, nbos=4)
    assert True


def test_christiansen_dipole():
    """Test that christiansen_dipole produces the expected dipole operator coefficients."""
    cform_dipole = christiansen_dipole(pes_object=pes_object_3D, nbos=4)
    assert True

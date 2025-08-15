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
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from pennylane.bose.bosonic import BoseWord
from pennylane.qchem.vibrational.christiansen_ham import (
    christiansen_bosonic,
    christiansen_dipole,
    christiansen_hamiltonian,
)
from pennylane.qchem.vibrational.christiansen_utils import (
    _cform_onemode,
    _cform_onemode_dipole,
    _cform_threemode,
    _cform_threemode_dipole,
    _cform_twomode,
    _cform_twomode_dipole,
    _read_data,
    _write_data,
    christiansen_integrals,
    christiansen_integrals_dipole,
)
from pennylane.qchem.vibrational.vibrational_class import VibrationalPES

from .test_ref_files.cform_ops_data import (
    cform_coeffs_ref,
    cform_dipole_ref_x,
    cform_ham_ref,
    cform_ops_ref,
)

cform_file = Path(__file__).resolve().parent / "test_ref_files" / "H2S.hdf5"

pes_file = Path(__file__).resolve().parent / "test_ref_files" / "H2S_3D_PES.hdf5"

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


@pytest.mark.parametrize(
    "ordered",
    [
        True,
        False,
    ],
)
def test_christiansen_bosonic(ordered):
    """Test that christiansen_bosonic produces the correct bosonic operator."""
    christiansen_bos_op = christiansen_bosonic(one=H1, two=H2, three=H3, ordered=ordered)
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
    assert len(cform_ham.pauli_rep) == 4160

    # Tolerance is low here because values smaller than 1e-5 get converted to 0.0
    # We only compare a small percentage of the Hamiltonian since there are many terms
    assert all(
        np.allclose(abs(cform_ham_ref.pauli_rep[term]), abs(coeff), atol=1e-4)
        for term, coeff in list(cform_ham.pauli_rep.items())[:84]
    )


def test_christiansen_dipole():
    """Test that christiansen_dipole produces the expected dipole operator coefficients."""
    cform_dipole_x, _, _ = christiansen_dipole(pes=pes_object_2D, n_states=4)
    assert len(cform_dipole_x.pauli_rep) == len(cform_dipole_ref_x)
    assert all(
        np.allclose(abs(cform_dipole_ref_x.pauli_rep[term]), abs(coeff), atol=1e-8)
        for term, coeff in cform_dipole_x.pauli_rep.items()
    )


@pytest.mark.parametrize(
    ("pes", "n_states", "num_workers", "backend"),
    [
        (pes_object_3D, 4, 1, "serial"),
        (pes_object_3D, 4, 2, "mp_pool"),
        (pes_object_3D, 4, 2, "cf_procpool"),
        (pes_object_3D, 4, 2, "mpi4py_pool"),
        (pes_object_3D, 4, 2, "mpi4py_comm"),
    ],
)
def test_christiansen_integrals(pes, n_states, num_workers, backend, mpi4py_support):
    """Test that christiansen_integrals produces the expected integrals."""
    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")

    one, two, three = christiansen_integrals(
        pes=pes, n_states=n_states, cubic=True, num_workers=num_workers, backend=backend
    )
    assert np.allclose(abs(one), abs(H1), atol=1e-8)
    assert np.allclose(abs(two), abs(H2), atol=1e-8)
    assert np.allclose(abs(three), abs(H3), atol=1e-8)


@pytest.mark.parametrize(
    ("pes", "n_states", "num_workers", "backend"),
    [
        (pes_object_3D, 4, 1, "serial"),
        (pes_object_3D, 4, 2, "mp_pool"),
        (pes_object_3D, 4, 2, "cf_procpool"),
        (pes_object_3D, 4, 2, "mpi4py_pool"),
        (pes_object_3D, 4, 2, "mpi4py_comm"),
    ],
)
def test_christiansen_integrals_dipole(pes, n_states, num_workers, backend, mpi4py_support):
    """Test that christiansen_integrals_dipole produces the expected dipole integrals."""
    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")

    one, two, three = christiansen_integrals_dipole(
        pes=pes, n_states=n_states, num_workers=num_workers, backend=backend
    )
    assert np.allclose(abs(one), abs(D1), atol=1e-8)
    assert np.allclose(abs(two), abs(D2), atol=1e-8)
    assert np.allclose(abs(three), abs(D3), atol=1e-8)


@pytest.mark.parametrize(
    ("pes", "n_states", "num_workers", "backend"),
    [
        (pes_object_3D, 4, 1, "serial"),
        (pes_object_3D, 4, 2, "mp_pool"),
        (pes_object_3D, 4, 2, "cf_procpool"),
        (pes_object_3D, 4, 2, "mpi4py_pool"),
        (pes_object_3D, 4, 2, "mpi4py_comm"),
    ],
)
def test_cform_onemode(pes, n_states, num_workers, backend, mpi4py_support):
    """Test that _cform_onemode produces the expected one-body integral."""
    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")

    with TemporaryDirectory() as tmpdir:
        assert np.allclose(
            abs(H1),
            abs(
                _cform_onemode(
                    pes=pes,
                    n_states=n_states,
                    num_workers=num_workers,
                    backend=backend,
                    path=tmpdir,
                )
            ),
            atol=1e-8,
        )


@pytest.mark.parametrize(
    ("pes", "n_states", "num_workers", "backend"),
    [
        (pes_object_3D, 4, 1, "serial"),
        (pes_object_3D, 4, 2, "mp_pool"),
        (pes_object_3D, 4, 2, "cf_procpool"),
        (pes_object_3D, 4, 2, "mpi4py_pool"),
        (pes_object_3D, 4, 2, "mpi4py_comm"),
    ],
)
def test_cform_onemode_dipole(pes, n_states, num_workers, backend, mpi4py_support):
    """Test that _cform_onemode_dipole produces the expected one-body dipole integral."""
    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")
    with TemporaryDirectory() as tmpdir:
        assert np.allclose(
            abs(D1),
            abs(
                _cform_onemode_dipole(
                    pes=pes,
                    n_states=n_states,
                    num_workers=num_workers,
                    backend=backend,
                    path=tmpdir,
                )
            ),
            atol=1e-8,
        )


@pytest.mark.parametrize(
    ("pes", "n_states", "num_workers", "backend"),
    [
        (pes_object_3D, 4, 1, "serial"),
        (pes_object_3D, 4, 2, "mp_pool"),
        (pes_object_3D, 4, 2, "cf_procpool"),
        (pes_object_3D, 4, 2, "mpi4py_pool"),
        (pes_object_3D, 4, 2, "mpi4py_comm"),
    ],
)
def test_cform_threemode(pes, n_states, num_workers, backend, mpi4py_support):
    """Test that _cform_threemode produces the expected three-body integral."""
    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")
    with TemporaryDirectory() as tmpdir:
        assert np.allclose(
            abs(H3),
            abs(
                _cform_threemode(
                    pes=pes,
                    n_states=n_states,
                    num_workers=num_workers,
                    backend=backend,
                    path=tmpdir,
                )
            ),
            atol=1e-8,
        )


@pytest.mark.parametrize(
    ("pes", "n_states", "num_workers", "backend"),
    [
        (pes_object_3D, 4, 1, "serial"),
        (pes_object_3D, 4, 2, "mp_pool"),
        (pes_object_3D, 4, 2, "cf_procpool"),
        (pes_object_3D, 4, 2, "mpi4py_pool"),
        (pes_object_3D, 4, 2, "mpi4py_comm"),
    ],
)
def test_cform_threemode_dipole(pes, n_states, num_workers, backend, mpi4py_support):
    """Test that _cform_threemode_dipole produces the expected three-body dipole integral."""
    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")
    with TemporaryDirectory() as tmpdir:
        assert np.allclose(
            abs(D3),
            abs(
                _cform_threemode_dipole(
                    pes=pes,
                    n_states=n_states,
                    num_workers=num_workers,
                    backend=backend,
                    path=tmpdir,
                )
            ),
            atol=1e-8,
        )


@pytest.mark.parametrize(
    ("pes", "n_states", "num_workers", "backend"),
    [
        (pes_object_3D, 4, 1, "serial"),
        (pes_object_3D, 4, 2, "mp_pool"),
        (pes_object_3D, 4, 2, "cf_procpool"),
        (pes_object_3D, 4, 2, "mpi4py_pool"),
        (pes_object_3D, 4, 2, "mpi4py_comm"),
    ],
)
def test_cform_twomode(pes, n_states, num_workers, backend, mpi4py_support):
    """Test that _cform_twomode produces the expected two-body integral."""
    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")
    with TemporaryDirectory() as tmpdir:
        assert np.allclose(
            abs(H2),
            abs(
                _cform_twomode(
                    pes=pes,
                    n_states=n_states,
                    num_workers=num_workers,
                    backend=backend,
                    path=tmpdir,
                )
            ),
            atol=1e-8,
        )


@pytest.mark.parametrize(
    ("pes", "n_states", "num_workers", "backend"),
    [
        (pes_object_3D, 4, 1, "serial"),
        (pes_object_3D, 4, 2, "mp_pool"),
        (pes_object_3D, 4, 2, "cf_procpool"),
        (pes_object_3D, 4, 2, "mpi4py_pool"),
        (pes_object_3D, 4, 2, "mpi4py_comm"),
    ],
)
def test_cform_twomode_dipole(pes, n_states, num_workers, backend, mpi4py_support):
    """Test that _cform_twomode_dipole produces the expected two-body dipole integral."""
    if backend in {"mpi4py_pool", "mpi4py_comm"} and not mpi4py_support:
        pytest.skip(f"Skipping test: '{backend}' requires mpi4py, which is not installed.")
    with TemporaryDirectory() as tmpdir:
        assert np.allclose(
            abs(D2),
            abs(
                _cform_twomode_dipole(
                    pes=pes,
                    n_states=n_states,
                    num_workers=num_workers,
                    backend=backend,
                    path=tmpdir,
                )
            ),
            atol=1e-8,
        )


def test_write_and_read_data():
    """Test that _read_data return the data written using _write_data with the same args."""
    with TemporaryDirectory() as tmpdirname:
        rank = 0
        file_name = "testfile"
        dataset_name = "testdata"
        original_data = np.array([1, 2, 3, 4, 5])
        _write_data(tmpdirname, rank, file_name, dataset_name, original_data)
        read_data = _read_data(tmpdirname, rank, file_name, dataset_name)
        np.testing.assert_array_equal(original_data, read_data)

# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the :class:`pennylane.data.Dataset` class and its functions.
"""
# pylint:disable=protected-access
from copy import copy
import os
import sys
import pytest
from pennylane import data
import pennylane as qml

pytestmark = pytest.mark.data

torch = pytest.importorskip("zarr")


def test_build_dataset():
    """Test that a dataset builds correctly and returns the correct values."""
    hamiltonian = qml.Hamiltonian(coeffs=[1], observables=[qml.PauliZ(wires=0)])
    test_dataset = data.Dataset(kw1=1, kw2="2", kw3=[3], hamiltonian=hamiltonian)

    assert test_dataset.kw1 == 1
    assert test_dataset.kw2 == "2"
    assert test_dataset.kw3 == [3]
    assert test_dataset.hamiltonian == hamiltonian


def test_write_dataset(tmp_path):
    """Test that datasets are saved correctly."""
    test_dataset = data.Dataset(kw1=1, kw2="2", kw3=[3])
    d = tmp_path / "sub"
    p = d / "test_dataset"
    test_dataset.write(p)


def test_read_dataset(tmp_path):
    """Test that datasets are loaded correctly."""
    test_dataset = data.Dataset(kw1=1, kw2="2", kw3=[3])
    d = tmp_path / "sub"
    p = d / "test_dataset"
    test_dataset.write(p)

    test_dataset = data.Dataset()
    test_dataset.read(p)

    assert test_dataset.kw1 == 1
    assert test_dataset.kw2 == "2"
    assert test_dataset.kw3 == [3]


def test_list_attributes():
    """Test the list_attributes method."""
    test_dataset = data.Dataset(kw1=1)
    assert test_dataset.list_attributes() == ["kw1"]


def test_copy_non_standard():
    """Test that datasets can be built from other datasets."""
    test_dataset = data.Dataset(dtype="test_data", kw1=1, kw2="2", kw3=[3])
    new_dataset = copy(test_dataset)
    assert new_dataset.attrs == test_dataset.attrs


def test_copy_non_standard_with_lazy(tmp_path):
    """Test that non-standard datasets copy lazy-loading metadata."""
    filepath = str(tmp_path / "copyfile.dat")
    test_dataset = data.Dataset(foo=1, bar=2)
    test_dataset.write(filepath)

    base_dataset = data.Dataset()
    base_dataset.read(filepath, lazy=True)
    assert base_dataset.attrs == {"foo": None, "bar": None}

    copied_dataset = copy(base_dataset)
    assert copied_dataset.attrs == {"foo": None, "bar": None}
    assert copied_dataset._attr_filemap == {"foo": (filepath, True), "bar": (filepath, True)}
    assert copied_dataset.foo == 1
    assert copied_dataset._attr_filemap == {"bar": (filepath, True)}
    assert base_dataset._attr_filemap == {"foo": (filepath, True), "bar": (filepath, True)}


def test_copy_standard(tmp_path):
    """Test that standard datasets can be built from other standard datasets."""
    filepath = tmp_path / "myset_full.dat"
    data.Dataset._write_file({"molecule": 1, "hf_state": 2}, str(filepath))
    test_dataset = data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
    new_dataset = copy(test_dataset)

    assert new_dataset._is_standard == test_dataset._is_standard
    assert new_dataset._dtype == test_dataset._dtype
    assert new_dataset._folder == test_dataset._folder
    assert new_dataset._attr_filemap == test_dataset._attr_filemap
    assert new_dataset._attr_filemap is not test_dataset._attr_filemap
    assert new_dataset._fullfile == test_dataset._fullfile
    assert new_dataset.__doc__ == test_dataset.__doc__
    assert new_dataset.attrs == test_dataset.attrs
    assert new_dataset.attrs == {"molecule": None, "hf_state": None}
    assert new_dataset.molecule == 1
    assert new_dataset.hf_state == 2
    assert new_dataset.attrs == {"molecule": 1, "hf_state": 2}


def test_getattribute_dunder_non_full(tmp_path):
    """Test the getattribute override."""
    non_standard_dataset = data.Dataset(foo="bar")
    with pytest.raises(AttributeError):
        _ = non_standard_dataset.baz

    folder = tmp_path / "datasets" / "myset"

    # would not usually be done by users, bypassing qml.data.load
    standard_dataset = data.Dataset("qchem", str(folder), "myset", "", standard=True)

    # no hf_state file exists (yet!)
    with pytest.raises(AttributeError):
        _ = standard_dataset.hf_state
    # create an hf_state file
    os.makedirs(folder)
    filepath = str(folder / "myset_hf_state.dat")
    data.Dataset._write_file(2, filepath)
    # getattribute does not try to find files that have not yet been read
    with pytest.raises(AttributeError):
        _ = standard_dataset.hf_state


def test_getattribute_dunder_full(tmp_path):
    """Test the getattribute behaviour when a fullfile is set."""
    folder = tmp_path / "datasets" / "myset"
    os.makedirs(folder)
    data.Dataset._write_file({"hf_state": 2}, str(folder / "myset_full.dat"))

    # this getattribute will read from the above created file
    dataset = data.Dataset("qchem", str(folder), "myset", "", standard=True)
    assert dataset.hf_state == 2
    with pytest.raises(AttributeError):
        _ = dataset.molecule


def test_getattribute_fail_when_attribute_deleted_from_file(tmp_path):
    """Test that getattribute fails if an expected attribute is not in a data file."""
    filename = str(tmp_path / "dest.dat")
    dataset = data.Dataset(foo=1, bar=2)
    dataset.write(filename)

    new_dataset = data.Dataset()
    new_dataset.read(filename, lazy=True)
    assert new_dataset.attrs == {"foo": None, "bar": None}

    del dataset.foo
    dataset.write(filename)  # overwrite the shared data file, deleting foo
    assert new_dataset.bar == 2
    with pytest.raises(data.Dataset.DatasetLoadError, match="no longer appears to be in the file"):
        _ = new_dataset.foo


def test_none_attribute_value(tmp_path):
    """Test that both non-standard and standard datasets allow None values."""
    non_standard_dataset = data.Dataset(molecule=None)
    assert non_standard_dataset.molecule is None

    standard_dataset = data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
    standard_dataset.molecule = None  # if set manually to None, it should be allowed
    assert standard_dataset.molecule is None


def test_lazy_load_until_access_non_full(tmp_path):
    """Test that Datasets do not load values until accessed with non-full files."""
    filename = str(tmp_path / "myset_hf_state.dat")
    data.Dataset._write_file(2, filename)
    dataset = data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
    assert dataset.attrs == {"hf_state": None}
    assert dataset.hf_state == 2
    assert dataset.attrs == {"hf_state": 2}


def test_lazy_load_until_access_full(tmp_path):
    """Test that Datasets do not load values until accessed with full files."""
    filename = str(tmp_path / "myset_full.dat")
    data.Dataset._write_file({"molecule": 1, "hf_state": 2}, filename)
    dataset = data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
    assert dataset.attrs == {"molecule": None, "hf_state": None}
    assert dataset.molecule == 1
    assert dataset.attrs == {"molecule": 1, "hf_state": None}
    assert dataset.hf_state == 2
    assert dataset.attrs == {"molecule": 1, "hf_state": 2}


def test_hamiltonian_is_loaded_properly(tmp_path):
    """Test that __getattribute__ correctly loads hamiltonians from dicts."""
    filename = str(tmp_path / "myset_hamiltonian.dat")
    data.Dataset._write_file(
        {"terms": {"IIII": 0.1, "ZIII": 0.2}, "wire_map": {0: 0, 1: 1, 2: 2, 3: 3}}, filename
    )
    dataset = data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
    ham = dataset.hamiltonian
    assert isinstance(ham, qml.Hamiltonian)
    coeffs, ops = ham.terms()
    assert coeffs == [0.1, 0.2]
    assert qml.equal(qml.Identity(0), ops[0])
    assert qml.equal(qml.PauliZ(0), ops[1])


def test_hamiltonian_is_loaded_properly_with_read(tmp_path):
    """Tests that read() and __getattribute__() agree on how to handle hamiltonians."""
    filename = str(tmp_path / "somefile.dat")
    compressed_ham = {"terms": {"IIII": 0.1, "ZIII": 0.2}, "wire_map": {0: 0, 1: 1, 2: 2, 3: 3}}
    data.Dataset._write_file(compressed_ham, filename)
    dataset = data.Dataset()

    # assigning to hamiltonian allows for special processing
    dataset.read(filename, assign_to="hamiltonian")
    assert isinstance(dataset.hamiltonian, qml.Hamiltonian)
    # lazy-loading doesn't break this convention
    dataset.read(filename, assign_to="tapered_hamiltonian", lazy=True)
    assert isinstance(dataset.tapered_hamiltonian, qml.Hamiltonian)
    # for non-special keys, assign_to will simply save the value
    dataset.read(filename, assign_to="not_hamiltonian")
    assert dataset.not_hamiltonian == compressed_ham
    # ...and lazy-loading doesn't break this convention either
    dataset.read(filename, assign_to="not_tapered_hamiltonian", lazy=True)
    assert dataset.not_tapered_hamiltonian == compressed_ham


def test_repr_standard(tmp_path):
    """Test that __repr__ for standard Datasets look as expected."""
    folder = tmp_path / "qchem" / "H2" / "STO-3G" / "1.02"
    os.makedirs(folder)
    data.Dataset._write_file(
        {"molecule": 1, "hf_state": 2}, str(folder / "H2_STO-3G_1.02_full.dat")
    )

    dataset = data.Dataset("qchem", str(folder), "H2_STO-3G_1.02", "", standard=True)
    assert (
        repr(dataset)
        == "<Dataset = description: qchem/H2/STO-3G/1.02, attributes: ['molecule', 'hf_state']>"
    )

    dataset.vqe_energy = 1.1
    assert (
        repr(dataset)
        == "<Dataset = description: qchem/H2/STO-3G/1.02, attributes: ['molecule', 'hf_state', ...]>"
    )


def test_repr_non_standard():
    """Test that __repr__ for non-standard Datasets look as expected."""
    dataset = data.Dataset(foo=1, bar=2)
    assert repr(dataset) == "<Dataset = attributes: ['foo', 'bar']>"

    dataset.baz = 3
    assert repr(dataset) == "<Dataset = attributes: ['foo', 'bar', ...]>"

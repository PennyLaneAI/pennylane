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
import qml

pytestmark = pytest.mark.data


def test_build_dataset():
    """Test that a dataset builds correctly and returns the correct values."""
    hamiltonian = qml.Hamiltonian(coeffs=[1], observables=[qml.PauliZ(wires=0)])
    test_dataset = qml.data.Dataset(kw1=1, kw2="2", kw3=[3], hamiltonian=hamiltonian)

    assert test_dataset.kw1 == 1
    assert test_dataset.kw2 == "2"
    assert test_dataset.kw3 == [3]
    assert test_dataset.hamiltonian == hamiltonian


def test_write_dataset(tmp_path):
    """Test that datasets are saved correctly."""
    test_dataset = qml.data.Dataset(kw1=1, kw2="2", kw3=[3])
    d = tmp_path / "sub"
    p = d / "test_dataset"
    test_dataset.write(p)


def test_write_standard_loads_before_writing(tmp_path):
    """Test that the write method loads values before writing if they were None (lazy-loaded)."""
    filepath = tmp_path / "myset_full.dat"
    qml.data.Dataset._write_file({"molecule": 1, "hf_state": 2}, str(filepath))
    dataset = qml.data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
    assert dataset.attrs == {"molecule": None, "hf_state": None}

    dest_file = str(tmp_path / "written.dat")
    dataset.write(dest_file)
    # these were None before - calling write() loaded them!
    assert dataset.attrs == {"molecule": 1, "hf_state": 2}

    read_dataset = qml.data.Dataset()
    read_dataset.read(dest_file)
    assert read_dataset.attrs == {"molecule": 1, "hf_state": 2}


def test_read_dataset(tmp_path):
    """Test that datasets are loaded correctly."""
    test_dataset = qml.data.Dataset(kw1=1, kw2="2", kw3=[3])
    d = tmp_path / "sub"
    p = d / "test_dataset"
    test_dataset.write(p)

    test_dataset = qml.data.Dataset()
    test_dataset.read(p)

    assert test_dataset.kw1 == 1
    assert test_dataset.kw2 == "2"
    assert test_dataset.kw3 == [3]


def test_read_does_not_understand_single_attr_files(tmp_path):
    """Test that single-attribute files are not understood by the read() method, even if it follows standard naming conventions."""
    dataset = qml.data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
    filename = str(tmp_path / "myset_molecule.dat")
    dataset._write_file(1, filename)
    assert dataset.list_attributes() == []
    with pytest.raises(AttributeError, match="'int' object has no attribute 'items'"):
        dataset.read(filename)


def test_read_lazy(tmp_path):
    """Test that read with lazy=True works for non-standard datasets."""
    filename = str(tmp_path / "myfile.dat")
    qml.data.Dataset._write_file({"molecule": 1}, filename)
    dataset = qml.data.Dataset()
    dataset.read(filename, lazy=True)
    assert dataset.attrs == {"molecule": None}
    assert dataset._attr_filemap == {"molecule": (filename, True)}
    assert dataset.molecule == 1
    assert dataset._attr_filemap == {}


def test_read_fails_if_lazy_file_was_deleted(tmp_path):
    """Test that read() fails if the file an attribute was loaded from was deleted before loading."""
    filename = str(tmp_path / "myfile.dat")
    qml.data.Dataset._write_file({"molecule": 1}, filename)
    dataset = qml.data.Dataset()
    dataset.read(filename, lazy=True)
    os.remove(filename)
    with pytest.raises(
        qml.data.dataset.DatasetLoadError, match="the file originally containing it"
    ):
        _ = dataset.molecule


def test_read_with_assign_to_without_lazy(tmp_path):
    """Test that read() with assign_to simply sets the file contents."""
    filename = str(tmp_path / "myfile.dat")
    qml.data.Dataset._write_file(1, filename)
    dataset = qml.data.Dataset()
    dataset.read(filename, assign_to="molecule")
    assert dataset.molecule == 1

    dataset.write(filename)
    dataset.read(filename, assign_to="molecule")
    assert dataset.molecule == {"molecule": 1}


def test_read_with_assign_to_with_lazy(tmp_path):
    """Test that read() with assign_to and lazy set simply tracks the file location."""
    filename = str(tmp_path / "myfile.dat")
    qml.data.Dataset._write_file(1, filename)
    dataset = qml.data.Dataset()
    dataset.read(filename, assign_to="molecule", lazy=True)
    assert dataset.attrs == {"molecule": None}
    assert dataset._attr_filemap == {"molecule": (filename, False)}
    assert dataset.molecule == 1


def test_list_attributes():
    """Test the list_attributes method."""
    test_dataset = qml.data.Dataset(kw1=1)
    assert test_dataset.list_attributes() == ["kw1"]


def test_copy_non_standard():
    """Test that datasets can be built from other datasets."""
    test_dataset = qml.data.Dataset(dtype="test_data", kw1=1, kw2="2", kw3=[3])
    new_dataset = copy(test_dataset)
    assert new_dataset.attrs == test_dataset.attrs
    assert new_dataset._is_standard is False


def test_copy_non_standard_with_lazy(tmp_path):
    """Test that non-standard datasets copy lazy-loading metadata."""
    filepath = str(tmp_path / "copyfile.dat")
    test_dataset = qml.data.Dataset(foo=1, bar=2)
    test_dataset.write(filepath)

    base_dataset = qml.data.Dataset()
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
    qml.data.Dataset._write_file({"molecule": 1, "hf_state": 2}, str(filepath))
    test_dataset = qml.data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
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


def test_invalid_init():
    """Test that __init__ fails with invalid arguments."""
    with pytest.raises(
        TypeError,
        match=r"Standard datasets expect 4 arguments: \['data_name', 'data_folder', 'attr_prefix', 'docstring'\]",
    ):
        qml.data.Dataset("first", "second", standard=True)

    with pytest.raises(ValueError, match="Expected data_name to be a str, got int"):
        qml.data.Dataset(1, "some_folder", "some_prefix", "some_docstr", standard=True)


def test_getattribute_dunder_non_full(tmp_path):
    """Test the getattribute override."""
    non_standard_dataset = qml.data.Dataset(foo="bar")
    with pytest.raises(AttributeError):
        _ = non_standard_dataset.baz

    folder = tmp_path / "datasets" / "myset"

    # would not usually be done by users, bypassing qml.data.load
    standard_dataset = qml.data.Dataset("qchem", str(folder), "myset", "", standard=True)

    # no hf_state file exists (yet!)
    with pytest.raises(AttributeError):
        _ = standard_dataset.hf_state
    # create an hf_state file
    os.makedirs(folder)
    filepath = str(folder / "myset_hf_state.dat")
    qml.data.Dataset._write_file(2, filepath)
    # getattribute does not try to find files that have not yet been read
    with pytest.raises(AttributeError):
        _ = standard_dataset.hf_state


def test_getattribute_dunder_full(tmp_path):
    """Test the getattribute behaviour when a fullfile is set."""
    folder = tmp_path / "datasets" / "myset"
    os.makedirs(folder)
    qml.data.Dataset._write_file({"hf_state": 2}, str(folder / "myset_full.dat"))

    # this getattribute will read from the above created file
    dataset = qml.data.Dataset("qchem", str(folder), "myset", "", standard=True)
    assert dataset.hf_state == 2
    with pytest.raises(AttributeError):
        _ = dataset.molecule


def test_getattribute_fail_when_attribute_deleted_from_file(tmp_path):
    """Test that getattribute fails if an expected attribute is not in a data file."""
    filename = str(tmp_path / "dest.dat")
    dataset = qml.data.Dataset(foo=1, bar=2)
    dataset.write(filename)

    new_dataset = qml.data.Dataset()
    new_dataset.read(filename, lazy=True)
    assert new_dataset.attrs == {"foo": None, "bar": None}

    del dataset.foo
    dataset.write(filename)  # overwrite the shared data file, deleting foo
    assert new_dataset.bar == 2
    with pytest.raises(
        qml.data.dataset.DatasetLoadError, match="no longer appears to be in the file"
    ):
        _ = new_dataset.foo


def test_none_attribute_value(tmp_path):
    """Test that both non-standard and standard datasets allow None values."""
    non_standard_dataset = qml.data.Dataset(molecule=None)
    assert non_standard_dataset.molecule is None

    standard_dataset = qml.data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
    standard_dataset.molecule = None  # if set manually to None, it should be allowed
    assert standard_dataset.molecule is None


def test_lazy_load_until_access_non_full(tmp_path):
    """Test that Datasets do not load values until accessed with non-full files."""
    filename = str(tmp_path / "myset_hf_state.dat")
    qml.data.Dataset._write_file(2, filename)
    dataset = qml.data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
    assert dataset.attrs == {"hf_state": None}
    assert dataset.hf_state == 2
    assert dataset.attrs == {"hf_state": 2}


def test_lazy_load_until_access_full(tmp_path):
    """Test that Datasets do not load values until accessed with full files."""
    filename = str(tmp_path / "myset_full.dat")
    qml.data.Dataset._write_file({"molecule": 1, "hf_state": 2}, filename)
    dataset = qml.data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
    assert dataset.attrs == {"molecule": None, "hf_state": None}
    assert dataset.molecule == 1
    assert dataset.attrs == {"molecule": 1, "hf_state": None}
    assert dataset.hf_state == 2
    assert dataset.attrs == {"molecule": 1, "hf_state": 2}


def test_hamiltonian_is_loaded_properly(tmp_path):
    """Test that __getattribute__ correctly loads hamiltonians from dicts."""
    filename = str(tmp_path / "myset_hamiltonian.dat")
    qml.data.Dataset._write_file(
        {"terms": {"IIII": 0.1, "ZIII": 0.2}, "wire_map": {0: 0, 1: 1, 2: 2, 3: 3}}, filename
    )
    dataset = qml.data.Dataset("qchem", str(tmp_path), "myset", "", standard=True)
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
    qml.data.Dataset._write_file(compressed_ham, filename)
    dataset = qml.data.Dataset()

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


def test_hamiltonian_write_preserves_wire_map(tmp_path):
    """Test that writing hamiltonians to file converts to the condensed format."""
    filename = str(tmp_path / "myset_full.dat")
    dataset = qml.data.Dataset()
    obs = [qml.PauliX("a") @ qml.PauliY("c"), qml.PauliZ("a") @ qml.PauliZ("b") @ qml.PauliZ("c")]
    dataset.hamiltonian = qml.Hamiltonian([0.1, 0.2], obs)
    dataset.write(filename)

    # ensure that the non-standard dataset wrote the Hamiltonian in condensed format
    dataset.read(filename, assign_to="terms_and_wiremap")
    assert dataset.terms_and_wiremap == {
        "hamiltonian": {"terms": {"XIY": 0.1, "ZZZ": 0.2}, "wire_map": {"a": 0, "b": 1, "c": 2}}
    }

    # ensure std dataset reads what was written as expected (conversion happens in getattr dunder)
    std_ham = qml.data.Dataset("qchem", str(tmp_path), "myset", "", standard=True).hamiltonian
    assert qml.equal(std_ham, dataset.hamiltonian)
    assert std_ham.wires.tolist() == ["a", "b", "c"]

    # ensure non-std dataset read works (conversion happens in read() instance method)
    non_std_dataset = qml.data.Dataset()
    non_std_dataset.read(filename)
    assert qml.equal(non_std_dataset.hamiltonian, dataset.hamiltonian)


def test_import_zstd_dill(monkeypatch):
    """Test if an ImportError is raised by _import_zstd_dill function."""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "zstd", None)

        with pytest.raises(ImportError, match="This feature requires zstd and dill"):
            qml.data.dataset._import_zstd_dill()

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "dill", None)

        with pytest.raises(ImportError, match="This feature requires zstd and dill"):
            qml.data.dataset._import_zstd_dill()


def test_repr_standard(tmp_path):
    """Test that __repr__ for standard Datasets look as expected."""
    folder = tmp_path / "qchem" / "H2" / "STO-3G" / "1.02"
    os.makedirs(folder)
    qml.data.Dataset._write_file(
        {"molecule": 1, "hf_state": 2}, str(folder / "H2_STO-3G_1.02_full.dat")
    )

    dataset = qml.data.Dataset("qchem", str(folder), "H2_STO-3G_1.02", "", standard=True)
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
    dataset = qml.data.Dataset(foo=1, bar=2)
    assert repr(dataset) == "<Dataset = attributes: ['foo', 'bar']>"

    dataset.baz = 3
    assert repr(dataset) == "<Dataset = attributes: ['foo', 'bar', ...]>"

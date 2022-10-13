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
Unit tests for the :mod:`pennylane.data.Dataset` class and its functions.
"""
import pytest
import pennylane as qml
import dill
import zstd
from unittest.mock import patch, mock_open

def test_build_dataset():
    """Test that a dataset builds correctly and returns the correct values."""
    hamiltonian = qml.Hamiltonian(coeffs=[1], observables=[qml.PauliZ(wires=0)])
    test_dataset = qml.data.Dataset(kw1=1, kw2="2", kw3=[3], hamiltonian=hamiltonian)

    assert test_dataset.kw1 == 1
    assert test_dataset.kw2 == "2"
    assert test_dataset.kw3 == [3]
    assert test_dataset.hamiltonian == hamiltonian


def test_write_dataset():
    """Test that datasets are saved correctly."""
    open_mock = mock_open()
    test_dataset = qml.data.Dataset(kw1=1, kw2="2", kw3=[3])

    with patch("builtins.open", open_mock) as mock_file:
        test_dataset.write("./path/to/file.dat")

    test_dict = {"dtype": None, "__doc__": "", "kw1": 1, "kw2": "2", "kw3": [3]}
    pickled_data = dill.dumps(test_dict, protocol=4)  # returns data as a bytes object
    compressed_pickle = zstd.compress(pickled_data)

    open_mock.assert_called_with("./path/to/file.dat", "wb")  # check written to correct file
    open_mock.return_value.write.assert_called_with(compressed_pickle)  # check correct data written


def test_read_dataset():
    """Test that datasets are loaded correctly."""
    # before conversion, read data is a compressed dictionary
    # generate correct read data
    test_dict = {'kw1':1, 'kw2':"2", 'kw3':[3]}
    pickled_data = dill.dumps(test_dict, protocol=4)
    compressed_pickle = zstd.compress(pickled_data)
    
    #generate the expected dataset
    read_dataset = qml.data.Dataset(kw1=1, kw2="2", kw3=[3])
    
    open_mock = mock_open(read_data=compressed_pickle)
    test_dataset = qml.data.Dataset()

    with patch("builtins.open", open_mock):
        test_dataset.read("./path/to/file.dat")

    open_mock.return_value.read.assert_called()
    open_mock.assert_called_with("./path/to/file.dat", "rb")

    assert test_dataset == read_dataset


def test_from_dataset():
    """Test that datasets can be built from other datasets"""
    test_dataset = qml.data.Dataset(dtype="test_data", kw1=1, kw2="2", kw3=[3])
    new_dataset = qml.data.Dataset.from_dataset(test_dataset)

    assert new_dataset == test_dataset

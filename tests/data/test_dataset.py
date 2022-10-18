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


def test_write_dataset(tmp_path):
    """Test that datasets are saved correctly."""
    test_dataset = qml.data.Dataset(kw1=1, kw2="2", kw3=[3])
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_dataset"
    test_dataset.write(p)


def test_read_dataset(tmp_path):
    """Test that datasets are loaded correctly."""
    test_dataset = qml.data.Dataset(kw1=1, kw2="2", kw3=[3])
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_dataset"
    test_dataset.write(p)

    test_dataset = qml.data.Dataset()
    test_dataset.read(p)

    assert test_dataset.kw1 == 1
    assert test_dataset.kw2 == "2"
    assert test_dataset.kw3 == [3]


def test_from_dataset():
    """Test that datasets can be built from other datasets"""
    test_dataset = qml.data.Dataset(dtype="test_data", kw1=1, kw2="2", kw3=[3])
    new_dataset = qml.data.Dataset.from_dataset(test_dataset)

    assert new_dataset == test_dataset

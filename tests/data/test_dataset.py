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
Unit tests for the :mod:`pennylane.data.Dataset` class.
"""
import pytest
import numpy as np
import pennylane as qml
import dill
import zstd
from unittest.mock import patch, mock_open

from pennylane.ops.qubit import hamiltonian


def test_build_dataset():
    """Test that a dataset builds correctly and returns the correct values."""
    hamiltonian = qml.Hamiltonian(coeffs=[1], observables=[qml.PauliZ(wires=0)])
    test_dataset = qml.data.Dataset(kw1=1, kw2="2", kw3=[3], hamiltonian=hamiltonian)

    assert test_dataset.kw1 == 1
    assert test_dataset.kw2 == "2"
    assert test_dataset.kw3 == [3]
    assert test_dataset.hamiltonian == hamiltonian


def test_write_dataset(tmp_path):
    """Test that a dataset writes to memory without errors."""
    test_dataset = qml.data.Dataset(kw1=1, kw2="2", kw3=[3])
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_dataset"
    test_dataset.write(p)


def test_read_dataset(tmp_path):
    """Test that a dataset can be read from memory and that the data was written correctly."""
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

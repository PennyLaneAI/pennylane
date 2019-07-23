# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Pytest configuration file for PennyLane test suite.
"""
import pytest
import os
import numpy as np
import pennylane as qml
from pennylane.plugins import DefaultGaussian

# defaults
TOL = 1e-3


class DummyDevice(DefaultGaussian):
    """Dummy device to allow Kerr operations"""
    _operation_map = DefaultGaussian._operation_map.copy()
    _operation_map['Kerr'] = lambda *x, **y: np.identity(2)


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope="session", params=[1, 2])
def n_layers(request):
    """Number of layers."""
    return request.param


@pytest.fixture(scope="session", params=[2, 3])
def n_subsystems(request):
    """Number of qubits or qumodes."""
    return request.param


@pytest.fixture(scope="session")
def qubit_device(n_subsystems):
    """Number of qubits or modes."""
    return qml.device('default.qubit', wires=n_subsystems)

@pytest.fixture(scope="function")
def qubit_device_1_wire():
    return qml.device('default.qubit', wires=1)


@pytest.fixture(scope="function")
def qubit_device_2_wires():
    return qml.device('default.qubit', wires=2)


@pytest.fixture(scope="function")
def qubit_device_3_wires():
    return qml.device('default.qubit', wires=3)


@pytest.fixture(scope="session")
def gaussian_device(n_subsystems):
    """Number of qubits or modes."""
    return DummyDevice(wires=n_subsystems)


@pytest.fixture(scope="session")
def gaussian_device_4modes():
    """A 4 mode Gaussian device."""
    return DummyDevice(wires=4)


@pytest.fixture(scope='session')
def torch_support():
    """Boolean fixture for PyTorch support"""
    try:
        import torch
        from torch.autograd import Variable
        torch_support = True
    except ImportError as e:
        torch_support = False

    return torch_support


@pytest.fixture(scope='session')
def tf_support():
    """Boolean fixture for TensorFlow support"""
    try:
        import tensorflow as tf
        import tensorflow.contrib.eager as tfe
        tf.enable_eager_execution()
        tf_support = True
    except ImportError as e:
        tf_support = False

    return tf_support


@pytest.fixture(scope="module",
                params=[1, 2, 3])
def seed(request):
    """Different seeds."""
    return request.param


# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
import os

import pytest
import numpy as np

import pennylane as qml
from pennylane.devices import DefaultGaussian


# defaults
TOL = 1e-3
TF_TOL = 2e-2


class DummyDevice(DefaultGaussian):
    """Dummy device to allow Kerr operations"""
    _operation_map = DefaultGaussian._operation_map.copy()
    _operation_map['Kerr'] = lambda *x, **y: np.identity(2)


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope="session")
def tf_tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TF_TOL", TF_TOL))


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
def gaussian_dummy():
    """Gaussian device with dummy Kerr gate."""
    return DummyDevice


@pytest.fixture(scope="session")
def gaussian_device_2_wires():
    """A 2-mode Gaussian device."""
    return DummyDevice(wires=2)


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


@pytest.fixture()
def skip_if_no_torch_support(torch_support):
    if not torch_support:
        pytest.skip("Skipped, no torch support")


@pytest.fixture(scope='module')
def tf_support():
    """Boolean fixture for TensorFlow support"""
    try:
        import tensorflow as tf
        tf_support = True

    except ImportError as e:
        tf_support = False

    return tf_support


@pytest.fixture()
def skip_if_no_tf_support(tf_support):
    if not tf_support:
        pytest.skip("Skipped, no tf support")


@pytest.fixture(scope="module",
                params=[1, 2, 3])
def seed(request):
    """Different seeds."""
    return request.param


@pytest.fixture(scope="function")
def mock_device(monkeypatch):
    """A mock instance of the abstract Device class"""

    with monkeypatch.context() as m:
        dev = qml.Device
        m.setattr(dev, '__abstractmethods__', frozenset())
        m.setattr(dev, 'short_name', 'mock_device')
        m.setattr(dev, 'capabilities', lambda cls: {"model": "qubit"})
        m.setattr(dev, "operations", {"RX", "RY", "RZ", "CNOT", "SWAP"})
        yield qml.Device(wires=2)


@pytest.fixture
def tear_down_hermitian():
    yield None
    qml.Hermitian._eigs = {}


@pytest.fixture
def non_tape_mode_only():
    """Run the test in tape mode"""
    qml.disable_tape()
    yield
    qml.enable_tape()


@pytest.fixture
def in_tape_mode():
    return


@pytest.fixture(params=[False, True])
def tape_mode(request, mocker):
    """Tests using this fixture will be run twice, once in tape mode and once without."""

    if request.param:
        # Several attributes and methods on the old QNode have a new location on the new QNode/tape.
        # Here, we dynamically mock so that the tests do not have to be modified to support both
        # tape and non-tape mode. Once tape mode is default, we can make the equivalent
        # changes directly in the tests.
        mocker.patch("pennylane.tape.QNode.ops", property(lambda self: self.qtape.operations + self.qtape.observables), create=True)
        mocker.patch("pennylane.tape.QNode.h", property(lambda self: self.diff_options["h"]), create=True)
        mocker.patch("pennylane.tape.QNode.order", property(lambda self: self.diff_options["order"]), create=True)
        mocker.patch("pennylane.tape.QNode.circuit", property(lambda self: self.qtape.graph), create=True)

        def patched_jacobian(self, args, **kwargs):
            method = kwargs.get("method", "best")

            if method == "A":
                method = "analytic"
            elif method == "F":
                method = "numeric"

            kwargs["method"] = method
            dev = kwargs["options"]["device"]

            return self.qtape.jacobian(dev, **kwargs)


        mocker.patch("pennylane.tape.QNode.jacobian", patched_jacobian, create=True)

    else:
        qml.disable_tape()

    yield

    qml.enable_tape()

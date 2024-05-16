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
"""
Unit tests for the DefaultTensor class.
"""


import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.default_tensor import DefaultTensor
from pennylane.qnode import QNode


def test_name():
    """Test the name of DefaultTensor."""
    assert DefaultTensor().name == "default.tensor"


def test_wires():
    """Test that a device can be created with wires."""
    assert DefaultTensor().wires is None
    assert DefaultTensor(wires=2).wires == qml.wires.Wires([0, 1])
    assert DefaultTensor(wires=[0, 2]).wires == qml.wires.Wires([0, 2])

    with pytest.raises(AttributeError):
        DefaultTensor().wires = [0, 1]


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_data_type(dtype):
    """Test the data type."""
    assert DefaultTensor(dtype=dtype).dtype == dtype


def test_ivalid_data_type():
    """Test that data type can only be np.complex64 or np.complex128."""
    with pytest.raises(TypeError):
        DefaultTensor(dtype=float)


def test_invalid_method():
    """Test an invalid method."""
    method = "invalid_method"
    with pytest.raises(ValueError, match=f"Unsupported method: {method}"):
        DefaultTensor(method=method)


def test_invalid_keyword_arg():
    """Test an invalid keyword argument."""
    with pytest.raises(
        TypeError,
        match=f"Unexpected argument: fake_arg during initialization of the default.tensor device.",
    ):
        DefaultTensor(fake_arg=None)


def test_invalid_shots():
    """Test that an error is raised if finite number of shots are requestd."""
    with pytest.raises(ValueError, match="default.tensor does not support finite shots."):
        DefaultTensor(shots=5)

    with pytest.raises(AttributeError):
        DefaultTensor().shots = 10


def test_support_derivatives():
    """Test that the device does not support derivatives yet."""
    dev = DefaultTensor()
    assert not dev.supports_derivatives()


def test_compute_derivatives():
    """Test that an error is raised if the `compute_derivatives` method is called."""
    dev = DefaultTensor()
    with pytest.raises(
        NotImplementedError,
        match="The computation of derivatives has yet to be implemented for the default.tensor device.",
    ):
        dev.compute_derivatives(circuits=None)


def test_execute_and_compute_derivatives():
    """Test that an error is raised if `execute_and_compute_derivative` method is called."""
    dev = DefaultTensor()
    with pytest.raises(
        NotImplementedError,
        match="The computation of derivatives has yet to be implemented for the default.tensor device.",
    ):
        dev.execute_and_compute_derivatives(circuits=None)


def test_supports_vjp():
    """Test that the device does not support VJP yet."""
    dev = DefaultTensor()
    assert not dev.supports_vjp()


def test_compute_vjp():
    """Test that an error is raised if `compute_vjp` method is called."""
    dev = DefaultTensor()
    with pytest.raises(
        NotImplementedError,
        match="The computation of vector-Jacobian product has yet to be implemented for the default.tensor device.",
    ):
        dev.compute_vjp(circuits=None, cotangents=None)


def test_execute_and_compute_vjp():
    """Test that an error is raised if `execute_and_compute_vjp` method is called."""
    dev = DefaultTensor()
    with pytest.raises(
        NotImplementedError,
        match="The computation of vector-Jacobian product has yet to be implemented for the default.tensor device.",
    ):
        dev.execute_and_compute_vjp(circuits=None, cotangents=None)


def test_interface_jax(self, backend, method):
    """Test the interface with JAX."""

    jax = pytest.importorskip("jax")
    dev = DefaultTensor(wires=qml.wires.Wires(range(1)), backend=backend, method=method)
    ref_dev = qml.device("default.qubit.jax", wires=1)

    def circuit(x):
        qml.RX(x[1], wires=0)
        qml.Rot(x[0], x[1], x[2], wires=0)
        return qml.expval(qml.Z(0))

    weights = jax.numpy.array([0.2, 0.5, 0.1])
    qnode = QNode(circuit, dev, interface="jax")
    ref_qnode = QNode(circuit, ref_dev, interface="jax")

    assert np.allclose(qnode(weights), ref_qnode(weights))


def test_interface_jax_jit(self, backend, method):
    """Test the interface with JAX's JIT compiler."""

    jax = pytest.importorskip("jax")
    dev = DefaultTensor(wires=qml.wires.Wires(range(1)), backend=backend, method=method)

    @jax.jit
    @qml.qnode(dev, interface="jax")
    def circuit():
        qml.Hadamard(0)
        return qml.expval(qml.Z(0))

    assert np.allclose(circuit(), 0.0)


def test_(self, backend, method):
    """..."""

    # jax = pytest.importorskip("jax")
    dev = DefaultTensor(wires=qml.wires.Wires(range(1)), backend=backend, method=method)

    def circuit():
        qml.RX(0.0, wires=0)

    with pytest.raises(qml.QuantumFunctionError):
        QNode(circuit, dev, interface="jax", diff_method="adjoint")

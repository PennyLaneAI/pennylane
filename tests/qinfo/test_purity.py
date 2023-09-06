# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for purities."""
# pylint: disable=too-many-arguments
import pytest

import pennylane as qml
from pennylane import numpy as np


def expected_purity_ising_xx(param):
    """Returns the analytical purity for subsystems of the IsingXX"""
    return np.cos(param / 2) ** 4 + np.sin(param / 2) ** 4


def expected_purity_grad_ising_xx(param):
    """The analytic gradient purity for the IsingXX"""

    eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    grad_expected_purity = (
        2
        * eig_1
        * (np.sin(param / 2) ** 3 * np.cos(param / 2) - np.sin(param / 2) * np.cos(param / 2) ** 3)
        / np.sqrt(1 - 4 * np.sin(param / 2) ** 2 * np.cos(param / 2) ** 2)
    ) + (
        2
        * eig_2
        * (
            np.sin(param / 2)
            * np.cos(param / 2)
            * (np.cos(param / 2) ** 2 - np.sin(param / 2) ** 2)
        )
        / np.sqrt(1 - 4 * np.sin(param / 2) ** 2 * np.cos(param / 2) ** 2)
    )
    return grad_expected_purity


class TestPurity:
    """Tests for purity transform"""

    devices = ["default.qubit", "lightning.qubit", "default.mixed"]
    grad_supported_devices = ["default.qubit", "default.mixed"]
    mix_supported_devices = ["default.mixed"]
    diff_methods = ["backprop"]

    parameters = np.linspace(0, 2 * np.pi, 10)
    probs = np.array([0.001, 0.01, 0.1, 0.2])

    wires_list = [([0], True), ([1], True), ([0, 1], False)]

    def test_qnode_not_returning_state(self):
        """Test that the QNode of reduced_dm function must return state."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.RZ(0, wires=[0])
            return qml.expval(qml.PauliX(wires=0))

        with pytest.raises(ValueError, match="The qfunc return type needs to be a state"):
            qml.qinfo.purity(circuit, wires=[0])()

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    def test_IsingXX_qnode_purity(self, device, param, wires, is_partial):
        """Tests purity for a qnode"""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        purity = qml.qinfo.purity(circuit_state, wires=wires)(param)
        expected_purity = expected_purity_ising_xx(param) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    def test_IsingXX_qnode_purity_no_params(self, device, wires, is_partial):
        """Tests purity for a qnode without parameters"""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit_state():
            qml.IsingXX(0, wires=[0, 1])
            return qml.state()

        purity = qml.qinfo.purity(circuit_state, wires=wires)()
        expected_purity = expected_purity_ising_xx(0) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.parametrize("device", mix_supported_devices)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("param", probs)
    def test_bit_flip_qnode_purity(self, device, wires, param, is_partial):
        """Tests purity for a qnode on a noisy device with bit flips"""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit_state(p):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.BitFlip(p, wires=0)
            qml.BitFlip(p, wires=1)
            return qml.state()

        purity = qml.qinfo.purity(circuit_state, wires=wires)(param)
        expected_purity = (
            0.5
            if is_partial
            else 4 * (0.5 - (1 - param) * param) ** 2 + 4 * (1 - param) ** 2 * param ** 2
        )
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.autograd
    @pytest.mark.parametrize("device", grad_supported_devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_IsingXX_qnode_purity_grad(self, device, param, wires, is_partial, diff_method):
        """Tests gradient of the purity for a qnode"""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_purity = qml.grad(qml.qinfo.purity(circuit_state, wires=wires))(param)
        expected_grad = expected_purity_grad_ising_xx(param) if is_partial else 0
        assert qml.math.allclose(grad_purity, expected_grad)

    @pytest.mark.autograd
    @pytest.mark.parametrize("device", mix_supported_devices)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("param", probs)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_bit_flip_qnode_purity_grad(self, device, wires, param, is_partial, diff_method):
        """Tests gradient of purity for a qnode on a noisy device with bit flips"""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit_state(p):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.BitFlip(p, wires=0)
            qml.BitFlip(p, wires=1)
            return qml.state()

        purity_grad = qml.grad(qml.qinfo.purity(circuit_state, wires=wires))(param)
        expected_purity_grad = 0 if is_partial else 32 * (param - 0.5) ** 3
        assert qml.math.allclose(purity_grad, expected_purity_grad)

    interfaces = ["jax"]

    @pytest.mark.jax
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("interface", interfaces)
    def test_IsingXX_qnode_purity_jax(self, device, param, wires, is_partial, interface):
        """Test purity for a QNode with jax interface."""

        import jax.numpy as jnp

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        purity = qml.qinfo.purity(circuit_state, wires=wires)(jnp.array(param))
        expected_purity = expected_purity_ising_xx(param) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.jax
    @pytest.mark.parametrize("device", grad_supported_devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", interfaces)
    def test_IsingXX_qnode_purity_grad_jax(
        self, device, param, wires, is_partial, diff_method, interface
    ):
        """Test purity for a QNode gradient with Jax."""

        import jax

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_purity = jax.grad(qml.qinfo.purity(circuit_state, wires=wires))(jax.numpy.array(param))
        grad_expected_purity = expected_purity_grad_ising_xx(param) if is_partial else 0

        assert qml.math.allclose(grad_purity, grad_expected_purity, rtol=1e-04, atol=1e-05)

    @pytest.mark.jax
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("interface", interfaces)
    def test_IsingXX_qnode_purity_jax_jit(self, device, param, wires, is_partial, interface):
        """Test purity for a QNode with jax-jit interface."""

        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        purity = jax.jit(qml.qinfo.purity(circuit_state, wires=wires))(jnp.array(param))
        expected_purity = expected_purity_ising_xx(param) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.jax
    @pytest.mark.parametrize("device", grad_supported_devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", interfaces)
    def test_IsingXX_qnode_purity_grad_jax_jit(
        self, device, param, wires, is_partial, diff_method, interface
    ):
        """Test purity for a QNode gradient with jax-jit."""

        import jax

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_purity = jax.jit(jax.grad(qml.qinfo.purity(circuit_state, wires=wires)))(
            jax.numpy.array(param)
        )
        grad_expected_purity = expected_purity_grad_ising_xx(param) if is_partial else 0

        assert qml.math.allclose(grad_purity, grad_expected_purity, rtol=1e-04, atol=1e-05)

    interfaces = ["auto", "torch"]

    @pytest.mark.torch
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("interface", interfaces)
    def test_IsingXX_qnode_purity_torch(self, device, param, wires, is_partial, interface):
        """Tests purity for a qnode for the torch interface"""

        import torch

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        purity = qml.qinfo.purity(circuit_state, wires=wires)(torch.tensor(param))
        expected_purity = expected_purity_ising_xx(param) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.torch
    @pytest.mark.parametrize("device", grad_supported_devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", interfaces)
    def test_IsingXX_qnode_purity_grad_torch(
        self, device, param, wires, is_partial, diff_method, interface
    ):
        """Test purity for a QNode gradient with torch."""

        import torch

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        expected_grad = expected_purity_grad_ising_xx(param) if is_partial else 0

        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        purity = qml.qinfo.purity(circuit_state, wires=wires)(param)
        purity.backward()
        grad_purity = param.grad

        assert qml.math.allclose(grad_purity, expected_grad)

    interfaces = ["tf"]

    @pytest.mark.tf
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("interface", interfaces)
    def test_IsingXX_qnode_purity_tf(self, device, param, wires, is_partial, interface):
        """Tests purity for a qnode for the tf interface."""

        import tensorflow as tf

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        purity = qml.qinfo.purity(circuit_state, wires=wires)(tf.Variable(param))
        expected_purity = expected_purity_ising_xx(param) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.tf
    @pytest.mark.parametrize("device", grad_supported_devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", interfaces)
    def test_IsingXX_qnode_purity_grad_tf(
        self, device, param, wires, is_partial, diff_method, interface
    ):
        """Test purity for a QNode gradient with tf."""

        import tensorflow as tf

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_expected_purity = expected_purity_grad_ising_xx(param) if is_partial else 0

        param = tf.Variable(param)
        with tf.GradientTape() as tape:
            purity = qml.qinfo.purity(circuit_state, wires=wires)(param)

        grad_purity = tape.gradient(purity, param)

        assert qml.math.allclose(grad_purity, grad_expected_purity)

    @pytest.mark.parametrize("device", devices)
    def test_purity_wire_labels(self, device, tol):
        """Test that purity is correct with custom wire labels"""
        param = np.array(1.234)
        wires = ["a", 8]
        dev = qml.device(device, wires=wires)

        @qml.qnode(dev)
        def circuit_state(x):
            qml.PauliX(wires=wires[0])
            qml.IsingXX(x, wires=wires)
            return qml.state()

        purity0 = qml.qinfo.purity(circuit_state, wires=[wires[0]])(param)
        purity1 = qml.qinfo.purity(circuit_state, wires=[wires[1]])(param)
        expected = expected_purity_ising_xx(param)

        assert qml.math.allclose(purity0, expected, atol=tol)
        assert qml.math.allclose(purity1, expected, atol=tol)


@pytest.mark.parametrize("device", ["default.qubit", "default.mixed"])
def test_broadcasting(device):
    """Test that the purity transform supports broadcasting"""
    dev = qml.device(device, wires=2)

    @qml.qnode(dev)
    def circuit_state(x):
        qml.IsingXX(x, wires=[0, 1])
        return qml.state()

    x = np.array([0.4, 0.6, 0.8])
    purity = qml.qinfo.purity(circuit_state, wires=[0])(x)

    expected = expected_purity_ising_xx(x)
    assert qml.math.allclose(purity, expected)

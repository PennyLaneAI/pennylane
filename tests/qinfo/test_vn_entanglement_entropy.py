# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the vn entanglement entropy transform"""


import numpy as np
import pytest

import pennylane as qml


class TestVnEntanglementEntropy:
    """Tests for the vn entanglement entropy transform"""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("device", ["default.qubit", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 4)[1:])
    def test_qnode_transform(self, device, interface, params):
        """Test that the entanglement entropy transform works for QNodes"""

        dev = qml.device(device, wires=2)
        params = qml.math.asarray(params, like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit(theta):
            qml.RY(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = qml.qinfo.vn_entanglement_entropy(circuit, wires0=[0], wires1=[1])(params)

        # Compare transform results with analytic values
        expected = -np.cos(params / 2) ** 2 * np.log(np.cos(params / 2) ** 2) - np.sin(
            params / 2
        ) ** 2 * np.log(np.sin(params / 2) ** 2)

        assert qml.math.allclose(actual, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 4)[1:])
    def test_qnode_transform_jax_jit(self, params):
        """Test that the entanglement entropy works for QNodes for the JAX-jit interface"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        params = jnp.array(params)

        @qml.qnode(dev, interface="jax-jit")
        def circuit(theta):
            qml.RY(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = jax.jit(qml.qinfo.vn_entanglement_entropy(circuit, wires0=[0], wires1=[1]))(params)

        # Compare transform results with analytic values
        expected = -jnp.cos(params / 2) ** 2 * jnp.log(jnp.cos(params / 2) ** 2) - jnp.sin(
            params / 2
        ) ** 2 * jnp.log(jnp.sin(params / 2) ** 2)

        assert qml.math.allclose(actual, expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 4)[1:])
    def test_qnode_transform_grad(self, params):

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(theta):
            qml.RY(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = qml.grad(qml.qinfo.vn_entanglement_entropy(circuit, wires0=[0], wires1=[1]))(
            params
        )

        # Compare transform results with analytic values
        expected = (
            np.sin(params / 2)
            * np.cos(params / 2)
            * (np.log(np.cos(params / 2) ** 2) - np.log(np.sin(params / 2) ** 2))
        )

        assert qml.math.allclose(actual, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 4)[1:])
    def test_qnode_transform_grad_jax(self, params):
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(theta):
            qml.RY(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = jax.grad(qml.qinfo.vn_entanglement_entropy(circuit, wires0=[0], wires1=[1]))(
            jax.numpy.array(params)
        )

        # Compare transform results with analytic values
        expected = (
            np.sin(params / 2)
            * np.cos(params / 2)
            * (np.log(np.cos(params / 2) ** 2) - np.log(np.sin(params / 2) ** 2))
        )

        assert qml.math.allclose(actual, expected, rtol=1e-04, atol=1e-05)

    @pytest.mark.jax
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 4)[1:])
    def test_qnode_transform_grad_jax_jit(self, params):
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(theta):
            qml.RY(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = jax.jit(
            jax.grad(qml.qinfo.vn_entanglement_entropy(circuit, wires0=[0], wires1=[1]))
        )(jax.numpy.array(params))

        # Compare transform results with analytic values
        expected = (
            np.sin(params / 2)
            * np.cos(params / 2)
            * (np.log(np.cos(params / 2) ** 2) - np.log(np.sin(params / 2) ** 2))
        )

        assert qml.math.allclose(actual, expected, rtol=1e-04, atol=1e-05)

    @pytest.mark.torch
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 4)[1:])
    def test_qnode_transform_grad_torch(self, params):

        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(theta):
            qml.RY(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        # Compare transform results with analytic values
        expected = (
            np.sin(params / 2)
            * np.cos(params / 2)
            * (np.log(np.cos(params / 2) ** 2) - np.log(np.sin(params / 2) ** 2))
        )

        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        entropy = qml.qinfo.vn_entanglement_entropy(circuit, wires0=[0], wires1=[1])(params)
        entropy.backward()
        actual = params.grad

        assert qml.math.allclose(actual, expected)

    @pytest.mark.tf
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 4)[1:])
    def test_qnode_transform_grad_tf(self, params):
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(theta):
            qml.RY(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        # Compare transform results with analytic values
        expected = (
            np.sin(params / 2)
            * np.cos(params / 2)
            * (np.log(np.cos(params / 2) ** 2) - np.log(np.sin(params / 2) ** 2))
        )

        params = tf.Variable(params)
        with tf.GradientTape() as tape:
            entropy = qml.qinfo.vn_entanglement_entropy(circuit, wires0=[0], wires1=[1])(params)
        actual = tape.gradient(entropy, params)

        assert qml.math.allclose(actual, expected)

    def test_qnode_transform_not_state(self):
        """Test the qnode transform needs a QNode returning state."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(theta):
            qml.RY(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0))

        with pytest.raises(
            ValueError,
            match="The qfunc return type needs to be a state.",
        ):
            qml.qinfo.vn_entanglement_entropy(circuit, wires0=[0], wires1=[1])(0.1)

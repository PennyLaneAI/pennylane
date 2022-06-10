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
"""Unit tests for differentiable quantum entropies.
"""

import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")


def expected_entropy_ising_xx(param):
    """
    Return the analytical entropy for the IsingXX.
    """
    eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eigs = [eig_1, eig_2]
    eigs = [eig for eig in eigs if eig > 0]

    expected_entropy = eigs * np.log(eigs)

    expected_entropy = -np.sum(expected_entropy)
    return expected_entropy


def expected_entropy_grad_ising_xx(param):
    """
    Return the analytical gradient entropy for the IsingXX.
    """
    eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eigs = [eig_1, eig_2]
    eigs = np.maximum(eigs, 1e-08)

    grad_expected_entropy = -(
        (np.log(eigs[0]) + 1)
        * (np.sin(param / 2) ** 3 * np.cos(param / 2) - np.sin(param / 2) * np.cos(param / 2) ** 3)
        / np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)
    ) - (
        (np.log(eigs[1]) + 1)
        * (
            np.sin(param / 2)
            * np.cos(param / 2)
            * (np.cos(param / 2) ** 2 - np.sin(param / 2) ** 2)
        )
        / np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)
    )
    return grad_expected_entropy


class TestVonNeumannEntropy:
    """Tests for creating a density matrix from state vectors."""

    single_wires_list = [
        [0],
        [1],
    ]

    base = [2, np.exp(1), 10]

    check_state = [True, False]

    parameters = np.linspace(0, 2 * np.pi, 20)
    devices = ["default.qubit", "default.mixed"]

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_entropy(self, param, wires, device, base):
        """Test entropy for a QNode numpy."""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy(circuit_state, wires=wires, base=base)(param)

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)
        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.autograd
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_entropy_grad(self, param, wires, base):
        """Test entropy for a QNode gradient with autograd."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_entropy = qml.grad(qml.qinfo.vn_entropy(circuit_state, wires=wires, base=base))(param)

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)
        assert qml.math.allclose(grad_entropy, grad_expected_entropy)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_torch_entropy(self, param, wires, device, base):
        """Test entropy for a QNode with torch interface."""
        import torch

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy(circuit_state, wires=wires, base=base)(torch.tensor(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)
        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_entropy_grad_torch(self, param, wires, base):
        """Test entropy for a QNode gradient with torch."""
        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = np.maximum(eigs, 1e-08)

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        entropy = qml.qinfo.vn_entropy(circuit_state, wires=wires, base=base)(param)
        entropy.backward()
        grad_entropy = param.grad

        assert qml.math.allclose(grad_entropy, grad_expected_entropy)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_tf_entropy(self, param, wires, device, base):
        """Test entropy for a QNode with tf interface."""
        import tensorflow as tf

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="tf")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy(circuit_state, wires=wires, base=base)(tf.Variable(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_entropy_grad_tf(self, param, wires, base):
        """Test entropy for a QNode gradient with tf."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        param = tf.Variable(param)
        with tf.GradientTape() as tape:
            entropy = qml.qinfo.vn_entropy(circuit_state, wires=wires, base=base)(param)

        grad_entropy = tape.gradient(entropy, param)

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        assert qml.math.allclose(grad_entropy, grad_expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_jax_entropy(self, param, wires, device, base):
        """Test entropy for a QNode with jax interface."""
        import jax.numpy as jnp

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy(circuit_state, wires=wires, base=base)(jnp.array(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_entropy_grad_jax(self, param, wires, base):
        """Test entropy for a QNode gradient with Jax."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_entropy = jax.grad(qml.qinfo.vn_entropy(circuit_state, wires=wires, base=base))(
            jax.numpy.array(param)
        )

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=1e-05)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_jax_jit_entropy(self, param, wires, base):
        """Test entropy for a QNode with jax-jit interface."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax-jit")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = jax.jit(qml.qinfo.vn_entropy(circuit_state, wires=wires, base=base))(
            jnp.array(param)
        )

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_entropy_grad_jax_jit(self, param, wires, base):
        """Test entropy for a QNode gradient with Jax-jit."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax-jit", diff_method="backprop")
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        grad_entropy = jax.jit(
            jax.grad(qml.qinfo.vn_entropy(circuit_state, wires=wires, base=base))
        )(jax.numpy.array(param))

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=1e-05)

    def test_qnode_entropy_wires_full_range_not_state(self):
        """Test entropy needs a QNode returning state."""
        param = 0.1
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0))

        with pytest.raises(
            ValueError,
            match="The qfunc return type needs to be a state.",
        ):
            qml.qinfo.vn_entropy(circuit_state, wires=[0, 1])(param)

    def test_qnode_entropy_wires_full_range_state_vector(self):
        """Test entropy for a QNode that returns a state vector with all wires, entropy is 0."""
        param = 0.1
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy(circuit_state, wires=[0, 1])(param)

        expected_entropy = 0.0
        assert qml.math.allclose(entropy, expected_entropy)

    def test_qnode_entropy_wires_full_range_density_mat(self):
        """Test entropy for a QNode that returns a density mat with all wires, entropy is 0."""
        param = 0.1
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev)
        def circuit_state(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        entropy = qml.qinfo.vn_entropy(circuit_state, wires=[0, 1])(param)
        expected_entropy = 0.0

        assert qml.math.allclose(entropy, expected_entropy)


class TestMutualInformation:
    """Tests for the mutual information functions"""

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 8))
    def test_qnode_state(self, device, interface, params):
        """Test that the mutual information transform works for QNodes by comparing
        against analytic values"""
        dev = qml.device(device, wires=2)

        params = qml.math.asarray(params, like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = qml.qinfo.mutual_info(circuit, wires0=[0], wires1=[1])(params)

        # compare transform results with analytic values
        expected = -2 * np.cos(params / 2) ** 2 * np.log(
            np.cos(params / 2) ** 2 + 1e-10
        ) - 2 * np.sin(params / 2) ** 2 * np.log(np.sin(params / 2) ** 2 + 1e-10)

        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize("params", zip(np.linspace(0, np.pi, 8), np.linspace(0, 2 * np.pi, 8)))
    def test_qnode_mutual_info(self, device, interface, params):
        """Test that the measurement process for mutual information works for QNodes
        by comparing against the mutual information transform"""
        dev = qml.device(device, wires=2)

        params = qml.math.asarray(np.array(params), like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit_mutual_info(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        @qml.qnode(dev, interface=interface)
        def circuit_state(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = circuit_mutual_info(params)

        # compare measurement results with transform results
        expected = qml.qinfo.mutual_info(circuit_state, wires0=[0], wires1=[1])(params)

        assert np.allclose(actual, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 8))
    def test_qnode_state_jax_jit(self, params):
        """Test that the mutual information transform works for QNodes by comparing
        against analytic values, for the JAX-jit interface"""
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        params = jnp.array(params)

        @qml.qnode(dev, interface="jax-jit")
        def circuit(params):
            qml.RY(params, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = jax.jit(qml.qinfo.mutual_info(circuit, wires0=[0], wires1=[1]))(params)

        # compare transform results with analytic values
        expected = -2 * jnp.cos(params / 2) ** 2 * jnp.log(
            jnp.cos(params / 2) ** 2 + 1e-10
        ) - 2 * jnp.sin(params / 2) ** 2 * jnp.log(jnp.sin(params / 2) ** 2 + 1e-10)

        assert np.allclose(actual, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("params", zip(np.linspace(0, np.pi, 8), np.linspace(0, 2 * np.pi, 8)))
    def test_qnode_mutual_info_jax_jit(self, params):
        """Test that the measurement process for mutual information works for QNodes
        by comparing against the mutual information transform, for the JAX-jit interface"""
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        params = qml.math.asarray(np.array(params), like="jax")

        @qml.qnode(dev, interface="jax-jit")
        def circuit_mutual_info(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        @qml.qnode(dev, interface="jax-jit")
        def circuit_state(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = jax.jit(circuit_mutual_info)(params)

        # compare measurement results with transform results
        expected = jax.jit(qml.qinfo.mutual_info(circuit_state, wires0=[0], wires1=[1]))(params)

        assert np.allclose(actual, expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    def test_qnode_grad(self, param):
        """Test that the gradient of mutual information works for QNodes
        with the autograd interface"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="autograd")
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        if param == 0:
            # we don't allow gradients to flow through the discontinuity at 0
            expected = 0
        else:
            expected = np.sin(param) * (
                np.log(np.cos(param / 2) ** 2) - np.log(np.sin(param / 2) ** 2)
            )

        actual = qml.grad(circuit)(param)
        assert np.allclose(actual, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    def test_qnode_grad_jax(self, param):
        """Test that the gradient of mutual information works for QNodes
        with the JAX interface"""
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        param = jnp.array(param)

        @qml.qnode(dev, interface="jax")
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        if param == 0:
            # we don't allow gradients to flow through the discontinuity at 0
            expected = 0
        else:
            expected = jnp.sin(param) * (
                jnp.log(jnp.cos(param / 2) ** 2) - jnp.log(jnp.sin(param / 2) ** 2)
            )

        actual = jax.grad(circuit)(param)
        assert np.allclose(actual, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    def test_qnode_grad_jax_jit(self, param):
        """Test that the gradient of mutual information works for QNodes
        with the JAX-jit interface"""
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        param = jnp.array(param)

        @qml.qnode(dev, interface="jax-jit")
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        if param == 0:
            # we don't allow gradients to flow through the discontinuity at 0
            expected = 0
        else:
            expected = jnp.sin(param) * (
                jnp.log(jnp.cos(param / 2) ** 2) - jnp.log(jnp.sin(param / 2) ** 2)
            )

        actual = jax.jit(jax.grad(circuit))(param)
        assert np.allclose(actual, expected)

    @pytest.mark.tf
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    def test_qnode_grad_tf(self, param):
        """Test that the gradient of mutual information works for QNodes
        with the tensorflow interface"""
        dev = qml.device("default.qubit", wires=2)

        param = tf.Variable(param)

        @qml.qnode(dev, interface="tensorflow")
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        if param == 0:
            # we don't allow gradients to flow through the discontinuity at 0
            expected = 0
        else:
            expected = np.sin(param) * (
                np.log(np.cos(param / 2) ** 2) - np.log(np.sin(param / 2) ** 2)
            )

        with tf.GradientTape() as tape:
            out = circuit(param)

        actual = tape.gradient(out, param)
        assert np.allclose(actual, expected)

    @pytest.mark.torch
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    def test_qnode_grad_torch(self, param):
        """Test that the gradient of mutual information works for QNodes
        with the torch interface"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        if param == 0:
            # we don't allow gradients to flow through the discontinuity at 0
            expected = 0
        else:
            expected = np.sin(param) * (
                np.log(np.cos(param / 2) ** 2) - np.log(np.sin(param / 2) ** 2)
            )

        param = torch.tensor(param, requires_grad=True)
        out = circuit(param)
        out.backward()

        actual = param.grad
        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "params", [np.array([0.0, 0.0]), np.array([0.3, 0.4]), np.array([0.6, 0.8])]
    )
    def test_subsystem_overlap_error(self, device, interface, params):
        """Test that an error is raised when the subsystems overlap"""
        dev = qml.device(device, wires=3)

        params = qml.math.asarray(params, like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            return qml.mutual_info(wires0=[0, 1], wires1=[1, 2])

        msg = "Subsystems for computing mutual information must not overlap"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            circuit(params)

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "params", [np.array([0.0, 0.0]), np.array([0.3, 0.4]), np.array([0.6, 0.8])]
    )
    def test_custom_wire_labels_error(self, device, interface, params):
        """Tests that an error is raised when mutual information is measured
        with custom wire labels"""
        dev = qml.device(device, wires=["a", "b"])

        params = qml.math.asarray(params, like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params[0], wires="a")
            qml.RY(params[1], wires="b")
            qml.CNOT(wires=["a", "b"])
            return qml.mutual_info(wires0=["a"], wires1=["b"])

        msg = "Returning the mutual information is not supported when using custom wire labels"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            circuit(params)

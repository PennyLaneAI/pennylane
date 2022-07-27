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
    """Tests Von Neumann entropy transform"""

    single_wires_list = [
        [0],
        [1],
    ]

    base = [2, np.exp(1), 10]

    check_state = [True, False]

    parameters = np.linspace(0, 2 * np.pi, 10)
    devices = ["default.qubit", "default.mixed", "lightning.qubit"]

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


class TestVonNeumannEntropyQNode:

    parameters = np.linspace(0, 2 * np.pi, 10)

    devices = ["default.qubit", "default.mixed", "lightning.qubit"]

    single_wires_list = [
        [0],
        [1],
    ]

    base = [2, np.exp(1), 10]

    check_state = [True, False]

    devices = ["default.qubit", "default.mixed"]
    diff_methods = ["backprop", "finite-diff"]

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_entropy(self, param, wires, device, base):
        """Test entropy for a QNode numpy."""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        entropy = circuit_entropy(param)

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)
        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.autograd
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_IsingXX_qnode_entropy_grad(self, param, wires, base, diff_method):
        """Test entropy for a QNode gradient with autograd."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        grad_entropy = qml.grad(circuit_entropy)(param)

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)
        assert qml.math.allclose(grad_entropy, grad_expected_entropy, atol=tol)

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
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        entropy = circuit_entropy(torch.tensor(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_IsingXX_qnode_entropy_grad_torch(self, param, wires, base, diff_method):
        """Test entropy for a QNode gradient with torch."""
        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        entropy = circuit_entropy(param)
        entropy.backward()
        grad_entropy = param.grad

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, atol=tol)

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
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        entropy = circuit_entropy(tf.Variable(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_IsingXX_qnode_entropy_grad_tf(self, param, wires, base, diff_method):
        """Test entropy for a QNode gradient with tf."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        param = tf.Variable(param)
        with tf.GradientTape() as tape:
            entropy = circuit_entropy(param)

        grad_entropy = tape.gradient(entropy, param)

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, atol=tol)

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
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        entropy = circuit_entropy(jnp.array(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_IsingXX_qnode_entropy_grad_jax(self, param, wires, base, diff_method):
        """Test entropy for a QNode gradient with Jax."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        grad_entropy = jax.grad(circuit_entropy)(jax.numpy.array(param))
        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=1e-05)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("device", devices)
    def test_IsingXX_qnode_jax_jit_entropy(self, param, wires, base, device):
        """Test entropy for a QNode with jax-jit interface."""
        import jax
        import jax.numpy as jnp

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface="jax-jit")
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        entropy = jax.jit(circuit_entropy)(jnp.array(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_IsingXX_qnode_entropy_grad_jax_jit(self, param, wires, base, diff_method):
        """Test entropy for a QNode gradient with Jax-jit."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax-jit", diff_method=diff_method)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        grad_entropy = jax.jit(jax.grad(circuit_entropy))(jax.numpy.array(param))

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=1e-05)

    @pytest.mark.parametrize("device", devices)
    def test_qnode_entropy_no_custom_wires(self, device):
        """Test that entropy cannot be returned with custom wires."""

        dev = qml.device(device, wires=["a", 1])

        @qml.qnode(dev)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=["a", 1])
            return qml.vn_entropy(wires=["a"])

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Returning the Von Neumann entropy is not supported when using custom wire labels",
        ):
            circuit_entropy(0.1)


class TestMutualInformation:
    """Tests for the mutual information functions"""

    diff_methods = ["backprop", "finite-diff"]

    @pytest.mark.all_interfaces
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

    @pytest.mark.all_interfaces
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
        import jax
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
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        params = jnp.array(params)

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
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_qnode_grad(self, param, diff_method):
        """Test that the gradient of mutual information works for QNodes
        with the autograd interface"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="autograd", diff_method=diff_method)
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

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        actual = qml.grad(circuit)(param)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_qnode_grad_jax(self, param, diff_method):
        """Test that the gradient of mutual information works for QNodes
        with the JAX interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        param = jnp.array(param)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
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

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        actual = jax.grad(circuit)(param)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_qnode_grad_jax_jit(self, param, diff_method):
        """Test that the gradient of mutual information works for QNodes
        with the JAX-jit interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        param = jnp.array(param)

        @qml.qnode(dev, interface="jax-jit", diff_method=diff_method)
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

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        actual = jax.jit(jax.grad(circuit))(param)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.tf
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_qnode_grad_tf(self, param, diff_method):
        """Test that the gradient of mutual information works for QNodes
        with the tensorflow interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        param = tf.Variable(param)

        @qml.qnode(dev, interface="tensorflow", diff_method=diff_method)
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

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        actual = tape.gradient(out, param)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.torch
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_qnode_grad_torch(self, param, diff_method):
        """Test that the gradient of mutual information works for QNodes
        with the torch interface"""
        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
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

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        actual = param.grad
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.all_interfaces
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

    @pytest.mark.all_interfaces
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


class TestRelativeEntropy:
    """Tests for the mutual information functions"""

    diff_methods = ["backprop", "finite-diff"]

    params = [[0.0, 0.0], [np.pi, 0.0], [0.0, np.pi], [0.123, 0.456], [0.789, 1.618]]

    # to avoid nan values in the gradient for relative entropy
    grad_params = [[0.123, 0.456], [0.789, 1.618]]

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize("param", params)
    def test_qnode_relative_entropy(self, device, interface, param):
        """Test that the relative entropy transform works for QNodes by comparing
        against analytic values"""
        dev = qml.device(device, wires=2)

        param = qml.math.asarray(np.array(param), like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit1(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        @qml.qnode(dev, interface=interface)
        def circuit2(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        rel_ent_circuit = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [1])
        actual = rel_ent_circuit((param[0],), (param[1],))

        # compare transform results with analytic results
        first_term = (
            0
            if np.cos(param[0] / 2) == 0
            else np.cos(param[0] / 2) ** 2
            * (np.log(np.cos(param[0] / 2) ** 2) - np.log(np.cos(param[1] / 2) ** 2))
        )
        second_term = (
            0
            if np.sin(param[0] / 2) == 0
            else np.sin(param[0] / 2) ** 2
            * (np.log(np.sin(param[0] / 2) ** 2) - np.log(np.sin(param[1] / 2) ** 2))
        )
        expected = first_term + second_term

        assert np.allclose(actual, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", params)
    def test_qnode_relative_entropy_jax_jit(self, param):
        """Test that the mutual information transform works for QNodes by comparing
        against analytic values, for the JAX-jit interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        param = jnp.array(param)

        @qml.qnode(dev, interface="jax-jit")
        def circuit1(params):
            qml.RY(params, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        @qml.qnode(dev, interface="jax-jit")
        def circuit2(params):
            qml.RY(params, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        rel_ent_circuit = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [1])
        actual = jax.jit(rel_ent_circuit)((param[0],), (param[1],))

        # compare transform results with analytic results
        first_term = (
            0
            if jnp.cos(param[0] / 2) == 0
            else jnp.cos(param[0] / 2) ** 2
            * (jnp.log(jnp.cos(param[0] / 2) ** 2) - jnp.log(jnp.cos(param[1] / 2) ** 2))
        )
        second_term = (
            0
            if jnp.sin(param[0] / 2) == 0
            else jnp.sin(param[0] / 2) ** 2
            * (jnp.log(jnp.sin(param[0] / 2) ** 2) - jnp.log(jnp.sin(param[1] / 2) ** 2))
        )
        expected = first_term + second_term

        assert np.allclose(actual, expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", grad_params)
    def test_qnode_grad(self, param):
        """Test that the gradient of relative entropy works for QNodes
        with the autograd interface"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit1(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit2(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        rel_ent_circuit = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [1])

        def wrapper(param0, param1):
            return rel_ent_circuit((param0,), (param1,))

        expected = [
            np.sin(param[0] / 2)
            * np.cos(param[0] / 2)
            * (np.log(np.tan(param[0] / 2) ** 2) - np.log(np.tan(param[1] / 2) ** 2)),
            np.cos(param[0] / 2) ** 2 * np.tan(param[1] / 2)
            - np.sin(param[0] / 2) ** 2 / np.tan(param[1] / 2),
        ]

        param0, param1 = np.array(param[0]), np.array(param[1])
        actual = qml.grad(wrapper)(param0, param1)

        assert np.allclose(actual, expected, atol=1e-8)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", grad_params)
    def test_qnode_grad_jax(self, param):
        """Test that the gradient of relative entropy works for QNodes
        with the JAX interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit1(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit2(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        rel_ent_circuit = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [1])

        def wrapper(param0, param1):
            return rel_ent_circuit((param0,), (param1,))

        expected = [
            np.sin(param[0] / 2)
            * np.cos(param[0] / 2)
            * (np.log(np.tan(param[0] / 2) ** 2) - np.log(np.tan(param[1] / 2) ** 2)),
            np.cos(param[0] / 2) ** 2 * np.tan(param[1] / 2)
            - np.sin(param[0] / 2) ** 2 / np.tan(param[1] / 2),
        ]

        param0, param1 = jnp.array(param[0]), jnp.array(param[1])
        actual = jax.grad(wrapper, argnums=[0, 1])(param0, param1)

        assert np.allclose(actual, expected, atol=1e-8)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", grad_params)
    def test_qnode_grad_jax_jit(self, param):
        """Test that the gradient of relative entropy works for QNodes
        with the JAX interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit1(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit2(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        rel_ent_circuit = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [1])

        def wrapper(param0, param1):
            return rel_ent_circuit((param0,), (param1,))

        expected = [
            np.sin(param[0] / 2)
            * np.cos(param[0] / 2)
            * (np.log(np.tan(param[0] / 2) ** 2) - np.log(np.tan(param[1] / 2) ** 2)),
            np.cos(param[0] / 2) ** 2 * np.tan(param[1] / 2)
            - np.sin(param[0] / 2) ** 2 / np.tan(param[1] / 2),
        ]

        param0, param1 = jnp.array(param[0]), jnp.array(param[1])
        actual = jax.jit(jax.grad(wrapper, argnums=[0, 1]))(param0, param1)

        assert np.allclose(actual, expected, atol=1e-8)

    @pytest.mark.tf
    @pytest.mark.parametrize("param", grad_params)
    def test_qnode_grad_tf(self, param):
        """Test that the gradient of relative entropy works for QNodes
        with the TensorFlow interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit1(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit2(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        expected = [
            np.sin(param[0] / 2)
            * np.cos(param[0] / 2)
            * (np.log(np.tan(param[0] / 2) ** 2) - np.log(np.tan(param[1] / 2) ** 2)),
            np.cos(param[0] / 2) ** 2 * np.tan(param[1] / 2)
            - np.sin(param[0] / 2) ** 2 / np.tan(param[1] / 2),
        ]

        param0, param1 = tf.Variable(param[0]), tf.Variable(param[1])
        with tf.GradientTape() as tape:
            out = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [1])((param0,), (param1,))

        actual = tape.gradient(out, [param0, param1])

        assert np.allclose(actual, expected, atol=1e-5)

    @pytest.mark.torch
    @pytest.mark.parametrize("param", grad_params)
    def test_qnode_grad_torch(self, param):
        """Test that the gradient of relative entropy works for QNodes
        with the Torch interface"""
        import torch

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit1(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit2(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

            first_expected = (
                np.sin(param[0] / 2)
                * np.cos(param[0] / 2)
                * (np.log(np.tan(param[0] / 2) ** 2) - np.log(np.tan(param[1] / 2) ** 2))
            )

        expected = [
            np.sin(param[0] / 2)
            * np.cos(param[0] / 2)
            * (np.log(np.tan(param[0] / 2) ** 2) - np.log(np.tan(param[1] / 2) ** 2)),
            np.cos(param[0] / 2) ** 2 * np.tan(param[1] / 2)
            - np.sin(param[0] / 2) ** 2 / np.tan(param[1] / 2),
        ]

        param0 = torch.tensor(param[0], requires_grad=True)
        param1 = torch.tensor(param[1], requires_grad=True)
        out = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [1])((param0,), (param1,))
        out.backward()

        actual = [param0.grad, param1.grad]

        assert np.allclose(actual, expected, atol=1e-8)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    def test_num_wires_mismatch(self, device, interface):
        """Test that an error is raised when the number of wires in the
        two QNodes are different"""
        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit1(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        @qml.qnode(dev, interface=interface)
        def circuit2(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        msg = "The two states must have the same number of wires"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            rel_ent_circuit = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [0, 1])

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    def test_full_wires(self, device):
        """Test that the relative entropy transform for full wires works for QNodes"""
        dev = qml.device(device, wires=1)

        @qml.qnode(dev)
        def circuit1(param):
            qml.RY(param, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit2(param):
            qml.RY(param, wires=0)
            return qml.state()

        rel_ent_circuit = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [0])

        x, y = np.array(0.3), np.array(0.7)

        # test that the circuit executes
        actual = rel_ent_circuit(x, y)

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    def test_qnode_no_args(self, device):
        """Test that the relative entropy transform works for QNodes without arguments"""
        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit1():
            qml.PauliY(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        @qml.qnode(dev)
        def circuit2():
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        rel_ent_circuit = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [1])

        # test that the circuit executes
        actual = rel_ent_circuit()

    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    def test_qnode_kwargs(self, device):
        """Test that the relative entropy transform works for QNodes that take keyword arguments"""
        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit1(param=0):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        @qml.qnode(dev)
        def circuit2(param=0):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        rel_ent_circuit = qml.qinfo.relative_entropy(circuit1, circuit2, [0], [1])

        x, y = np.array(0.4), np.array(0.8)
        actual = rel_ent_circuit(({"param": x},), ({"param": y},))

        # compare transform results with analytic results
        expected = (
            np.cos(x / 2) ** 2 * (np.log(np.cos(x / 2) ** 2) - np.log(np.cos(y / 2) ** 2))
        ) + (np.sin(x / 2) ** 2 * (np.log(np.sin(x / 2) ** 2) - np.log(np.sin(y / 2) ** 2)))

        assert np.allclose(actual, expected)

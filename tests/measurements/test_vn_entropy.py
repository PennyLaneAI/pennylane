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
"""Unit tests for the vn_entropy module"""
import copy

import numpy as np
import pytest

import pennylane as qml
from pennylane.interfaces import INTERFACE_MAP
from pennylane.measurements import VnEntropy, Shots
from pennylane.measurements.vn_entropy import VnEntropyMP
from pennylane.wires import Wires

# pylint: disable=too-many-arguments, no-member


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

    return -(
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


class TestInitialization:
    """Unit tests for the ``qml.vn_entropy`` function."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "state_vector,expected",
        [([1.0, 0.0, 0.0, 1.0] / qml.math.sqrt(2), qml.math.log(2)), ([1.0, 0.0, 0.0, 0.0], 0)],
    )
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    def test_vn_entropy(self, interface, state_vector, expected):
        """Tests the output of qml.vn_entropy"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.StatePrep(state_vector, wires=[0, 1])
            return qml.vn_entropy(wires=0)

        res = circuit()
        new_res = qml.vn_entropy(wires=0).process_state(
            state=circuit.device.state, wire_order=circuit.device.wires
        )
        assert qml.math.allclose(res, expected)
        assert qml.math.allclose(new_res, expected)
        assert INTERFACE_MAP.get(qml.math.get_interface(new_res)) == interface
        assert res.dtype == new_res.dtype

    def test_queue(self):
        """Test that the right measurement class is queued."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.vn_entropy(wires=0, log_base=2)

        circuit()

        assert isinstance(circuit.tape[0], VnEntropyMP)

    def test_copy(self):
        """Test that the ``__copy__`` method also copies the ``log_base`` information."""
        meas = qml.vn_entropy(wires=0, log_base=2)
        meas_copy = copy.copy(meas)
        assert meas_copy.log_base == 2
        assert meas_copy.wires == Wires(0)

    def test_properties(self):
        """Test that the properties are correct."""
        meas = qml.vn_entropy(wires=0)
        assert meas.numeric_type == float
        assert meas.return_type == VnEntropy

    @pytest.mark.parametrize("shots, shape", [(None, ()), (10, ()), ((1, 10), ((), ()))])
    def test_shape(self, shots, shape):
        """Test the ``shape`` method."""
        meas = qml.vn_entropy(wires=0)
        dev = qml.device("default.qubit", wires=1, shots=shots)

        assert meas.shape(dev, Shots(shots)) == shape


class TestIntegration:
    """Integration tests for the vn_entropy measurement function."""

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

    def test_shot_vec_error(self):
        """Test an error is raised when using shot vectors with vn_entropy."""
        dev = qml.device("default.qubit", wires=2, shots=[1, 10, 10, 1000])

        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.vn_entropy(wires=[0])

        with pytest.raises(
            NotImplementedError, match="Von Neumann entropy is not supported with shot vectors"
        ):
            circuit(0.5)

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
    @pytest.mark.parametrize("interface", ["torch"])
    def test_IsingXX_qnode_torch_entropy(self, param, wires, device, base, interface):
        """Test entropy for a QNode with torch interface."""
        import torch

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
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
    @pytest.mark.parametrize("interface", ["tf"])
    def test_IsingXX_qnode_tf_entropy(self, param, wires, device, base, interface):
        """Test entropy for a QNode with tf interface."""
        import tensorflow as tf

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
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
    @pytest.mark.parametrize("interface", ["tf"])
    def test_IsingXX_qnode_entropy_grad_tf(self, param, wires, base, diff_method, interface):
        """Test entropy for a QNode gradient with tf."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
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
    @pytest.mark.parametrize("interface", ["jax"])
    def test_IsingXX_qnode_jax_entropy(self, param, wires, device, base, interface):
        """Test entropy for a QNode with jax interface."""
        import jax.numpy as jnp

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
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
    @pytest.mark.parametrize("interface", ["jax"])
    def test_IsingXX_qnode_entropy_grad_jax(self, param, wires, base, diff_method, interface):
        """Test entropy for a QNode gradient with Jax."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        grad_entropy = jax.grad(circuit_entropy)(jax.numpy.array(param))
        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("interface", ["jax"])
    def test_IsingXX_qnode_jax_jit_entropy(self, param, wires, base, device, interface):
        """Test entropy for a QNode with jax-jit interface."""
        import jax
        import jax.numpy as jnp

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        entropy = jax.jit(circuit_entropy)(jnp.array(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qml.math.allclose(entropy, expected_entropy)
        jax.clear_caches()

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["jax-jit"])
    def test_IsingXX_qnode_entropy_grad_jax_jit(self, param, wires, base, diff_method, interface):
        """Test entropy for a QNode gradient with Jax-jit."""
        import jax

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit_entropy(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.vn_entropy(wires=wires, log_base=base)

        grad_entropy = jax.jit(jax.grad(circuit_entropy))(jax.numpy.array(param))

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        assert qml.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=1e-05)
        jax.clear_caches()

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

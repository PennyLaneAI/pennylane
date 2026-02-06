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

import pennylane as qp
from pennylane.exceptions import DeviceError
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
    """Unit tests for the ``qp.vn_entropy`` function."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "state_vector,expected",
        [([1.0, 0.0, 0.0, 1.0] / qp.math.sqrt(2), qp.math.log(2)), ([1.0, 0.0, 0.0, 0.0], 0)],
    )
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    def test_vn_entropy(self, interface, state_vector, expected):
        """Tests the output of qp.vn_entropy"""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface=interface)
        def circuit():
            qp.StatePrep(state_vector, wires=[0, 1])
            return qp.vn_entropy(wires=0)

        assert qp.math.allclose(circuit(), expected)

    def test_queue(self):
        """Test that the right measurement class is queued."""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit():
            return qp.vn_entropy(wires=0, log_base=2)

        tape = qp.workflow.construct_tape(circuit)()

        assert isinstance(tape[0], VnEntropyMP)

    def test_copy(self):
        """Test that the ``__copy__`` method also copies the ``log_base`` information."""
        meas = qp.vn_entropy(wires=0, log_base=2)
        meas_copy = copy.copy(meas)
        assert meas_copy.log_base == 2
        assert meas_copy.wires == Wires(0)

    def test_properties(self):
        """Test that the properties are correct."""
        meas = qp.vn_entropy(wires=0)
        assert meas.numeric_type == float

    @pytest.mark.parametrize("shots, shape", [(None, ()), (10, ())])
    def test_shape(self, shots, shape):
        """Test the ``shape`` method."""
        meas = qp.vn_entropy(wires=0)

        assert meas.shape(shots, 1) == shape

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "autograd"])
    @pytest.mark.parametrize(
        "subset_wires, log_base, expected_entropy",
        [
            ([0], None, 0.6931471805599453),  # Mixed state on first qubit (ln(2))
            ([1], None, 0.6931471805599453),  # Mixed state on second qubit (ln(2))
            ([0, 1], None, 0.0),  # Pure state |00>+|11> on both qubits
            ([0], 2, 1.0),
            ([1], 2, 1.0),
            ([0, 1], 2, 0.0),
        ],
    )
    def test_process_density_matrix_vn_entropy(
        self, interface, subset_wires, log_base, expected_entropy
    ):
        """Test von Neumann entropy calculation for different subsystems and log bases."""
        # Define a non-trivial two-qubit density matrix
        dm = qp.math.array(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            like=interface,
        )

        if interface == "tensorflow":
            dm = qp.math.cast(dm, "float64")

        vn_entropy = qp.vn_entropy(wires=subset_wires, log_base=log_base).process_density_matrix(
            dm, subset_wires
        )

        # Set tolerance based on interface
        atol = 1.0e-7 if interface == "torch" else 1.0e-8

        assert qp.math.allclose(
            vn_entropy, expected_entropy, atol=atol
        ), f"Subset wires: {subset_wires}, Log base: {log_base}, VN Entropy doesn't match expected value. Got {vn_entropy}, expected {expected_entropy}"

        # Test if the result is real
        assert qp.math.allclose(
            qp.math.imag(vn_entropy), 0, atol=atol
        ), f"VN Entropy should be real, but got imaginary part: {qp.math.imag(vn_entropy)}"


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

    diff_methods = ["backprop", "finite-diff"]

    @pytest.mark.parametrize("shots", [1000, [1, 10, 10, 1000]])
    def test_finite_shots_error(self, shots):
        """Test an error is raised when using shot vectors with vn_entropy."""
        dev = qp.device("default.qubit", wires=2)

        @qp.set_shots(shots)
        @qp.qnode(device=dev)
        def circuit(x):
            qp.Hadamard(wires=[0])
            qp.CRX(x, wires=[0, 1])
            return qp.vn_entropy(wires=[0])

        with pytest.raises(DeviceError, match="not accepted with finite shots on default.qubit"):
            circuit(0.5)

    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    def test_IsingXX_qnode_entropy(self, param, wires, device, base):
        """Test entropy for a QNode numpy."""

        dev = qp.device(device, wires=2)

        @qp.qnode(dev)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=wires, log_base=base)

        entropy = circuit_entropy(param)

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)
        assert qp.math.allclose(entropy, expected_entropy)

    @pytest.mark.autograd
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_IsingXX_qnode_entropy_grad(self, param, wires, base, diff_method):
        """Test entropy for a QNode gradient with autograd."""

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, diff_method=diff_method)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=wires, log_base=base)

        grad_entropy = qp.grad(circuit_entropy)(param)

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)
        assert qp.math.allclose(grad_entropy, grad_expected_entropy, atol=tol)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("interface", ["torch"])
    def test_IsingXX_qnode_torch_entropy(self, param, wires, device, base, interface):
        """Test entropy for a QNode with torch interface."""
        import torch

        dev = qp.device(device, wires=2)

        @qp.qnode(dev, interface=interface)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=wires, log_base=base)

        entropy = circuit_entropy(torch.tensor(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qp.math.allclose(entropy, expected_entropy)

    @pytest.mark.torch
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_IsingXX_qnode_entropy_grad_torch(self, param, wires, base, diff_method):
        """Test entropy for a QNode gradient with torch."""
        import torch

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=wires, log_base=base)

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        entropy = circuit_entropy(param)
        entropy.backward()
        grad_entropy = param.grad

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        assert qp.math.allclose(grad_entropy, grad_expected_entropy, atol=tol)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("interface", ["tf"])
    def test_IsingXX_qnode_tf_entropy(self, param, wires, device, base, interface):
        """Test entropy for a QNode with tf interface."""
        import tensorflow as tf

        dev = qp.device(device, wires=2)

        @qp.qnode(dev, interface=interface)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=wires, log_base=base)

        entropy = circuit_entropy(tf.Variable(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qp.math.allclose(entropy, expected_entropy)

    @pytest.mark.tf
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["tf"])
    def test_IsingXX_qnode_entropy_grad_tf(self, param, wires, base, diff_method, interface):
        """Test entropy for a QNode gradient with tf."""
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=wires, log_base=base)

        param = tf.Variable(param)
        with tf.GradientTape() as tape:
            entropy = circuit_entropy(param)

        grad_entropy = tape.gradient(entropy, param)

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        assert qp.math.allclose(grad_entropy, grad_expected_entropy, atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("interface", ["jax"])
    def test_IsingXX_qnode_jax_entropy(self, param, wires, device, base, interface):
        """Test entropy for a QNode with jax interface."""
        import jax.numpy as jnp

        dev = qp.device(device, wires=2)

        @qp.qnode(dev, interface=interface)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=wires, log_base=base)

        entropy = circuit_entropy(jnp.array(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qp.math.allclose(entropy, expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["jax"])
    def test_IsingXX_qnode_entropy_grad_jax(self, param, wires, base, diff_method, interface):
        """Test entropy for a QNode gradient with Jax."""
        import jax

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=wires, log_base=base)

        grad_entropy = jax.grad(circuit_entropy)(jax.numpy.array(param))
        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        assert qp.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=tol)

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

        dev = qp.device(device, wires=2)

        @qp.qnode(dev, interface=interface)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=wires, log_base=base)

        entropy = jax.jit(circuit_entropy)(jnp.array(param))

        expected_entropy = expected_entropy_ising_xx(param) / np.log(base)

        assert qp.math.allclose(entropy, expected_entropy)

    @pytest.mark.jax
    @pytest.mark.parametrize("wires", single_wires_list)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("base", base)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["jax-jit"])
    def test_IsingXX_qnode_entropy_grad_jax_jit(self, param, wires, base, diff_method, interface):
        """Test entropy for a QNode gradient with Jax-jit."""
        import jax

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=[0, 1])
            return qp.vn_entropy(wires=wires, log_base=base)

        grad_entropy = jax.jit(jax.grad(circuit_entropy))(jax.numpy.array(param))

        grad_expected_entropy = expected_entropy_grad_ising_xx(param) / np.log(base)

        assert qp.math.allclose(grad_entropy, grad_expected_entropy, rtol=1e-04, atol=1e-05)

    def test_qnode_entropy_custom_wires(self):
        """Test that entropy can be returned with custom wires."""

        dev = qp.device("default.qubit", wires=["a", 1])

        @qp.qnode(dev)
        def circuit_entropy(x):
            qp.IsingXX(x, wires=["a", 1])
            return qp.vn_entropy(wires=["a"])

        assert np.isclose(circuit_entropy(0.1), expected_entropy_ising_xx(0.1))

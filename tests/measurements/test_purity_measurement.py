# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the purity measurement process"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import PurityMP

# pylint: disable=too-many-arguments


def expected_purity_ising_xx(param):
    """Returns the analytical purity for subsystems of the IsingXX"""

    eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    return eig_1**2 + eig_2**2


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


class TestPurityUnitTest:
    """Tests for purity measurements"""

    def test_numeric_type(self):
        """Test that the numeric type of PurityMP is float."""
        m = PurityMP(wires=qml.wires.Wires(0))
        assert m.numeric_type is float

    @pytest.mark.parametrize("shots, shape", [(None, ()), (10, ())])
    def test_shape_new(self, shots, shape):
        """Test the ``shape_new`` method."""
        meas = qml.purity(wires=0)
        assert meas.shape(shots, 1) == shape

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "autograd"])
    def test_process_density_matrix_pure_state(self, interface):
        """Test purity calculation for a pure single-qubit state."""
        dm = qml.math.array([[1, 0], [0, 0]], like=interface)
        if interface == "tensorflow":
            dm = qml.math.cast(dm, "float64")
        wires = qml.wires.Wires(range(1))
        expected = qml.math.array(1.0, like=interface)
        if interface == "tensorflow":
            expected = qml.math.cast(expected, "float64")
        purity = qml.purity(wires=wires).process_density_matrix(dm, wires)
        atol = 1.0e-7 if interface == "torch" else 1.0e-8
        assert qml.math.allclose(purity, expected, atol=atol), f"Expected {expected}, got {purity}"

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "autograd"])
    @pytest.mark.parametrize(
        "subset_wires, expected_purity",
        [
            ([0], 1.0),  # Purity of first qubit (pure state after tracing out second qubit)
            ([1], 0.625),  # Purity of second qubit (mixed state after tracing out first qubit)
            ([0, 1], 0.655),  # Purity of both qubits (mixed state for the full system)
        ],
    )
    def test_process_density_matrix_purity_subsets(self, interface, subset_wires, expected_purity):
        """
        Test purity calculation of density matrix with subsets of wires.
        This test checks the purity calculation for different subsets of a two-qubit system,
        including single-qubit reduced states and the full two-qubit state.
        """
        # Define a non-trivial two-qubit density matrix
        # This represents a mixed state of two qubits
        dm = qml.math.array(
            [[0.15, 0, 0.1, 0], [0, 0.35, 0, 0.4], [0.1, 0, 0.1, 0], [0, 0.4, 0, 0.4]],
            like=interface,
        )

        # TensorFlow requires explicit casting to float64 for consistency
        if interface == "tensorflow":
            dm = qml.math.cast(dm, "float64")

        # Define the wires (qubits) of our system
        wires = qml.wires.Wires(range(2))

        # Calculate the purity using the PurityMP class
        purity = qml.purity(wires=subset_wires).process_density_matrix(dm, wires)

        # Set the tolerance for floating-point comparisons
        # TensorFlow and PyTorch may require a slightly higher tolerance due to numerical precision issues
        atol = 1.0e-7 if interface == "torch" else 1.0e-8

        # Assert that the calculated purity matches the expected value within the tolerance
        assert qml.math.allclose(
            purity, expected_purity, atol=atol
        ), f"Subsetwire: {subset_wires} Purity doesn't match expected value. Got {purity}, expected {expected_purity}"


class TestPurityIntegration:
    """Test the purity meausrement with qnodes and devices."""

    devices = ["default.qubit", "lightning.qubit", "default.mixed"]
    grad_supported_devices = ["default.qubit", "default.mixed"]
    mix_supported_devices = ["default.mixed"]

    diff_methods = ["backprop", "finite-diff"]

    parameters = np.linspace(0, 2 * np.pi, 3)
    probs = np.array([0.001, 0.01, 0.1, 0.2])

    wires_list = [([0], True), ([1], True), ([0, 1], False)]

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    def test_IsingXX_qnode_purity(self, device, param, wires, is_partial):
        """Tests purity for a qnode"""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.purity(wires=wires)

        purity = circuit(param)
        expected_purity = expected_purity_ising_xx(param) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.parametrize("device", mix_supported_devices)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("param", probs)
    def test_bit_flip_qnode_purity(self, device, wires, param, is_partial):
        """Tests purity for a qnode on a noisy device with bit flips"""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev)
        def circuit(p):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.BitFlip(p, wires=0)
            qml.BitFlip(p, wires=1)
            return qml.purity(wires=wires)

        purity = circuit(param)
        expected_purity = (
            0.5
            if is_partial
            else 4 * (0.5 - (1 - param) * param) ** 2 + 4 * (1 - param) ** 2 * param**2
        )
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.parametrize("device", grad_supported_devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_IsingXX_qnode_purity_grad(self, device, param, wires, is_partial, diff_method):
        """Tests purity for a qnode"""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.purity(wires=wires)

        grad_purity = qml.grad(circuit)(param)
        expected_grad = expected_purity_grad_ising_xx(param) if is_partial else 0
        assert qml.math.allclose(grad_purity, expected_grad, rtol=1e-04, atol=1e-05)

    @pytest.mark.parametrize("device", mix_supported_devices)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("param", probs)
    @pytest.mark.parametrize("diff_method", diff_methods)
    def test_bit_flip_qnode_purity_grad(self, device, wires, param, is_partial, diff_method):
        """Tests gradient of purity for a qnode on a noisy device with bit flips"""

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(p):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.BitFlip(p, wires=0)
            qml.BitFlip(p, wires=1)
            return qml.purity(wires=wires)

        purity_grad = qml.grad(circuit)(param)
        expected_purity_grad = 0 if is_partial else 32 * (param - 0.5) ** 3
        assert qml.math.allclose(purity_grad, expected_purity_grad, rtol=1e-04, atol=1e-05)

    @pytest.mark.jax
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("interface", ["jax"])
    def test_IsingXX_qnode_purity_jax(self, device, param, wires, is_partial, interface):
        """Test purity for a QNode with jax interface."""

        import jax.numpy as jnp

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.purity(wires=wires)

        purity = circuit(jnp.array(param))
        expected_purity = expected_purity_ising_xx(param) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.jax
    @pytest.mark.parametrize("device", grad_supported_devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["jax"])
    def test_IsingXX_qnode_purity_grad_jax(
        self, device, param, wires, is_partial, diff_method, interface
    ):
        """Test purity for a QNode gradient with Jax."""

        import jax

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.purity(wires=wires)

        grad_purity = jax.grad(circuit)(jax.numpy.array(param))
        grad_expected_purity = expected_purity_grad_ising_xx(param) if is_partial else 0

        assert qml.math.allclose(grad_purity, grad_expected_purity, rtol=1e-04, atol=1e-05)

    @pytest.mark.jax
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("interface", ["jax-jit"])
    def test_IsingXX_qnode_purity_jax_jit(self, device, param, wires, is_partial, interface):
        """Test purity for a QNode with jax interface."""

        import jax
        import jax.numpy as jnp

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.purity(wires=wires)

        purity = jax.jit(circuit)(jnp.array(param))
        expected_purity = expected_purity_ising_xx(param) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.jax
    @pytest.mark.parametrize("device", grad_supported_devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["jax-jit"])
    def test_IsingXX_qnode_purity_grad_jax_jit(
        self, device, param, wires, is_partial, diff_method, interface
    ):
        """Test purity for a QNode gradient with Jax."""

        import jax

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.purity(wires=wires)

        grad_purity = jax.jit(jax.grad(circuit))(jax.numpy.array(param))
        grad_expected_purity = expected_purity_grad_ising_xx(param) if is_partial else 0

        assert qml.math.allclose(grad_purity, grad_expected_purity, rtol=1e-04, atol=1e-05)

    @pytest.mark.torch
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("interface", ["torch"])
    def test_IsingXX_qnode_purity_torch(self, device, param, wires, is_partial, interface):
        """Tests purity for a qnode"""

        import torch

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.purity(wires=wires)

        purity = circuit(torch.tensor(param))
        expected_purity = expected_purity_ising_xx(param) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.torch
    @pytest.mark.parametrize("device", grad_supported_devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["torch"])
    def test_IsingXX_qnode_purity_grad_torch(
        self, device, param, wires, is_partial, diff_method, interface
    ):
        """Test purity for a QNode gradient with torch."""

        import torch

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.purity(wires=wires)

        expected_grad = expected_purity_grad_ising_xx(param) if is_partial else 0

        param = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        purity = circuit(param)
        purity.backward()  # pylint: disable=no-member
        grad_purity = param.grad

        assert qml.math.allclose(grad_purity, expected_grad, rtol=1e-04, atol=1e-05)

    @pytest.mark.tf
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("interface", ["tf"])
    def test_IsingXX_qnode_purity_tf(self, device, param, wires, is_partial, interface):
        """Tests purity for a qnode"""

        import tensorflow as tf

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.purity(wires=wires)

        purity = circuit(tf.Variable(param))
        expected_purity = expected_purity_ising_xx(param) if is_partial else 1
        assert qml.math.allclose(purity, expected_purity)

    @pytest.mark.tf
    @pytest.mark.parametrize("device", grad_supported_devices)
    @pytest.mark.parametrize("param", parameters)
    @pytest.mark.parametrize("wires,is_partial", wires_list)
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["tf"])
    def test_IsingXX_qnode_purity_grad_tf(
        self, device, param, wires, is_partial, diff_method, interface
    ):
        """Test purity for a QNode gradient with tf."""

        import tensorflow as tf

        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.purity(wires=wires)

        grad_expected_purity = expected_purity_grad_ising_xx(param) if is_partial else 0

        param = tf.Variable(param)
        with tf.GradientTape() as tape:
            purity = circuit(param)

        grad_purity = tape.gradient(purity, param)

        assert qml.math.allclose(grad_purity, grad_expected_purity, rtol=1e-04, atol=1e-05)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("param", parameters)
    def test_qnode_entropy_custom_wires(self, device, param):
        """Test that purity can be returned with custom wires."""

        dev = qml.device(device, wires=["a", 1])

        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=["a", 1])
            return qml.purity(wires=["a"])

        purity = circuit(param)
        expected_purity = expected_purity_ising_xx(param)
        assert qml.math.allclose(purity, expected_purity)

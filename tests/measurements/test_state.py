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
"""Unit tests for the state module"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.devices import DefaultQubit
from pennylane.measurements import State, density_matrix, expval, state


class TestState:
    """Tests for the state function"""

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_shape_and_dtype(self, wires):
        """Test that the state is of correct size and dtype for a trivial circuit"""

        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def func():
            return state()

        state_val = func()
        assert state_val.shape == (2**wires,)
        assert state_val.dtype == np.complex128

    def test_return_type_is_state(self):
        """Test that the return type of the observable is State"""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            return state()

        func()
        obs = func.qtape.observables
        assert len(obs) == 1
        assert obs[0].return_type is State

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_correct_ghz(self, wires):
        """Test that the correct state is returned when the circuit prepares a GHZ state"""

        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return state()

        state_val = func()
        assert np.allclose(np.sum(np.abs(state_val) ** 2), 1)
        assert np.allclose(state_val[0], 1 / np.sqrt(2))
        assert np.allclose(state_val[-1], 1 / np.sqrt(2))

    def test_return_with_other_types(self):
        """Test that an exception is raised when a state is returned along with another return
        type"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            return state(), expval(qml.PauliZ(1))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The state or density matrix cannot be returned in combination with other return types",
        ):
            func()

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_equal_to_dev_state(self, wires):
        """Test that the returned state is equal to the one stored in dev.state for a template
        circuit"""

        dev = qml.device("default.qubit", wires=wires)

        weights = np.random.random(
            qml.templates.StronglyEntanglingLayers.shape(n_layers=3, n_wires=wires)
        )

        @qml.qnode(dev)
        def func():
            qml.templates.StronglyEntanglingLayers(weights, wires=range(wires))
            return state()

        state_val = func()
        assert np.allclose(state_val, func.device.state)

    @pytest.mark.tf
    def test_interface_tf(self):
        """Test that the state correctly outputs in the tensorflow interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface="tf")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_expected = 0.25 * tf.ones(16)
        state_val = func()

        assert isinstance(state_val, tf.Tensor)
        assert state_val.dtype == tf.complex128
        assert np.allclose(state_expected, state_val.numpy())
        assert state_val.shape == (16,)

    @pytest.mark.torch
    def test_interface_torch(self):
        """Test that the state correctly outputs in the torch interface"""
        import torch

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface="torch")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_expected = 0.25 * torch.ones(16, dtype=torch.complex128)
        state_val = func()

        assert isinstance(state_val, torch.Tensor)
        assert state_val.dtype == torch.complex128
        assert torch.allclose(state_expected, state_val)
        assert state_val.shape == (16,)

    @pytest.mark.autograd
    def test_jacobian_not_supported(self):
        """Test if an error is raised if the jacobian method is called via qml.grad"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, diff_method="parameter-shift")
        def func(x):
            for i in range(4):
                qml.RX(x, wires=i)
            return state()

        d_func = qml.jacobian(func)

        with pytest.raises(
            ValueError,
            match="Computing the gradient of circuits that return the state is not supported",
        ):
            d_func(pnp.array(0.1, requires_grad=True))

    def test_no_state_capability(self, monkeypatch):
        """Test if an error is raised for devices that are not capable of returning the state.
        This is tested by changing the capability of default.qubit"""
        dev = qml.device("default.qubit", wires=1)
        capabilities = dev.capabilities().copy()
        capabilities["returns_state"] = False

        @qml.qnode(dev)
        def func():
            return state()

        with monkeypatch.context() as m:
            m.setattr(DefaultQubit, "capabilities", lambda *args, **kwargs: capabilities)
            with pytest.raises(qml.QuantumFunctionError, match="The current device is not capable"):
                func()

    def test_state_not_supported(self):
        """Test if an error is raised for devices inheriting from the base Device class,
        which do not currently support returning the state"""
        dev = qml.device("default.gaussian", wires=1)

        @qml.qnode(dev)
        def func():
            return state()

        with pytest.raises(qml.QuantumFunctionError, match="Returning the state is not supported"):
            func()

    @pytest.mark.parametrize("diff_method", ["best", "finite-diff", "parameter-shift"])
    def test_default_qubit(self, diff_method):
        """Test that the returned state is equal to the expected returned state for all of
        PennyLane's built in statevector devices"""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, diff_method=diff_method)
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_val = func()
        state_expected = 0.25 * np.ones(16)

        assert np.allclose(state_val, state_expected)
        assert np.allclose(state_val, dev.state)

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ["best", "finite-diff", "parameter-shift"])
    def test_default_qubit_tf(self, diff_method):
        """Test that the returned state is equal to the expected returned state for all of
        PennyLane's built in statevector devices"""

        dev = qml.device("default.qubit.tf", wires=4)

        @qml.qnode(dev, diff_method=diff_method)
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_val = func()
        state_expected = 0.25 * np.ones(16)

        assert np.allclose(state_val, state_expected)
        assert np.allclose(state_val, dev.state)

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["best", "finite-diff", "parameter-shift"])
    def test_default_qubit_autograd(self, diff_method):
        """Test that the returned state is equal to the expected returned state for all of
        PennyLane's built in statevector devices"""

        dev = qml.device("default.qubit.autograd", wires=4)

        @qml.qnode(dev, diff_method=diff_method)
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_val = func()
        state_expected = 0.25 * np.ones(16)

        assert np.allclose(state_val, state_expected)
        assert np.allclose(state_val, dev.state)

    @pytest.mark.tf
    def test_gradient_with_passthru_tf(self):
        """Test that the gradient of the state is accessible when using default.qubit.tf with the
        backprop diff_method."""
        import tensorflow as tf

        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def func(x):
            qml.RY(x, wires=0)
            return state()

        x = tf.Variable(0.1, dtype=tf.float64)

        with tf.GradientTape() as tape:
            result = func(x)

        grad = tape.jacobian(result, x)
        expected = tf.stack([-0.5 * tf.sin(x / 2), 0.5 * tf.cos(x / 2)])
        assert np.allclose(grad, expected)

    @pytest.mark.autograd
    def test_gradient_with_passthru_autograd(self):
        """Test that the gradient of the state is accessible when using default.qubit.autograd
        with the backprop diff_method."""
        from pennylane import numpy as anp

        dev = qml.device("default.qubit.autograd", wires=1)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def func(x):
            qml.RY(x, wires=0)
            return state()

        x = anp.array(0.1, requires_grad=True)

        def loss_fn(x):
            res = func(x)
            return anp.real(res)  # This errors without the real. Likely an issue with complex
            # numbers in autograd

        d_loss_fn = qml.jacobian(loss_fn)

        grad = d_loss_fn(x)
        expected = np.array([-0.5 * np.sin(x / 2), 0.5 * np.cos(x / 2)])
        assert np.allclose(grad, expected)

    @pytest.mark.parametrize("wires", [[0, 2, 3, 1], ["a", -1, "b", 1000]])
    def test_custom_wire_labels(self, wires):
        """Test the state when custom wire labels are used"""
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev, diff_method="parameter-shift")
        def func():
            for i in range(4):
                qml.Hadamard(wires[i])
            return state()

        state_expected = 0.25 * np.ones(16)
        state_val = func()

        assert np.allclose(state_expected, state_val)

    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_shape(self, shots):
        """Test that the shape is correct for qml.state."""
        dev = qml.device("default.qubit", wires=3, shots=shots)
        res = qml.state()
        assert res.shape(dev) == (1, 2**3)

    @pytest.mark.parametrize("s_vec", [(3, 2, 1), (1, 5, 10), (3, 1, 20)])
    def test_shape_shot_vector(self, s_vec):
        """Test that the shape is correct for qml.state with the shot vector too."""
        dev = qml.device("default.qubit", wires=3, shots=s_vec)
        res = qml.state()
        assert res.shape(dev) == (3, 2**3)

    def test_numeric_type(self):
        """Test that the numeric type of state measurements."""
        assert qml.state().numeric_type == complex
        assert qml.density_matrix(wires=[0, 1]).numeric_type == complex


class TestDensityMatrix:
    """Tests for the density matrix function"""

    @pytest.mark.parametrize("wires", range(2, 5))
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_density_matrix_shape_and_dtype(self, dev_name, wires):
        """Test that the density matrix is of correct size and dtype for a
        trivial circuit"""

        dev = qml.device(dev_name, wires=wires)

        @qml.qnode(dev)
        def circuit():
            return density_matrix([0])

        state_val = circuit()

        assert state_val.shape == (2, 2)
        assert state_val.dtype == np.complex128

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_return_type_is_state(self, dev_name):
        """Test that the return type of the observable is State"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            return density_matrix(0)

        func()
        obs = func.qtape.observables
        assert len(obs) == 1
        assert obs[0].return_type is State

    @pytest.mark.torch
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("diff_method", [None, "backprop"])
    def test_correct_density_matrix_torch(self, dev_name, diff_method):
        """Test that the correct density matrix is returned using torch interface."""
        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def func():
            qml.Hadamard(wires=0)
            return qml.density_matrix(wires=0)

        density_mat = func()

        assert np.allclose(
            np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]]), density_mat
        )

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("diff_method", [None, "backprop"])
    def test_correct_density_matrix_jax(self, dev_name, diff_method):
        """Test that the correct density matrix is returned using JAX interface."""
        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def func():
            qml.Hadamard(wires=0)
            return qml.density_matrix(wires=0)

        density_mat = func()

        assert np.allclose(
            np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]]), density_mat
        )

    @pytest.mark.tf
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("diff_method", [None, "backprop"])
    def test_correct_density_matrix_tf(self, dev_name, diff_method):
        """Test that the correct density matrix is returned using the TensorFlow interface."""
        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def func():
            qml.Hadamard(wires=0)
            return qml.density_matrix(wires=0)

        density_mat = func()

        assert np.allclose(
            np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]]), density_mat
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_product_state_first(self, dev_name):
        """Test that the correct density matrix is returned when
        tracing out a product state"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix(0)

        density_first = func()

        assert np.allclose(
            np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]), density_first
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_product_state_second(self, dev_name):
        """Test that the correct density matrix is returned when
        tracing out a product state"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix(1)

        density_second = func()
        assert np.allclose(
            np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]]), density_second
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_three_wires_first(self, dev_name):
        """Test that the correct density matrix for an example with three wires"""

        dev = qml.device(dev_name, wires=3)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix([0, 1])

        density_full = func()
        assert np.allclose(
            np.array(
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
                ]
            ),
            density_full,
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_three_wires_second(self, dev_name):
        """Test that the correct density matrix for an example with three wires"""

        dev = qml.device(dev_name, wires=3)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.CNOT(wires=[1, 2])
            return qml.density_matrix(wires=[1, 2])

        density = func()

        assert np.allclose(
            np.array(
                [
                    [
                        [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    ]
                ]
            ),
            density,
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_mixed_state(self, dev_name):
        """Test that the correct density matrix for an example with a mixed state"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.density_matrix(wires=[1])

        density = func()

        assert np.allclose(np.array([[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]]), density)

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_all_wires(self, dev_name):
        """Test that the correct density matrix is returned when all wires are given"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.density_matrix(wires=[0, 1])

        density = func()

        assert np.allclose(
            np.array(
                [
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                ]
            ),
            density,
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_return_with_other_types(self, dev_name):
        """Test that an exception is raised when a state is returned along with another return
        type"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            return density_matrix(0), expval(qml.PauliZ(1))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The state or density matrix"
            " cannot be returned in combination"
            " with other return types",
        ):
            func()

    def test_no_state_capability(self, monkeypatch):
        """Test if an error is raised for devices that are not capable of returning
        the density matrix. This is tested by changing the capability of default.qubit"""
        dev = qml.device("default.qubit", wires=2)
        capabilities = dev.capabilities().copy()
        capabilities["returns_state"] = False

        @qml.qnode(dev)
        def func():
            return density_matrix(0)

        with monkeypatch.context() as m:
            m.setattr(DefaultQubit, "capabilities", lambda *args, **kwargs: capabilities)
            with pytest.raises(
                qml.QuantumFunctionError,
                match="The current device is not capable" " of returning the state",
            ):
                func()

    def test_density_matrix_not_supported(self):
        """Test if an error is raised for devices inheriting from the base Device class,
        which do not currently support returning the state"""
        dev = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev)
        def func():
            return density_matrix(0)

        with pytest.raises(qml.QuantumFunctionError, match="Returning the state is not supported"):
            func()

    @pytest.mark.parametrize("wires", [[0, 2], ["a", -1]])
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_custom_wire_labels(self, wires, dev_name):
        """Test that the correct density matrix for an example with a mixed
        state when using custom wires"""

        dev = qml.device(dev_name, wires=wires)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            return qml.density_matrix(wires=wires[1])

        density = func()

        assert np.allclose(np.array([[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]]), density)

    @pytest.mark.parametrize("wires", [[3, 1], ["b", 1000]])
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_custom_wire_labels_all_wires(self, wires, dev_name):
        """Test that the correct density matrix for an example with a mixed
        state when using custom wires"""
        dev = qml.device(dev_name, wires=wires)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            return qml.density_matrix(wires=[wires[0], wires[1]])

        density = func()

        assert np.allclose(
            np.array(
                [
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                ]
            ),
            density,
        )

    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_shape(self, shots):
        """Test that the shape is correct for qml.density_matrix."""
        dev = qml.device("default.qubit", wires=3, shots=shots)
        res = qml.density_matrix(wires=[0, 1])
        assert res.shape(dev) == (1, 2**2, 2**2)

    @pytest.mark.parametrize("s_vec", [(3, 2, 1), (1, 5, 10), (3, 1, 20)])
    def test_shape_shot_vector(self, s_vec):
        """Test that the shape is correct for qml.density_matrix with the shot vector too."""
        dev = qml.device("default.qubit", wires=3, shots=s_vec)
        res = qml.density_matrix(wires=[0, 1])
        assert res.shape(dev) == (3, 2**2, 2**2)

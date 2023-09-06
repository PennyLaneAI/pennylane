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
from pennylane.measurements import (
    State,
    StateMP,
    DensityMatrixMP,
    Shots,
    density_matrix,
    expval,
    state,
)
from pennylane.math.quantum import reduce_statevector, reduce_dm
from pennylane.math.matrix_manipulation import _permute_dense_matrix
from pennylane.wires import Wires, WireError


class TestStateMP:
    """Tests for the State measurement process."""

    @pytest.mark.parametrize(
        "vec",
        [
            np.array([0.6, 0.8j]),
            np.eye(4)[3],
            np.array([0.48j, 0.48, -0.64j, 0.36]),
        ],
    )
    def test_process_state_vector(self, vec):
        """Test the processing of a state vector."""

        mp = StateMP(wires=None)
        assert mp.return_type == State
        assert mp.numeric_type is complex

        processed = mp.process_state(vec, None)
        assert qml.math.allclose(processed, vec)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "jax", "torch", "tensorflow"])
    def test_state_returns_itself_if_wires_match(self, interface):
        """Test that when wire_order matches the StateMP, the state is returned."""
        ket = qml.math.array([0.48j, 0.48, -0.64j, 0.36], like=interface)
        assert StateMP(wires=[1, 0]).process_state(ket, wire_order=Wires([1, 0])) is ket

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "jax", "torch", "tensorflow"])
    @pytest.mark.parametrize("wires, wire_order", [([1, 0], [0, 1]), (["b", "a"], ["a", "b"])])
    def test_reorder_state(self, interface, wires, wire_order):
        """Test that a state can be re-ordered."""
        ket = qml.math.array([0.48j, 0.48, -0.64j, 0.36], like=interface)
        result = StateMP(wires=wires).process_state(ket, wire_order=Wires(wire_order))
        assert qml.math.allclose(result, np.array([0.48j, -0.64j, 0.48, 0.36]))
        assert qml.math.get_interface(ket) == interface

    @pytest.mark.parametrize(
        "mp_wires, expected_state",
        [
            ([0, 1, 2], [1, 0, 2, 0, 3, 0, 4, 0]),
            ([2, 0, 1], [1, 2, 3, 4, 0, 0, 0, 0]),
            ([1, 0, 2], [1, 0, 3, 0, 2, 0, 4, 0]),
            ([1, 2, 0], [1, 3, 0, 0, 2, 4, 0, 0]),
        ],
    )
    @pytest.mark.parametrize("custom_wire_labels", [False, True])
    def test_expand_state_over_wires(self, mp_wires, expected_state, custom_wire_labels):
        """Test the expanded state is correctly ordered with extra wires being zero."""
        wire_order = [0, 1]
        if custom_wire_labels:
            # non-lexicographical-ordered
            wire_map = {0: "b", 1: "c", 2: "a"}
            mp_wires = [wire_map[w] for w in mp_wires]
            wire_order = ["b", "c"]
        mp = StateMP(wires=mp_wires)
        ket = np.arange(1, 5)
        result = mp.process_state(ket, wire_order=Wires(wire_order))
        assert np.array_equal(result, expected_state)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "jax", "torch", "tensorflow"])
    def test_expand_state_all_interfaces(self, interface):
        """Test that expanding the state over wires preserves interface."""
        mp = StateMP(wires=[4, 2, 0, 1])
        ket = qml.math.array([0.48j, 0.48, -0.64j, 0.36], like=interface)
        result = mp.process_state(ket, wire_order=Wires([1, 2]))
        reshaped = qml.math.reshape(result, (2, 2, 2, 2))
        assert qml.math.all(reshaped[1, :, 1, :] == 0)
        assert qml.math.allclose(reshaped[0, :, 0, :], np.array([[0.48j, -0.64j], [0.48, 0.36]]))
        if interface != "autograd":
            # autograd.numpy.pad drops pennylane tensor for some reason
            assert qml.math.get_interface(result) == interface

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "jax", "torch", "tensorflow"])
    def test_expand_state_batched_all_interfaces(self, interface):
        """Test that expanding the state over wires preserves interface."""
        mp = StateMP(wires=[4, 2, 0, 1])
        ket = qml.math.array(
            [
                [0.48j, 0.48, -0.64j, 0.36],
                [0.3, 0.4, 0.5, 1 / np.sqrt(2)],
                [-0.3, -0.4, -0.5, -1 / np.sqrt(2)],
            ],
            like=interface,
        )
        result = mp.process_state(ket, wire_order=Wires([1, 2]))
        reshaped = qml.math.reshape(result, (3, 2, 2, 2, 2))
        assert qml.math.all(reshaped[:, 1, :, 1, :] == 0)
        assert qml.math.allclose(
            reshaped[:, 0, :, 0, :],
            np.array(
                [
                    [[0.48j, -0.64j], [0.48, 0.36]],
                    [[0.3, 0.5], [0.4, 1 / np.sqrt(2)]],
                    [[-0.3, -0.5], [-0.4, -1 / np.sqrt(2)]],
                ],
            ),
        )
        if interface != "autograd":
            # autograd.numpy.pad drops pennylane tensor for some reason
            assert qml.math.get_interface(result) == interface

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "wires,expected",
        [
            ([1, 0], np.array([0.48j, -0.64j, 0.48, 0.36])),
            ([2, 1, 0], np.array([0.48j, -0.64j, 0.48, 0.36, 0.0, 0.0, 0.0, 0.0])),
        ],
    )
    def test_state_jax_jit(self, wires, expected):
        """Test that re-ordering and expanding works with jax-jit."""
        import jax

        @jax.jit
        def get_state(ket):
            return StateMP(wires=wires).process_state(ket, wire_order=Wires([0, 1]))

        result = get_state(jax.numpy.array([0.48j, 0.48, -0.64j, 0.36]))
        assert qml.math.allclose(result, expected)
        assert isinstance(result, jax.Array)

    def test_wire_ordering_error(self):
        """Test that a wire order error is raised when unknown wires are given."""
        with pytest.raises(WireError, match=r"Unexpected wires \{2\} found in wire order"):
            StateMP(wires=[0, 1]).process_state([1, 0], wire_order=[2])


class TestDensityMatrixMP:
    """Tests for the DensityMatrix measurement process"""

    @pytest.mark.parametrize(
        "vec, wires",
        [
            (np.array([0.6, 0.8j]), [0]),
            (np.eye(4, dtype=np.complex64)[3], [0]),
            (np.array([0.48j, 0.48, -0.64j, 0.36]), [0, 1]),
            (np.array([0.48j, 0.48, -0.64j, 0.36]), [0]),
        ],
    )
    def test_process_state_matrix_from_vec(self, vec, wires):
        """Test the processing of a state vector into a matrix."""

        mp = DensityMatrixMP(wires=wires)
        assert mp.return_type == State
        assert mp.numeric_type is complex

        num_wires = int(np.log2(len(vec)))
        processed = mp.process_state(vec, list(range(num_wires)))
        assert qml.math.shape(processed) == (2 ** len(wires), 2 ** len(wires))
        if len(wires) == num_wires:
            exp = np.outer(vec, vec.conj())
        else:
            exp = reduce_statevector(vec, wires)
        assert qml.math.allclose(processed, exp)

    @pytest.mark.xfail(
        reason="DensityMatrixMP.process_state no longer supports density matrix parameters"
    )
    @pytest.mark.parametrize(
        "mat, wires",
        [
            (np.eye(4, dtype=np.complex64) / 4, [0]),
            (np.eye(4, dtype=np.complex64) / 4, [1, 0]),
            (np.outer([0.6, 0.8j], [0.6, -0.8j]), [0]),
            (np.outer([0.36j, 0.48, 0.64, 0.48j], [-0.36j, 0.48, 0.64, -0.48j]), [0, 1]),
            (np.outer([0.36j, 0.48, 0.64, 0.48j], [-0.36j, 0.48, 0.64, -0.48j]), [0]),
            (np.outer([0.36j, 0.48, 0.64, 0.48j], [-0.36j, 0.48, 0.64, -0.48j]), [1]),
        ],
    )
    def test_process_state_matrix_from_matrix(self, mat, wires):
        """Test the processing of a density matrix into a matrix."""

        mp = DensityMatrixMP(wires=wires)
        assert mp.return_type == State
        assert mp.numeric_type is complex

        num_wires = int(np.log2(len(mat)))
        order = list(range(num_wires))
        processed = mp.process_state(mat, order)
        assert qml.math.shape(processed) == (2 ** len(wires), 2 ** len(wires))
        if len(wires) == num_wires:
            exp = _permute_dense_matrix(mat, wires, order, None)
        else:
            exp = reduce_dm(mat, wires)
        assert qml.math.allclose(processed, exp)


class TestState:
    """Tests for the state function"""

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_shape_and_dtype(self, wires):
        """Test that the state is of correct size and dtype for a trivial circuit"""

        dev = qml.device("default.qubit.legacy", wires=wires)

        @qml.qnode(dev)
        def func():
            return state()

        state_val = func()
        assert state_val.shape == (2**wires,)
        assert state_val.dtype == np.complex128

    def test_return_type_is_state(self):
        """Test that the return type of the observable is State"""

        dev = qml.device("default.qubit.legacy", wires=1)

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

        dev = qml.device("default.qubit.legacy", wires=wires)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return state()

        state_val = func()
        assert np.allclose(np.sum(np.abs(state_val) ** 2), 1)
        # pylint: disable=unsubscriptable-object
        assert np.allclose(state_val[0], 1 / np.sqrt(2))
        assert np.allclose(state_val[-1], 1 / np.sqrt(2))

        assert np.allclose(state().process_state(state=dev.state, wire_order=dev.wires), state_val)

    def test_return_with_other_types(self):
        """Test that an exception is raised when a state is returned along with another return
        type"""

        dev = qml.device("default.qubit.legacy", wires=2)

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

        dev = qml.device("default.qubit.legacy", wires=wires)

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

        dev = qml.device("default.qubit.legacy", wires=4)

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

        dev = qml.device("default.qubit.legacy", wires=4)

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
        dev = qml.device("default.qubit.legacy", wires=4)

        @qml.qnode(dev, diff_method="parameter-shift")
        def func(x):
            for i in range(4):
                qml.RX(x, wires=i)
            return state()

        d_func = qml.jacobian(func)

        with pytest.raises(
            ValueError,
            match=(
                "Computing the gradient of circuits that return the state with the "
                "parameter-shift rule gradient transform is not supported"
            ),
        ):
            d_func(pnp.array(0.1, requires_grad=True))

    def test_no_state_capability(self, monkeypatch):
        """Test if an error is raised for devices that are not capable of returning the state.
        This is tested by changing the capability of default.qubit"""
        dev = qml.device("default.qubit.legacy", wires=1)
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

        dev = qml.device("default.qubit.legacy", wires=4)

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

        dev = qml.device("default.qubit.autograd", wires=1)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def func(x):
            qml.RY(x, wires=0)
            return state()

        x = pnp.array(0.1, requires_grad=True)

        def loss_fn(x):
            res = func(x)
            return pnp.real(res)  # This errors without the real. Likely an issue with complex
            # numbers in autograd

        d_loss_fn = qml.jacobian(loss_fn)

        grad = d_loss_fn(x)
        expected = np.array([-0.5 * np.sin(x / 2), 0.5 * np.cos(x / 2)])
        assert np.allclose(grad, expected)

    @pytest.mark.parametrize("wires", [[0, 2, 3, 1], ["a", -1, "b", 1000]])
    def test_custom_wire_labels(self, wires):
        """Test the state when custom wire labels are used"""
        dev = qml.device("default.qubit.legacy", wires=wires)

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
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)
        res = qml.state()
        assert res.shape(dev, Shots(shots)) == (2**3,)

    @pytest.mark.parametrize("s_vec", [(3, 2, 1), (1, 5, 10), (3, 1, 20)])
    def test_shape_shot_vector(self, s_vec):
        """Test that the shape is correct for qml.state with the shot vector too."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=s_vec)
        res = qml.state()
        assert res.shape(dev, Shots(s_vec)) == ((2**3,), (2**3,), (2**3,))

    def test_numeric_type(self):
        """Test that the numeric type of state measurements."""
        assert qml.state().numeric_type == complex
        assert qml.density_matrix(wires=[0, 1]).numeric_type == complex


class TestDensityMatrix:
    """Tests for the density matrix function"""

    # pylint: disable=too-many-public-methods

    @pytest.mark.parametrize("wires", range(2, 5))
    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
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

    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
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
    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
    @pytest.mark.parametrize("diff_method", [None, "backprop"])
    def test_correct_density_matrix_torch(self, dev_name, diff_method):
        """Test that the correct density matrix is returned using torch interface."""
        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def func():
            qml.Hadamard(wires=0)
            return qml.density_matrix(wires=0)

        density_mat = func()
        expected = np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]])
        assert np.allclose(expected, density_mat)

        dev = func.device

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=0).process_state(state=dev.state, wire_order=dev.wires),
            )

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
    @pytest.mark.parametrize("diff_method", [None, "backprop"])
    def test_correct_density_matrix_jax(self, dev_name, diff_method):
        """Test that the correct density matrix is returned using JAX interface."""
        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def func():
            qml.Hadamard(wires=0)
            return qml.density_matrix(wires=0)

        density_mat = func()
        expected = np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]])

        assert np.allclose(expected, density_mat)

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=0).process_state(state=dev.state, wire_order=dev.wires),
            )

    @pytest.mark.tf
    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
    @pytest.mark.parametrize("diff_method", [None, "backprop"])
    def test_correct_density_matrix_tf(self, dev_name, diff_method):
        """Test that the correct density matrix is returned using the TensorFlow interface."""
        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev, interface="tf", diff_method=diff_method)
        def func():
            qml.Hadamard(wires=0)
            return qml.density_matrix(wires=0)

        density_mat = func()
        expected = np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]])

        assert np.allclose(expected, density_mat)

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=0).process_state(state=dev.state, wire_order=dev.wires),
            )

    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
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
        expected = np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]])

        assert np.allclose(expected, density_first)

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=0).process_state(state=dev.state, wire_order=dev.wires),
            )

    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
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
        expected = np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]])
        assert np.allclose(expected, density_second)

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=1).process_state(state=dev.state, wire_order=dev.wires),
            )

    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
    @pytest.mark.parametrize("return_wire_order", ([0, 1], [1, 0]))
    def test_correct_density_matrix_product_state_both(self, dev_name, return_wire_order):
        """Test that the correct density matrix is returned
        for a full product state on two wires."""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix(return_wire_order)

        density_both = func()
        single_statevectors = [[0, 1j], [1 / np.sqrt(2), 1 / np.sqrt(2)]]
        expected_statevector = np.kron(*[single_statevectors[w] for w in return_wire_order])
        expected = np.outer(expected_statevector.conj(), expected_statevector)

        assert np.allclose(expected, density_both)

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=return_wire_order).process_state(
                    state=dev.state, wire_order=dev.wires
                ),
            )

    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
    def test_correct_density_matrix_three_wires_first_two(self, dev_name):
        """Test that the correct density matrix is returned for an example with three wires,
        and tracing out the third wire."""

        dev = qml.device(dev_name, wires=3)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix([0, 1])

        density_full = func()
        expected = np.array(
            [
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
            ]
        )
        assert np.allclose(expected, density_full)

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=[0, 1]).process_state(
                    state=dev.state, wire_order=dev.wires
                ),
            )

    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
    def test_correct_density_matrix_three_wires_last_two(self, dev_name):
        """Test that the correct density matrix is returned for an example with three wires,
        and tracing out the first wire."""

        dev = qml.device(dev_name, wires=3)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.CNOT(wires=[1, 2])
            return qml.density_matrix(wires=[1, 2])

        density = func()
        expected = np.array(
            [
                [
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                ]
            ]
        )

        assert np.allclose(expected, density)

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=[1, 2]).process_state(
                    state=dev.state, wire_order=dev.wires
                ),
            )

    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
    @pytest.mark.parametrize(
        "return_wire_order", ([0], [1], [2], [0, 1], [1, 0], [0, 2], [2, 0], [1, 2, 0], [2, 1, 0])
    )
    def test_correct_density_matrix_three_wires_product(self, dev_name, return_wire_order):
        """Test that the correct density matrix is returned for an example with
        three wires and a product state, tracing out various combinations."""

        dev = qml.device(dev_name, wires=3)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.PauliX(1)
            qml.PauliZ(2)
            return density_matrix(return_wire_order)

        density_full = func()

        single_states = [[1 / np.sqrt(2), 1 / np.sqrt(2)], [0, 1], [1, 0]]
        if len(return_wire_order) == 1:
            exp_statevector = np.array(single_states[return_wire_order[0]])
        elif len(return_wire_order) == 2:
            i, j = return_wire_order
            exp_statevector = np.kron(single_states[i], single_states[j])
        elif len(return_wire_order) == 3:
            i, j, k = return_wire_order
            exp_statevector = np.kron(np.kron(single_states[i], single_states[j]), single_states[k])

        expected = np.outer(exp_statevector.conj(), exp_statevector)
        assert np.allclose(expected, density_full)

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=return_wire_order).process_state(
                    state=dev.state, wire_order=dev.wires
                ),
            )

    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
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

    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
    def test_correct_density_matrix_all_wires(self, dev_name):
        """Test that the correct density matrix is returned when all wires are given"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.density_matrix(wires=[0, 1])

        density = func()
        expected = np.array(
            [
                [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
            ]
        )

        assert np.allclose(expected, density)

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=[0, 1]).process_state(
                    state=dev.state, wire_order=dev.wires
                ),
            )

    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
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
        dev = qml.device("default.qubit.legacy", wires=2)
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
    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
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
        expected = np.array([[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]])

        assert np.allclose(expected, density)

        if dev_name != "default.mixed":
            assert np.allclose(
                expected,
                qml.density_matrix(wires=wires[1]).process_state(
                    state=dev.state, wire_order=dev.wires
                ),
            )

    @pytest.mark.parametrize("wires", [[3, 1], ["b", 1000]])
    @pytest.mark.parametrize("dev_name", ["default.qubit.legacy", "default.mixed"])
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
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)
        res = qml.density_matrix(wires=[0, 1])
        assert res.shape(dev, Shots(shots)) == (2**2, 2**2)

    @pytest.mark.parametrize("s_vec", [(3, 2, 1), (1, 5, 10), (3, 1, 20)])
    def test_shape_shot_vector(self, s_vec):
        """Test that the shape is correct for qml.density_matrix with the shot vector too."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=s_vec)
        res = qml.density_matrix(wires=[0, 1])
        assert res.shape(dev, Shots(s_vec)) == (
            (2**2, 2**2),
            (2**2, 2**2),
            (2**2, 2**2),
        )

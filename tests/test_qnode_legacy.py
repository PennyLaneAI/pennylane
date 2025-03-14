# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the QNode"""
import copy

# pylint: disable=import-outside-toplevel, protected-access, no-member
import warnings
from functools import partial

import numpy as np
import pytest
from default_qubit_legacy import DefaultQubitLegacy
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane import QNode
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane.resource import Resources
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn


def dummyfunc():
    """dummy func."""
    return None


class DummyDevice(qml.devices.LegacyDevice):
    """A minimal device that does not do anything."""

    author = "some string"
    name = "my legacy device"
    short_name = "something"
    version = 0.0

    observables = {"PauliX", "PauliY", "PauliZ"}
    operations = {"Rot", "RX", "RY", "RZ", "PauliX", "PauliY", "PauliZ", "CNOT"}
    pennylane_requires = 0.38

    def capabilities(self):
        return {"passthru_devices": {"autograd": "default.qubit.legacy"}}

    def reset(self):
        pass

    # pylint: disable=unused-argument
    def apply(self, operation, wires, par):
        return 0.0

    # pylint: disable=unused-argument
    def expval(self, observable, wires, par):
        return 0.0


class DeviceDerivatives(DummyDevice):
    """A dummy device with a jacobian."""

    # _capabilities = {"provides_jacobian": True}

    def capabilities(self):
        capabilities = super().capabilities().copy()
        capabilities.update(
            provides_jacobian=True,
        )
        return capabilities


# pylint: disable=too-many-public-methods
class TestValidation:
    """Tests for QNode creation and validation"""

    def test_invalid_interface(self):
        """Test that an exception is raised for an invalid interface"""
        dev = DefaultQubitLegacy(wires=1)
        test_interface = "something"
        expected_error = rf"Unknown interface {test_interface}\. Interface must be one of"

        with pytest.raises(ValueError, match=expected_error):
            QNode(dummyfunc, dev, interface="something")

    def test_changing_invalid_interface(self):
        """Test that an exception is raised for an invalid interface
        on a pre-existing QNode"""
        dev = DefaultQubitLegacy(wires=1)
        test_interface = "something"

        @qnode(dev)
        def circuit(x):
            """a circuit."""
            qml.RX(x, wires=0)
            return qml.probs(wires=0)

        expected_error = rf"Unknown interface {test_interface}\. Interface must be one of"

        with pytest.raises(ValueError, match=expected_error):
            circuit.interface = test_interface

    def test_invalid_device(self):
        """Test that an exception is raised for an invalid device"""
        with pytest.raises(qml.QuantumFunctionError, match="Invalid device"):
            QNode(dummyfunc, None)

    def test_unknown_diff_method_string(self):
        """Test that an exception is raised for an unknown differentiation method string"""
        dev = DefaultQubitLegacy(wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="Differentiation method hello not recognized"
        ):
            QNode(dummyfunc, dev, interface="autograd", diff_method="hello")

    def test_unknown_diff_method_type(self):
        """Test that an exception is raised for an unknown differentiation method type"""
        dev = DefaultQubitLegacy(wires=1)

        with pytest.raises(
            ValueError,
            match="Differentiation method 5 must be a str, TransformDispatcher, or None",
        ):
            QNode(dummyfunc, dev, interface="autograd", diff_method=5)

    def test_adjoint_finite_shots(self):
        """Tests that QuantumFunctionError is raised with the adjoint differentiation method
        on QNode construction when the device has finite shots
        """

        dev = DefaultQubitLegacy(wires=1, shots=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="does not support adjoint with requested circuit."
        ):

            @qnode(dev, diff_method="adjoint")
            def circ():
                return qml.expval(qml.PauliZ(0))

            circ()

    @pytest.mark.autograd
    def test_sparse_diffmethod_error(self):
        """Test that an error is raised when the observable is SparseHamiltonian and the
        differentiation method is not parameter-shift."""
        dev = DefaultQubitLegacy(wires=2, shots=None)

        @qnode(dev, diff_method="backprop")
        def circuit(param):
            qml.RX(param, wires=0)
            return qml.expval(qml.SparseHamiltonian(csr_matrix(np.eye(4)), [0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError, match="does not support backprop with requested circuit."
        ):
            qml.grad(circuit, argnum=0)([0.5])

    def test_qnode_print(self):
        """Test that printing a QNode object yields the right information."""
        dev = DefaultQubitLegacy(wires=1)

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qn = QNode(func, dev)

        assert (
            repr(qn)
            == "<QNode: wires=1, device='default.qubit.legacy', interface='auto', diff_method='best'>"
        )

        qn = QNode(func, dev, interface="autograd")

        assert (
            repr(qn)
            == "<QNode: wires=1, device='default.qubit.legacy', interface='autograd', diff_method='best'>"
        )

        # QNode can still be executed
        assert np.allclose(qn(0.5), np.cos(0.5), rtol=0)

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            grad = qml.grad(qn)(0.5)

        assert np.allclose(grad, 0)

    # pylint: disable=unused-variable
    def test_unrecognized_kwargs_raise_warning(self):
        """Test that passing gradient_kwargs not included in qml.gradients.SUPPORTED_GRADIENT_KWARGS raises warning"""
        dev = DefaultQubitLegacy(wires=2)

        with warnings.catch_warnings(record=True) as w:

            @qml.qnode(dev, gradient_kwargs={"random_kwarg": qml.gradients.finite_diff})
            def circuit(params):
                qml.RX(params[0], wires=0)
                return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))

            assert len(w) == 1
            assert "not included in the list of standard qnode gradient kwargs" in str(w[0].message)

    # pylint: disable=unused-variable
    def test_incorrect_diff_method_kwargs_raise_warning(self):
        """Tests that using one of the incorrect kwargs previously used in some examples in PennyLane
        (grad_method, gradient_fn) to set the qnode diff_method raises a warning"""
        dev = DefaultQubitLegacy(wires=2)

        with warnings.catch_warnings(record=True) as w:

            @qml.qnode(dev, grad_method=qml.gradients.finite_diff)
            def circuit0(params):
                qml.RX(params[0], wires=0)
                return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))

            @qml.qnode(dev, gradient_fn=qml.gradients.finite_diff)
            def circuit2(params):
                qml.RX(params[0], wires=0)
                return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))

        assert len(w) == 2
        assert "Use diff_method instead" in str(w[0].message)
        assert "Use diff_method instead" in str(w[1].message)

    def test_auto_interface_tracker_device_switched(self):
        """Test that checks that the tracker is switched to the new device."""
        dev = DefaultQubitLegacy(wires=1)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params, wires=0)
            return qml.expval(qml.PauliZ(0))

        with qml.Tracker(dev) as tracker:
            circuit(qml.numpy.array(0.1, requires_grad=True))

        assert tracker.totals == {"executions": 1, "batches": 1, "batch_len": 1}
        assert np.allclose(tracker.history.pop("results")[0], 0.99500417)
        assert tracker.history == {
            "executions": [1],
            "shots": [None],
            "batches": [1],
            "batch_len": [1],
            "resources": [Resources(1, 1, {"RX": 1}, {1: 1}, 1)],
        }

    def test_autograd_interface_device_switched_no_warnings(self):
        """Test that checks that no warning is raised for device switch when you define an interface,
        except for the deprecation warnings which will be caught by the fixture."""
        dev = DefaultQubitLegacy(wires=1)

        @qml.qnode(dev, interface="autograd")
        def circuit(params):
            qml.RX(params, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit(qml.numpy.array(0.1, requires_grad=True))

    def test_not_giving_mode_kwarg_does_not_raise_warning(self):
        """Test that not providing a value for mode does not raise a warning."""

        with warnings.catch_warnings(record=True) as record:
            qml.QNode(lambda f: f, DefaultQubitLegacy(wires=1))

        assert len(record) == 0


class TestTapeConstruction:
    """Tests for the tape construction"""

    def test_basic_tape_construction(self, tol):
        """Test that a quantum tape is properly constructed"""
        dev = DefaultQubitLegacy(wires=2)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        qn = QNode(func, dev)

        x = pnp.array(0.12, requires_grad=True)
        y = pnp.array(0.54, requires_grad=True)

        res = qn(x, y)
        tape = qml.workflow.construct_tape(qn)(x, y)

        assert isinstance(tape, QuantumScript)
        assert len(tape.operations) == 3
        assert len(tape.observables) == 1
        assert tape.num_params == 2
        assert tape.shots.total_shots is None

        expected = qml.execute([tape], dev, None)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # when called, a new quantum tape is constructed
        old_tape = tape
        res2 = qn(x, y)
        new_tape = qml.workflow.construct_tape(qn)(x, y)

        assert np.allclose(res, res2, atol=tol, rtol=0)
        assert new_tape is not old_tape

    def test_returning_non_measurements(self):
        """Test that an exception is raised if a non-measurement
        is returned from the QNode."""
        dev = DefaultQubitLegacy(wires=2)

        def func0(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return 5

        qn = QNode(func0, dev)

        with pytest.raises(
            qml.QuantumFunctionError, match="must return either a single measurement"
        ):
            qn(5, 1)

        def func2(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), 5

        qn = QNode(func2, dev)

        with pytest.raises(
            qml.QuantumFunctionError, match="must return either a single measurement"
        ):
            qn(5, 1)

        def func3(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return []

        qn = QNode(func3, dev)

        with pytest.raises(
            qml.QuantumFunctionError, match="must return either a single measurement"
        ):
            qn(5, 1)

    def test_inconsistent_measurement_order(self):
        """Test that an exception is raised if measurements are returned in an
        order different to how they were queued on the tape"""
        dev = DefaultQubitLegacy(wires=2)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            m = qml.expval(qml.PauliZ(0))
            return qml.expval(qml.PauliX(1)), m

        qn = QNode(func, dev)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="measurements must be returned in the order they are measured",
        ):
            qn(5, 1)

    def test_consistent_measurement_order(self):
        """Test evaluation proceeds as expected if measurements are returned in the
        same order to how they were queued on the tape"""
        dev = DefaultQubitLegacy(wires=2)

        contents = []

        def func(x, y):
            op1 = qml.RX(x, wires=0)
            op2 = qml.RY(y, wires=1)
            op3 = qml.CNOT(wires=[0, 1])
            m1 = qml.expval(qml.PauliZ(0))
            m2 = qml.expval(qml.PauliX(1))
            contents.append(op1)
            contents.append(op2)
            contents.append(op3)
            contents.append(m1)
            contents.append(m2)
            return [m1, m2]

        qn = QNode(func, dev)
        tape = qml.workflow.construct_tape(qn)(5, 1)
        assert tape.operations == contents[0:3]
        assert tape.measurements == contents[3:]

    @pytest.mark.jax
    def test_jit_counts_raises_error(self):
        """Test that returning counts in a quantum function with trainable parameters while
        jitting raises an error."""
        import jax

        dev = DefaultQubitLegacy(wires=2, shots=5)

        def circuit1(param):
            qml.Hadamard(0)
            qml.RX(param, wires=1)
            qml.CNOT([1, 0])
            return qml.counts()

        qn = qml.QNode(circuit1, dev)
        jitted_qnode1 = jax.jit(qn)

        with pytest.raises(
            NotImplementedError, match="The JAX-JIT interface doesn't support qml.counts."
        ):
            jitted_qnode1(0.123)

        # Test with qnode decorator syntax
        @qml.qnode(dev)
        def circuit2(param):
            qml.Hadamard(0)
            qml.RX(param, wires=1)
            qml.CNOT([1, 0])
            return qml.counts()

        jitted_qnode2 = jax.jit(circuit2)

        with pytest.raises(
            NotImplementedError, match="The JAX-JIT interface doesn't support qml.counts."
        ):
            jitted_qnode2(0.123)


def test_decorator(tol):
    """Test that the decorator correctly creates a QNode."""
    dev = DefaultQubitLegacy(wires=2)

    @qnode(dev)
    def func(x, y):
        """My function docstring"""
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    assert isinstance(func, QNode)
    assert func.__doc__ == "My function docstring"

    x = pnp.array(0.12, requires_grad=True)
    y = pnp.array(0.54, requires_grad=True)

    res = func(x, y)
    tape = qml.workflow.construct_tape(func)(x, y)

    assert isinstance(tape, QuantumScript)
    assert len(tape.operations) == 3
    assert len(tape.observables) == 1
    assert tape.num_params == 2

    expected = qml.execute([tape], dev, None)
    assert np.allclose(res, expected, atol=tol, rtol=0)

    # when called, a new quantum tape is constructed
    old_tape = tape
    res2 = func(x, y)
    new_tape = qml.workflow.construct_tape(func)(x, y)

    assert np.allclose(res, res2, atol=tol, rtol=0)
    assert new_tape is not old_tape


class TestIntegration:
    """Integration tests."""

    @pytest.mark.autograd
    def test_correct_number_of_executions_autograd(self):
        """Test that number of executions are tracked in the autograd interface."""

        def func():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = DefaultQubitLegacy(wires=2)
        qn = QNode(func, dev, interface="autograd")

        for _ in range(2):
            qn()

        assert dev.num_executions == 2

        qn2 = QNode(func, dev, interface="autograd")
        for _ in range(3):
            qn2()

        assert dev.num_executions == 5

    @pytest.mark.tf
    @pytest.mark.parametrize("interface", ["tf", "auto"])
    def test_correct_number_of_executions_tf(self, interface):
        """Test that number of executions are tracked in the tf interface."""

        def func():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = DefaultQubitLegacy(wires=2)
        qn = QNode(func, dev, interface=interface)
        for _ in range(2):
            qn()

        assert dev.num_executions == 2

        qn2 = QNode(func, dev, interface=interface)
        for _ in range(3):
            qn2()

        assert dev.num_executions == 5

        # qubit of different interface
        qn3 = QNode(func, dev, interface="autograd")
        qn3()

        assert dev.num_executions == 6

    @pytest.mark.torch
    @pytest.mark.parametrize("interface", ["torch", "auto"])
    def test_correct_number_of_executions_torch(self, interface):
        """Test that number of executions are tracked in the torch interface."""

        def func():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = DefaultQubitLegacy(wires=2)
        qn = QNode(func, dev, interface=interface)
        for _ in range(2):
            qn()

        assert dev.num_executions == 2

        qn2 = QNode(func, dev, interface=interface)
        for _ in range(3):
            qn2()

        assert dev.num_executions == 5

        # qubit of different interface
        qn3 = QNode(func, dev, interface="autograd")
        qn3()

        assert dev.num_executions == 6

    def test_num_exec_caching_device_swap(self):
        """Tests that if we swapped the original device (e.g., when
        diff_method='backprop') then the number of executions recorded is
        correct."""
        dev = DefaultQubitLegacy(wires=2)

        cache = {}

        @qml.qnode(dev, diff_method="backprop", cache=cache)
        def circuit():
            qml.RY(0.345, wires=0)
            return qml.expval(qml.PauliZ(0))

        for _ in range(15):
            circuit()

        # Although we've evaluated the QNode more than once, due to caching,
        # there was one device execution recorded
        assert dev.num_executions == 1
        assert cache

    def test_num_exec_caching_device_swap_two_exec(self):
        """Tests that if we swapped the original device (e.g., when
        diff_method='backprop') then the number of executions recorded is
        correct even with multiple QNode evaluations."""
        dev = DefaultQubitLegacy(wires=2)

        cache = {}

        @qml.qnode(dev, diff_method="backprop", cache=cache)
        def circuit0():
            qml.RY(0.345, wires=0)
            return qml.expval(qml.PauliZ(0))

        for _ in range(15):
            circuit0()

        @qml.qnode(dev, diff_method="backprop", cache=cache)
        def circuit2():
            qml.RZ(0.345, wires=0)
            return qml.expval(qml.PauliZ(0))

        for _ in range(15):
            circuit2()

        # Although we've evaluated the QNode several times, due to caching,
        # there were two device executions recorded
        assert dev.num_executions == 2
        assert cache

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff", "spsa", "hadamard"])
    def test_single_expectation_value_with_argnum_one(self, diff_method, tol):
        """Tests correct output shape and evaluation for a QNode
        with a single expval output where only one parameter is chosen to
        estimate the jacobian.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """
        dev = DefaultQubitLegacy(wires=3)

        x = pnp.array(0.543, requires_grad=True)
        y = pnp.array(-0.654, requires_grad=True)

        @qnode(
            dev, diff_method=diff_method, gradient_kwargs={"argnum": [1]}
        )  # <--- we only choose one trainable parameter
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        res = qml.grad(circuit)(x, y)
        assert len(res) == 2

        expected = (0, np.cos(y) * np.cos(x))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_no_defer_measurements_if_supported(self, mocker):
        """Test that the defer_measurements transform is not used during
        QNode construction if the device supports mid-circuit measurements."""
        dev = DefaultQubitLegacy(wires=3)
        mocker.patch.object(
            qml.devices.LegacyDevice, "_capabilities", {"supports_mid_measure": True}
        )
        spy = mocker.spy(qml.defer_measurements, "_transform")

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            qml.measure(0)
            return qml.expval(qml.PauliZ(1))

        circuit.construct(tuple(), {})

        spy.assert_not_called()
        tape = qml.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 2
        assert isinstance(tape.operations[1], qml.measurements.MidMeasureMP)

    @pytest.mark.parametrize("basis_state", [[1, 0], [0, 1]])
    def test_sampling_with_mcm(self, basis_state, mocker):
        """Tests that a QNode with qml.sample and mid-circuit measurements
        returns the expected results."""
        dev = DefaultQubitLegacy(wires=3, shots=1000)

        first_par = np.pi

        @qml.qnode(dev)
        def cry_qnode(x):
            """QNode where we apply a controlled Y-rotation."""
            qml.BasisState(basis_state, wires=[0, 1])
            qml.CRY(x, wires=[0, 1])
            return qml.sample(qml.PauliZ(1))

        @qml.qnode(dev)
        def conditional_ry_qnode(x):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.BasisState(basis_state, wires=[0, 1])
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)
            return qml.sample(qml.PauliZ(1))

        spy = mocker.spy(qml.defer_measurements, "_transform")
        r1 = cry_qnode(first_par)
        r2 = conditional_ry_qnode(first_par)
        assert np.allclose(r1, r2)
        spy.assert_called()

    @pytest.mark.tf
    @pytest.mark.parametrize("interface", ["tf", "auto"])
    def test_conditional_ops_tensorflow(self, interface):
        """Test conditional operations with TensorFlow."""
        import tensorflow as tf

        dev = DefaultQubitLegacy(wires=3)

        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def cry_qnode(x):
            """QNode where we apply a controlled Y-rotation."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        @qml.defer_measurements
        def conditional_ry_qnode(x):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        x_ = -0.654
        x1 = tf.Variable(x_, dtype=tf.float64)
        x2 = tf.Variable(x_, dtype=tf.float64)

        with tf.GradientTape() as tape1:
            r1 = cry_qnode(x1)

        with tf.GradientTape() as tape2:
            r2 = conditional_ry_qnode(x2)

        assert np.allclose(r1, r2)

        grad1 = tape1.gradient(r1, x1)
        grad2 = tape2.gradient(r2, x2)
        assert np.allclose(grad1, grad2)

    @pytest.mark.torch
    @pytest.mark.parametrize("interface", ["torch", "auto"])
    def test_conditional_ops_torch(self, interface):
        """Test conditional operations with Torch."""
        import torch

        dev = DefaultQubitLegacy(wires=3)

        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def cry_qnode(x):
            """QNode where we apply a controlled Y-rotation."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def conditional_ry_qnode(x):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        x1 = torch.tensor(-0.654, dtype=torch.float64, requires_grad=True)
        x2 = torch.tensor(-0.654, dtype=torch.float64, requires_grad=True)

        r1 = cry_qnode(x1)
        r2 = conditional_ry_qnode(x2)

        assert np.allclose(r1.detach(), r2.detach())

        r1.backward()
        r2.backward()
        assert np.allclose(x1.grad.detach(), x2.grad.detach())

    @pytest.mark.jax
    @pytest.mark.parametrize("jax_interface", ["jax-python", "jax-jit", "auto"])
    def test_conditional_ops_jax(self, jax_interface):
        """Test conditional operations with JAX."""
        import jax

        jnp = jax.numpy
        dev = DefaultQubitLegacy(wires=3)

        @qml.qnode(dev, interface=jax_interface, diff_method="parameter-shift")
        def cry_qnode(x):
            """QNode where we apply a controlled Y-rotation."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(dev, interface=jax_interface, diff_method="parameter-shift")
        def conditional_ry_qnode(x):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        x1 = jnp.array(-0.654)
        x2 = jnp.array(-0.654)

        r1 = cry_qnode(x1)
        r2 = conditional_ry_qnode(x2)

        assert np.allclose(r1, r2)
        assert np.allclose(jax.grad(cry_qnode)(x1), jax.grad(conditional_ry_qnode)(x2))

    def test_qnode_does_not_support_nested_queuing(self):
        """Test that operators in QNodes are not queued to surrounding contexts."""
        dev = DefaultQubitLegacy(wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(0)
            return qml.expval(qml.PauliX(0))

        with qml.queuing.AnnotatedQueue() as q:
            circuit()

        tape = qml.workflow.construct_tape(circuit)()
        assert q.queue == []  # pylint: disable=use-implicit-booleaness-not-comparison
        assert len(tape.operations) == 1


class TestShots:
    """Unit tests for specifying shots per call."""

    # pylint: disable=unexpected-keyword-arg
    def test_specify_shots_per_call_sample(self):
        """Tests that shots can be set per call for a sample return type."""
        dev = DefaultQubitLegacy(wires=1, shots=10)

        @qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        assert len(circuit(0.8)) == 10
        assert len(circuit(0.8, shots=2)) == 2
        assert len(circuit(0.8, shots=3178)) == 3178
        assert len(circuit(0.8)) == 10

    # pylint: disable=unexpected-keyword-arg, protected-access
    def test_specify_shots_per_call_expval(self):
        """Tests that shots can be set per call for an expectation value.
        Note: this test has a vanishingly small probability to fail."""
        dev = DefaultQubitLegacy(wires=1, shots=None)

        @qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        # check that the circuit is analytic
        res1 = [circuit() for _ in range(100)]
        assert np.std(res1) == 0.0
        assert circuit.device._shots is None

        # check that the circuit is temporary non-analytic
        res1 = [circuit(shots=1) for _ in range(100)]
        assert np.std(res1) != 0.0

        # check that the circuit is analytic again
        res1 = [circuit() for _ in range(100)]
        assert np.std(res1) == 0.0
        assert circuit.device._shots is None

    # pylint: disable=unexpected-keyword-arg
    def test_no_shots_per_call_if_user_has_shots_qfunc_kwarg(self):
        """Tests that the per-call shots overwriting is suspended if user
        has a shots keyword argument, but a warning is raised."""

        dev = DefaultQubitLegacy(wires=2, shots=10)

        def circuit(a, shots=0):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=0))

        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):
            circuit = QNode(circuit, dev)

        assert len(circuit(0.8)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8)
        assert tape.operations[0].wires.labels == (0,)

        assert len(circuit(0.8, shots=1)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8, shots=1)
        assert tape.operations[0].wires.labels == (1,)

        assert len(circuit(0.8, shots=0)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8, shots=0)
        assert tape.operations[0].wires.labels == (0,)

    # pylint: disable=unexpected-keyword-arg
    def test_no_shots_per_call_if_user_has_shots_qfunc_arg(self):
        """Tests that the per-call shots overwriting is suspended
        if user has a shots argument, but a warning is raised."""
        dev = DefaultQubitLegacy(wires=[0, 1], shots=10)

        def ansatz0(a, shots):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=0))

        # assert that warning is still raised
        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):
            circuit = QNode(ansatz0, dev)

        assert len(circuit(0.8, 1)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8, 1)
        assert tape.operations[0].wires.labels == (1,)

        dev = DefaultQubitLegacy(wires=2, shots=10)

        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):

            @qnode(dev)
            def ansatz1(a, shots):
                qml.RX(a, wires=shots)
                return qml.sample(qml.PauliZ(wires=0))

        assert len(ansatz1(0.8, shots=0)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8, 0)
        assert tape.operations[0].wires.labels == (0,)

    # pylint: disable=unexpected-keyword-arg
    def test_shots_setting_does_not_mutate_device(self):
        """Tests that per-call shots setting does not change the number of shots in the device."""

        dev = DefaultQubitLegacy(wires=1, shots=3)

        @qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        assert dev.shots == 3
        res = circuit(0.8, shots=2)
        assert len(res) == 2
        assert dev.shots == 3

    def test_warning_finite_shots_dev(self):
        """Tests that a warning is raised when caching is used with finite shots."""
        dev = DefaultQubitLegacy(wires=1, shots=5)

        @qml.qnode(dev, cache={})
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # no warning on the first execution
        circuit(0.3)
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            circuit(0.3)

    # pylint: disable=unexpected-keyword-arg
    def test_warning_finite_shots_override(self):
        """Tests that a warning is raised when caching is used with finite shots."""
        dev = DefaultQubitLegacy(wires=1, shots=5)

        @qml.qnode(dev, cache={})
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # no warning on the first execution
        circuit(0.3)
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            circuit(0.3, shots=5)

    def test_warning_finite_shots_tape(self):
        """Tests that a warning is raised when caching is used with finite shots."""
        dev = DefaultQubitLegacy(wires=1, shots=5)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RZ(0.3, wires=0)
            qml.expval(qml.PauliZ(0))

        tape = QuantumScript.from_queue(q, shots=5)
        # no warning on the first execution
        cache = {}
        qml.execute([tape], dev, None, cache=cache)
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            qml.execute([tape], dev, None, cache=cache)

    def test_no_warning_infinite_shots(self):
        """Tests that no warning is raised when caching is used with infinite shots."""
        dev = DefaultQubitLegacy(wires=1)

        @qml.qnode(dev, cache={})
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Cached execution with finite shots detected")
            circuit(0.3)
            circuit(0.3)

    @pytest.mark.autograd
    def test_no_warning_internal_cache_reuse(self):
        """Tests that no warning is raised when only the internal cache is reused."""
        dev = DefaultQubitLegacy(wires=1, shots=5)

        @qml.qnode(dev, cache=True)
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.probs(wires=0)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Cached execution with finite shots detected")
            qml.jacobian(circuit, argnum=0)(0.3)

    # pylint: disable=unexpected-keyword-arg
    @pytest.mark.parametrize(
        "shots, total_shots, shot_vector",
        [
            (None, None, ()),
            (1, 1, ((1, 1),)),
            (10, 10, ((10, 1),)),
            ([1, 1, 2, 3, 1], 8, ((1, 2), (2, 1), (3, 1), (1, 1))),
        ],
    )
    def test_tape_shots_set_on_call(self, shots, total_shots, shot_vector):
        """test that shots are placed on the tape if they are specified during a call."""
        dev = DefaultQubitLegacy(wires=2, shots=5)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.PauliZ(0))

        qn = QNode(func, dev)

        # No override
        tape = qml.workflow.construct_tape(qn)(0.1, 0.2)
        assert tape.shots.total_shots == 5

        # Override
        tape = qml.workflow.construct_tape(qn)(0.1, 0.2, shots=shots)
        assert tape.shots.total_shots == total_shots
        assert tape.shots.shot_vector == shot_vector

        # Decorator syntax
        @qnode(dev)
        def qn2(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.PauliZ(0))

        # No override
        tape = qml.workflow.construct_tape(qn2)(0.1, 0.2)
        assert tape.shots.total_shots == 5

        # Override
        tape = qml.workflow.construct_tape(qn2)(0.1, 0.2, shots=shots)
        assert tape.shots.total_shots == total_shots
        assert tape.shots.shot_vector == shot_vector


class TestTransformProgramIntegration:
    def test_transform_program_modifies_circuit(self):
        """Test qnode integration with a transform that turns the circuit into just a pauli x."""
        dev = DefaultQubitLegacy(wires=1)

        def null_postprocessing(results):
            return results[0]

        @qml.transform
        def just_pauli_x_out(
            tape: QuantumScript,
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (
                qml.tape.QuantumScript([qml.PauliX(0)], tape.measurements),
            ), null_postprocessing

        @just_pauli_x_out
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.transform_program[0].transform == just_pauli_x_out.transform

        assert qml.math.allclose(circuit(0.1), -1)

        with circuit.device.tracker as tracker:
            circuit(0.1)

        assert tracker.totals["executions"] == 1
        assert tracker.history["resources"][0].gate_types["PauliX"] == 1
        assert tracker.history["resources"][0].gate_types["RX"] == 0

    def tet_transform_program_modifies_results(self):
        """Test integration with a transform that modifies the result output."""

        dev = DefaultQubitLegacy(wires=2)

        @qml.transform
        def pin_result(
            tape: QuantumScript, requested_result
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            def postprocessing(_: qml.typing.ResultBatch) -> qml.typing.Result:
                return requested_result

            return (tape,), postprocessing

        @partial(pin_result, requested_result=3.0)
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.transform_program[0].transform == pin_result.transform
        assert circuit.transform_program[0].kwargs == {"requested_result": 3.0}

        assert qml.math.allclose(circuit(0.1), 3.0)

    def test_transform_order_circuit_processing(self):
        """Test that transforms are applied in the correct order in integration."""

        dev = DefaultQubitLegacy(wires=2)

        def null_postprocessing(results):
            return results[0]

        @qml.transform
        def just_pauli_x_out(
            tape: QuantumScript,
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (
                qml.tape.QuantumScript([qml.PauliX(0)], tape.measurements),
            ), null_postprocessing

        @qml.transform
        def repeat_operations(
            tape: QuantumScript,
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            new_tape = qml.tape.QuantumScript(
                tape.operations + copy.deepcopy(tape.operations), tape.measurements
            )
            return (new_tape,), null_postprocessing

        @repeat_operations
        @just_pauli_x_out
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit1(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        with circuit1.device.tracker as tracker:
            assert qml.math.allclose(circuit1(0.1), 1.0)

        assert tracker.history["resources"][0].gate_types["PauliX"] == 2

        @just_pauli_x_out
        @repeat_operations
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit2(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        with circuit2.device.tracker as tracker:
            assert qml.math.allclose(circuit2(0.1), -1.0)

        assert tracker.history["resources"][0].gate_types["PauliX"] == 1

    def test_transform_order_postprocessing(self):
        """Test that transform postprocessing is called in the right order."""

        dev = DefaultQubitLegacy(wires=2)

        def scale_by_factor(results, factor):
            return results[0] * factor

        def add_shift(results, shift):
            return results[0] + shift

        @qml.transform
        def scale_output(
            tape: QuantumScript, factor
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (tape,), partial(scale_by_factor, factor=factor)

        @qml.transform
        def shift_output(tape: QuantumScript, shift) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (tape,), partial(add_shift, shift=shift)

        @partial(shift_output, shift=1.0)
        @partial(scale_output, factor=2.0)
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit1():
            return qml.expval(qml.PauliZ(0))

        # first add one, then scale by 2.0.  Outer postprocessing transforms are applied first
        assert qml.math.allclose(circuit1(), 4.0)

        @partial(scale_output, factor=2.0)
        @partial(shift_output, shift=1.0)
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit2():
            return qml.expval(qml.PauliZ(0))

        # first scale by 2, then add one. Outer postprocessing transforms are applied first
        assert qml.math.allclose(circuit2(), 3.0)

    def test_scaling_shots_transform(self):
        """Test a transform that scales the number of shots used in an execution."""

        # note that this won't work with the old device interface :(
        dev = qml.devices.DefaultQubit()

        def num_of_shots_from_sample(results):
            return len(results[0])

        @qml.transform
        def use_n_shots(tape: QuantumScript, n) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (
                qml.tape.QuantumScript(tape.operations, tape.measurements, shots=n),
            ), num_of_shots_from_sample

        @partial(use_n_shots, n=100)
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit():
            return qml.sample(wires=0)

        assert circuit() == 100


# pylint: disable=unused-argument
class CustomDevice(qml.devices.Device):
    """A null device that just returns 0."""

    def __repr__(self):
        return "CustomDevice"

    def execute(self, circuits, execution_config=None):
        return (0,)


class TestTapeExpansion:
    """Test that tape expansion within the QNode works correctly"""

    @pytest.mark.parametrize(
        "diff_method,mode",
        [("parameter-shift", False), ("adjoint", True), ("adjoint", False)],
    )
    def test_device_expansion(self, diff_method, mode, mocker):
        """Test expansion of an unsupported operation on the device"""

        dev = DefaultQubitLegacy(wires=1)

        # pylint: disable=too-few-public-methods
        class UnsupportedOp(qml.operation.Operation):
            """custom unsupported op."""

            num_wires = 1

            def decomposition(self):
                return [qml.RX(3 * self.data[0], wires=self.wires)]

        @qnode(dev, diff_method=diff_method, grad_on_execution=mode)
        def circuit(x):
            UnsupportedOp(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        if diff_method == "adjoint" and mode:
            spy = mocker.spy(circuit.device, "execute_and_compute_derivatives")
        else:
            spy = mocker.spy(circuit.device, "execute")

        x = pnp.array(0.5)
        circuit(x)

        tape = spy.call_args[0][0][0]
        assert len(tape.operations) == 1
        assert tape.operations[0].name == "RX"
        assert np.allclose(tape.operations[0].parameters, 3 * x)

    @pytest.mark.autograd
    def test_no_gradient_expansion(self, mocker):
        """Test that an unsupported operation with defined gradient recipe is
        not expanded"""
        dev = DefaultQubitLegacy(wires=1)

        # pylint: disable=too-few-public-methods
        class UnsupportedOp(qml.operation.Operation):
            """custom unsupported op."""

            num_wires = 1

            grad_method = "A"
            grad_recipe = ([[3 / 2, 1, np.pi / 6], [-3 / 2, 1, -np.pi / 6]],)

            def decomposition(self):
                return [qml.RX(3 * self.data[0], wires=self.wires)]

        @qnode(dev, interface="autograd", diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            UnsupportedOp(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = pnp.array(0.5, requires_grad=True)
        qml.grad(circuit)(x)

        # check second derivative
        assert np.allclose(qml.grad(qml.grad(circuit))(x), -9 * np.cos(3 * x))

    @pytest.mark.autograd
    def test_gradient_expansion(self, mocker):
        """Test that a *supported* operation with no gradient recipe is
        expanded when applying the gradient transform, but not for execution."""
        dev = DefaultQubitLegacy(wires=1)

        # pylint: disable=too-few-public-methods
        class PhaseShift(qml.PhaseShift):
            """custom phase shift."""

            grad_method = None

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

        @qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            return qml.expval(qml.PauliX(0))

        x = pnp.array(0.5, requires_grad=True)
        circuit(x)

        res = qml.grad(circuit)(x)

        assert np.allclose(res, -3 * np.sin(3 * x))

        # test second order derivatives
        res = qml.grad(qml.grad(circuit))(x)
        assert np.allclose(res, -9 * np.cos(3 * x))

    def test_hamiltonian_expansion_analytic(self):
        """Test result if there are non-commuting groups and the number of shots is None"""
        dev = DefaultQubitLegacy(wires=3, shots=None)

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        c = np.array([-0.6543, 0.24, 0.54])
        H = qml.Hamiltonian(c, obs)
        H.compute_grouping()

        assert len(H.grouping_indices) == 2

        @qnode(dev)
        def circuit():
            return qml.expval(H)

        res = circuit()
        assert np.allclose(res, c[2], atol=0.1)

    def test_hamiltonian_expansion_finite_shots(self, mocker):
        """Test that the Hamiltonian is expanded if there
        are non-commuting groups and the number of shots is finite"""
        dev = DefaultQubitLegacy(wires=3, shots=50000)

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        c = np.array([-0.6543, 0.24, 0.54])
        H = qml.Hamiltonian(c, obs)
        H.compute_grouping()

        assert len(H.grouping_indices) == 2

        @qnode(dev)
        def circuit():
            return qml.expval(H)

        spy = mocker.spy(qml.transforms, "split_non_commuting")
        res = circuit()
        assert np.allclose(res, c[2], atol=0.3)

        spy.assert_called()
        tapes, _ = spy.spy_return

        assert len(tapes) == 2

    @pytest.mark.parametrize("grouping", [True, False])
    def test_multiple_hamiltonian_expansion_finite_shots(self, grouping):
        """Test that multiple Hamiltonians works correctly (sum_expand should be used)"""

        dev = DefaultQubitLegacy(wires=3, shots=50000)

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        c = np.array([-0.6543, 0.24, 0.54])
        H = qml.Hamiltonian(c, obs)

        if grouping:
            H.compute_grouping()
            assert len(H.grouping_indices) == 2

        @qnode(dev)
        def circuit():
            return qml.expval(H), qml.expval(H)

        res = circuit()
        assert qml.math.allclose(res, [0.54, 0.54], atol=0.05)
        assert res[0] == res[1]

    def test_expansion_multiple_qwc_observables(self, mocker):
        """Test that the QNode correctly expands tapes that return
        multiple measurements of commuting observables"""
        dev = DefaultQubitLegacy(wires=2)
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliY(1)]

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return [qml.expval(o) for o in obs]

        spy_expand = mocker.spy(circuit.device.target_device, "expand_fn")
        params = [0.1, 0.2]
        res = circuit(*params)

        tape = spy_expand.spy_return
        rotations, observables = qml.pauli.diagonalize_qwc_pauli_words(obs)

        assert tape.observables[0].name == observables[0].name
        assert tape.observables[1].name == observables[1].name

        assert tape.operations[-2].name == rotations[0].name
        assert tape.operations[-2].parameters == rotations[0].parameters
        assert tape.operations[-1].name == rotations[1].name
        assert tape.operations[-1].parameters == rotations[1].parameters

        # check output value is consistent with a Hamiltonian expectation
        coeffs = np.array([1.0, 1.0])
        H = qml.Hamiltonian(coeffs, obs)

        @qml.qnode(dev)
        def circuit2(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(H)

        res_H = circuit2(*params)
        assert np.allclose(coeffs @ res, res_H)

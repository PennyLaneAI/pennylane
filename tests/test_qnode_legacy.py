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
        return {"passthru_devices": {"autograd": "default.mixed"}}

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


def test_backprop_switching_deprecation():
    """Test that a PennyLaneDeprecationWarning is raised when a device is subtituted
    for a different backprop device.
    """

    with pytest.warns(qml.PennyLaneDeprecationWarning):

        @qml.qnode(DummyDevice(shots=None), interface="autograd")
        def circ(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        circ(pnp.array(3))


# pylint: disable=too-many-public-methods
class TestValidation:
    """Tests for QNode creation and validation"""

    def test_invalid_interface(self):
        """Test that an exception is raised for an invalid interface"""
        dev = qml.device("default.qubit.legacy", wires=1)
        test_interface = "something"
        expected_error = rf"Unknown interface {test_interface}\. Interface must be one of"

        with pytest.raises(qml.QuantumFunctionError, match=expected_error):
            QNode(dummyfunc, dev, interface="something")

    def test_changing_invalid_interface(self):
        """Test that an exception is raised for an invalid interface
        on a pre-existing QNode"""
        dev = qml.device("default.qubit.legacy", wires=1)
        test_interface = "something"

        @qnode(dev)
        def circuit(x):
            """a circuit."""
            qml.RX(x, wires=0)
            return qml.probs(wires=0)

        expected_error = rf"Unknown interface {test_interface}\. Interface must be one of"

        with pytest.raises(qml.QuantumFunctionError, match=expected_error):
            circuit.interface = test_interface

    def test_invalid_device(self):
        """Test that an exception is raised for an invalid device"""
        with pytest.raises(qml.QuantumFunctionError, match="Invalid device"):
            QNode(dummyfunc, None)

    def test_best_method_wraps_legacy_device_correctly(self, mocker):
        dev_legacy = qml.devices.DefaultQubitLegacy(wires=2)

        spy = mocker.spy(qml.devices.LegacyDeviceFacade, "__init__")

        QNode.get_best_method(dev_legacy, "some_interface")

        spy.assert_called_once()

    # pylint: disable=protected-access
    @pytest.mark.autograd
    def test_best_method_is_device(self, monkeypatch):
        """Test that the method for determining the best diff method
        for a given device and interface returns the device"""

        dev = qml.device("default.qubit.legacy", wires=1)

        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)

        # basic check if the device provides a Jacobian
        res = QNode.get_best_method(dev, "another_interface")
        assert res == ("device", {}, dev)

        # device is returned even if backpropagation is possible
        res = QNode.get_best_method(dev, "some_interface")
        assert res == ("device", {}, dev)

    # pylint: disable=protected-access
    @pytest.mark.parametrize("interface", ["jax", "tensorflow", "torch", "autograd"])
    def test_best_method_is_backprop(self, interface):
        """Test that the method for determining the best diff method
        for a given device and interface returns backpropagation"""
        dev = qml.device("default.qubit.legacy", wires=1)

        # backprop is returned when the interface is an allowed interface for the device and Jacobian is not provided
        res = QNode.get_best_method(dev, interface)
        assert res == ("backprop", {}, dev)

    # pylint: disable=protected-access
    def test_best_method_is_param_shift(self):
        """Test that the method for determining the best diff method
        for a given device and interface returns the parameter shift rule"""
        dev = qml.device("default.qubit.legacy", wires=1)

        tape = qml.tape.QuantumScript([], [], shots=50)
        res = QNode.get_best_method(dev, None, tape=tape)
        assert res == (qml.gradients.param_shift, {}, dev)

    # pylint: disable=protected-access
    @pytest.mark.xfail(reason="No longer possible thanks to the new Legacy Facade")
    def test_best_method_is_finite_diff(self, monkeypatch):
        """Test that the method for determining the best diff method
        for a given device and interface returns finite differences"""

        def capabilities(cls):
            capabilities = cls._capabilities
            capabilities.update(model="None")
            return capabilities

        # finite differences is the fallback when we know nothing about the device
        monkeypatch.setattr(qml.devices.DefaultQubitLegacy, "capabilities", capabilities)

        dev = qml.device("default.qubit.legacy", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)

        res = QNode.get_best_method(dev, "another_interface")
        assert res == (qml.gradients.finite_diff, {}, dev)

    # pylint: disable=protected-access
    def test_best_method_str_is_device(self, monkeypatch):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'device'"""
        dev = qml.device("default.qubit.legacy", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)

        # basic check if the device provides a Jacobian
        res = QNode.best_method_str(dev, "another_interface")
        assert res == "device"

        # device is returned even if backpropagation is possible
        res = QNode.best_method_str(dev, "some_interface")
        assert res == "device"

    # pylint: disable=protected-access
    def test_best_method_str_is_backprop(self, monkeypatch):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'backprop'"""
        dev = qml.device("default.qubit.legacy", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)

        # backprop is returned when the interfaces match and Jacobian is not provided
        res = QNode.best_method_str(dev, "some_interface")
        assert res == "backprop"

    def test_best_method_str_wraps_legacy_device_correctly(self, mocker):
        dev_legacy = qml.devices.DefaultQubitLegacy(wires=2)

        spy = mocker.spy(qml.devices.LegacyDeviceFacade, "__init__")

        QNode.best_method_str(dev_legacy, "some_interface")

        spy.assert_called_once()

    # pylint: disable=protected-access
    def test_best_method_str_is_param_shift(self):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'parameter-shift'"""
        dev = qml.device("default.qubit.legacy", wires=1, shots=50)

        # parameter shift is returned when Jacobian is not provided and
        # the backprop interfaces do not match
        res = QNode.best_method_str(dev, "another_interface")
        assert res == "parameter-shift"

    # pylint: disable=protected-access
    def test_best_method_str_is_finite_diff(self, mocker):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'finite-diff'"""
        dev = qml.device("default.qubit.legacy", wires=1)

        mocker.patch.object(QNode, "get_best_method", return_value=[qml.gradients.finite_diff])

        res = QNode.best_method_str(dev, "another_interface")

        assert res == "finite-diff"

    # pylint: disable=protected-access
    def test_diff_method(self, mocker):
        """Test that a user-supplied diff method correctly returns the right
        diff method."""
        dev = qml.device("default.qubit.legacy", wires=1)

        mock_best = mocker.patch("pennylane.QNode.get_best_method")
        mock_best.return_value = ("best", {}, dev)

        qn = QNode(dummyfunc, dev, diff_method="best")
        assert qn.diff_method == "best"
        assert qn.gradient_fn == "backprop"

        qn = QNode(dummyfunc, dev, interface="autograd", diff_method="best")
        assert qn.diff_method == "best"
        assert qn.gradient_fn == "backprop"

        qn = QNode(dummyfunc, dev, diff_method="backprop")
        assert qn.diff_method == "backprop"
        assert qn.gradient_fn == "backprop"

        qn = QNode(dummyfunc, dev, interface="autograd", diff_method="backprop")
        assert qn.diff_method == "backprop"
        assert qn.gradient_fn == "backprop"

        qn = QNode(dummyfunc, DeviceDerivatives(wires=1), diff_method="device")
        assert qn.diff_method == "device"
        assert qn.gradient_fn == "device"

        qn = QNode(
            dummyfunc, DeviceDerivatives(wires=1), interface="autograd", diff_method="device"
        )
        assert qn.diff_method == "device"
        assert qn.gradient_fn == "device"

        qn = QNode(dummyfunc, dev, diff_method="finite-diff")
        assert qn.diff_method == "finite-diff"
        assert qn.gradient_fn is qml.gradients.finite_diff

        qn = QNode(dummyfunc, dev, interface="autograd", diff_method="finite-diff")
        assert qn.diff_method == "finite-diff"
        assert qn.gradient_fn is qml.gradients.finite_diff

        qn = QNode(dummyfunc, dev, diff_method="spsa")
        assert qn.diff_method == "spsa"
        assert qn.gradient_fn is qml.gradients.spsa_grad

        qn = QNode(dummyfunc, dev, interface="autograd", diff_method="hadamard")
        assert qn.diff_method == "hadamard"
        assert qn.gradient_fn is qml.gradients.hadamard_grad

        qn = QNode(dummyfunc, dev, diff_method="parameter-shift")
        assert qn.diff_method == "parameter-shift"
        assert qn.gradient_fn is qml.gradients.param_shift

        qn = QNode(dummyfunc, dev, interface="autograd", diff_method="parameter-shift")
        assert qn.diff_method == "parameter-shift"
        assert qn.gradient_fn is qml.gradients.param_shift
        # check that get_best_method was only ever called once

    @pytest.mark.autograd
    def test_gradient_transform(self, mocker):
        """Test passing a gradient transform directly to a QNode"""
        dev = qml.device("default.qubit.legacy", wires=1)
        spy = mocker.spy(qml.gradients.finite_difference, "finite_diff_coeffs")

        @qnode(dev, diff_method=qml.gradients.finite_diff)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qml.grad(circuit)(pnp.array(0.5, requires_grad=True))
        assert circuit.gradient_fn is qml.gradients.finite_diff
        spy.assert_called()

    def test_unknown_diff_method_string(self):
        """Test that an exception is raised for an unknown differentiation method string"""
        dev = qml.device("default.qubit.legacy", wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="Differentiation method hello not recognized"
        ):
            QNode(dummyfunc, dev, interface="autograd", diff_method="hello")

    def test_unknown_diff_method_type(self):
        """Test that an exception is raised for an unknown differentiation method type"""
        dev = qml.device("default.qubit.legacy", wires=1)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Differentiation method 5 must be a gradient transform or a string",
        ):
            QNode(dummyfunc, dev, interface="autograd", diff_method=5)

    def test_adjoint_finite_shots(self):
        """Tests that QuantumFunctionError is raised with the adjoint differentiation method
        on QNode construction when the device has finite shots
        """

        dev = qml.device("default.qubit.legacy", wires=1, shots=1)

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
        dev = qml.device("default.qubit.legacy", wires=2, shots=None)

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
        dev = qml.device("default.qubit.legacy", wires=1)

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

    @pytest.mark.autograd
    def test_diff_method_none(self, tol):
        """Test that diff_method=None creates a QNode with no interface, and no
        device swapping."""
        dev = qml.device("default.qubit.legacy", wires=1)

        @qnode(dev, diff_method=None)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.interface is None
        assert circuit.gradient_fn is None
        assert circuit.device is dev

        # QNode can still be executed
        assert np.allclose(circuit(0.5), np.cos(0.5), atol=tol, rtol=0)

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            grad = qml.grad(circuit)(0.5)

        assert np.allclose(grad, 0)

    # pylint: disable=unused-variable
    def test_unrecognized_kwargs_raise_warning(self):
        """Test that passing gradient_kwargs not included in qml.gradients.SUPPORTED_GRADIENT_KWARGS raises warning"""
        dev = qml.device("default.qubit.legacy", wires=2)

        with warnings.catch_warnings(record=True) as w:

            @qml.qnode(dev, random_kwarg=qml.gradients.finite_diff)
            def circuit(params):
                qml.RX(params[0], wires=0)
                return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))

            assert len(w) == 1
            assert "not included in the list of standard qnode gradient kwargs" in str(w[0].message)

    # pylint: disable=unused-variable
    def test_incorrect_diff_method_kwargs_raise_warning(self):
        """Tests that using one of the incorrect kwargs previously used in some examples in PennyLane
        (grad_method, gradient_fn) to set the qnode diff_method raises a warning"""
        dev = qml.device("default.qubit.legacy", wires=2)

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
        dev = qml.device("default.qubit.legacy", wires=1)

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
        dev = qml.device("default.qubit.legacy", wires=1)

        @qml.qnode(dev, interface="autograd")
        def circuit(params):
            qml.RX(params, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit(qml.numpy.array(0.1, requires_grad=True))

    def test_not_giving_mode_kwarg_does_not_raise_warning(self):
        """Test that not providing a value for mode does not raise a warning
        except for the deprecation warning."""
        with warnings.catch_warnings(record=True) as record:
            qml.QNode(lambda f: f, qml.device("default.qubit.legacy", wires=1))

        assert len(record) == 1
        assert record[0].category == qml.PennyLaneDeprecationWarning


class TestTapeConstruction:
    """Tests for the tape construction"""

    def test_basic_tape_construction(self, tol):
        """Test that a quantum tape is properly constructed"""
        dev = qml.device("default.qubit.legacy", wires=2)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        qn = QNode(func, dev)

        x = pnp.array(0.12, requires_grad=True)
        y = pnp.array(0.54, requires_grad=True)

        res = qn(x, y)

        assert isinstance(qn.qtape, QuantumScript)
        assert len(qn.qtape.operations) == 3
        assert len(qn.qtape.observables) == 1
        assert qn.qtape.num_params == 2
        assert qn.qtape.shots.total_shots is None

        expected = qml.execute([qn.tape], dev, None)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # when called, a new quantum tape is constructed
        old_tape = qn.qtape
        res2 = qn(x, y)

        assert np.allclose(res, res2, atol=tol, rtol=0)
        assert qn.qtape is not old_tape

    def test_jacobian(self):
        """Test the jacobian computation"""
        dev = qml.device("default.qubit.legacy", wires=2)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        qn = QNode(
            func, dev, interface="autograd", diff_method="finite-diff", h=1e-8, approx_order=2
        )
        assert qn.gradient_kwargs["h"] == 1e-8
        assert qn.gradient_kwargs["approx_order"] == 2

        jac = qn.gradient_fn(qn)(
            pnp.array(0.45, requires_grad=True), pnp.array(0.1, requires_grad=True)
        )
        assert isinstance(jac, tuple) and len(jac) == 2
        assert len(jac[0]) == 2
        assert len(jac[1]) == 2

    def test_returning_non_measurements(self):
        """Test that an exception is raised if a non-measurement
        is returned from the QNode."""
        dev = qml.device("default.qubit.legacy", wires=2)

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
        dev = qml.device("default.qubit.legacy", wires=2)

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
        dev = qml.device("default.qubit.legacy", wires=2)

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
        qn(5, 1)  # evaluate the QNode
        assert qn.qtape.operations == contents[0:3]
        assert qn.qtape.measurements == contents[3:]

    def test_operator_all_wires(self, monkeypatch, tol):
        """Test that an operator that must act on all wires
        does, or raises an error."""
        monkeypatch.setattr(qml.RX, "num_wires", qml.operation.AllWires)

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit.legacy", wires=2)
        qn = QNode(circuit, dev)

        with pytest.raises(qml.QuantumFunctionError, match="Operator RX must act on all wires"):
            qn(0.5)

        dev = qml.device("default.qubit.legacy", wires=1)
        qn = QNode(circuit, dev)
        assert np.allclose(qn(0.5), np.cos(0.5), atol=tol, rtol=0)

    def test_all_wires_new_device(self):
        """Test that an operator must act on all tape wires with the new device API."""

        def circuit1(x):
            qml.GlobalPhase(x, wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        dev = qml.devices.DefaultQubit()  # TODO: add wires, change comment below
        qn = QNode(circuit1, dev)

        # fails when GlobalPhase is a strict subset of all tape wires
        with pytest.raises(qml.QuantumFunctionError, match="GlobalPhase must act on all wires"):
            qn(0.5)

        @qml.qnode(dev)
        def circuit2(x):
            qml.GlobalPhase(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        # passes here, does not care for device.wires because it has none
        assert circuit2(0.5) == 1

    @pytest.mark.jax
    def test_jit_counts_raises_error(self):
        """Test that returning counts in a quantum function with trainable parameters while
        jitting raises an error."""
        import jax

        dev = qml.device("default.qubit.legacy", wires=2, shots=5)

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
    dev = qml.device("default.qubit.legacy", wires=2)

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

    assert isinstance(func.qtape, QuantumScript)
    assert len(func.qtape.operations) == 3
    assert len(func.qtape.observables) == 1
    assert func.qtape.num_params == 2

    expected = qml.execute([func.tape], dev, None)
    assert np.allclose(res, expected, atol=tol, rtol=0)

    # when called, a new quantum tape is constructed
    old_tape = func.qtape
    res2 = func(x, y)

    assert np.allclose(res, res2, atol=tol, rtol=0)
    assert func.qtape is not old_tape


class TestIntegration:
    """Integration tests."""

    @pytest.mark.autograd
    def test_correct_number_of_executions_autograd(self):
        """Test that number of executions are tracked in the autograd interface."""

        def func():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit.legacy", wires=2)
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

        dev = qml.device("default.qubit.legacy", wires=2)
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

        dev = qml.device("default.qubit.legacy", wires=2)
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
        dev = qml.device("default.qubit.legacy", wires=2)

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
        dev = qml.device("default.qubit.legacy", wires=2)

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
        dev = qml.device("default.qubit.legacy", wires=3)

        x = pnp.array(0.543, requires_grad=True)
        y = pnp.array(-0.654, requires_grad=True)

        @qnode(
            dev, diff_method=diff_method, argnum=[1]
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
        dev = qml.device("default.qubit.legacy", wires=3)
        mocker.patch.object(qml.Device, "_capabilities", {"supports_mid_measure": True})
        spy = mocker.spy(qml.defer_measurements, "_transform")

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            qml.measure(0)
            return qml.expval(qml.PauliZ(1))

        circuit.construct(tuple(), {})

        spy.assert_not_called()
        assert len(circuit.tape.operations) == 2
        assert isinstance(circuit.tape.operations[1], qml.measurements.MidMeasureMP)

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.qubit.legacy"])
    @pytest.mark.parametrize("first_par", np.linspace(0.15, np.pi - 0.3, 3))
    @pytest.mark.parametrize("sec_par", np.linspace(0.15, np.pi - 0.3, 3))
    @pytest.mark.parametrize(
        "return_type", [qml.expval(qml.PauliZ(1)), qml.var(qml.PauliZ(1)), qml.probs(wires=[1])]
    )
    @pytest.mark.parametrize(
        "mv_return, mv_res",
        [
            (qml.expval, lambda x: np.sin(x / 2) ** 2),
            (qml.var, lambda x: np.sin(x / 2) ** 2 - np.sin(x / 2) ** 4),
            (qml.probs, lambda x: [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2]),
        ],
    )
    def test_defer_meas_if_mcm_unsupported(
        self, dev_name, first_par, sec_par, return_type, mv_return, mv_res, mocker
    ):  # pylint: disable=too-many-arguments
        """Tests that the transform using the deferred measurement principle is
        applied if the device doesn't support mid-circuit measurements
        natively."""

        dev = qml.device(dev_name, wires=3)

        @qml.qnode(dev)
        def cry_qnode(x, y):
            """QNode where we apply a controlled Y-rotation."""
            qml.Hadamard(1)
            qml.RY(x, wires=0)
            qml.CRY(y, wires=[0, 1])
            return qml.apply(return_type)

        @qml.qnode(dev)
        def conditional_ry_qnode(x, y):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.Hadamard(1)
            qml.RY(x, wires=0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(y, wires=1)
            return qml.apply(return_type), mv_return(op=m_0)

        spy = mocker.spy(qml.defer_measurements, "_transform")
        r1 = cry_qnode(first_par, sec_par)
        r2 = conditional_ry_qnode(first_par, sec_par)

        assert np.allclose(r1, r2[0])
        assert np.allclose(r2[1], mv_res(first_par))
        assert spy.call_count == 3 if dev.name == "defaut.qubit" else 1

    @pytest.mark.parametrize("basis_state", [[1, 0], [0, 1]])
    def test_sampling_with_mcm(self, basis_state, mocker):
        """Tests that a QNode with qml.sample and mid-circuit measurements
        returns the expected results."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=1000)

        first_par = np.pi

        @qml.qnode(dev)
        def cry_qnode(x):
            """QNode where we apply a controlled Y-rotation."""
            qml.BasisStatePreparation(basis_state, wires=[0, 1])
            qml.CRY(x, wires=[0, 1])
            return qml.sample(qml.PauliZ(1))

        @qml.qnode(dev)
        def conditional_ry_qnode(x):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.BasisStatePreparation(basis_state, wires=[0, 1])
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

        dev = qml.device("default.qubit.legacy", wires=3)

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

        dev = qml.device("default.qubit.legacy", wires=3)

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
        dev = qml.device("default.qubit.legacy", wires=3)

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
        dev = qml.device("default.qubit.legacy", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(0)
            return qml.expval(qml.PauliX(0))

        with qml.queuing.AnnotatedQueue() as q:
            circuit()

        assert q.queue == []  # pylint: disable=use-implicit-booleaness-not-comparison
        assert len(circuit.tape.operations) == 1


class TestShots:
    """Unit tests for specifying shots per call."""

    # pylint: disable=unexpected-keyword-arg
    def test_specify_shots_per_call_sample(self):
        """Tests that shots can be set per call for a sample return type."""
        dev = qml.device("default.qubit.legacy", wires=1, shots=10)

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
        dev = qml.device("default.qubit.legacy", wires=1, shots=None)

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

        dev = qml.device("default.qubit.legacy", wires=2, shots=10)

        def circuit(a, shots=0):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=0))

        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):
            circuit = QNode(circuit, dev)

        assert len(circuit(0.8)) == 10
        assert circuit.qtape.operations[0].wires.labels == (0,)

        assert len(circuit(0.8, shots=1)) == 10
        assert circuit.qtape.operations[0].wires.labels == (1,)

        assert len(circuit(0.8, shots=0)) == 10
        assert circuit.qtape.operations[0].wires.labels == (0,)

    # pylint: disable=unexpected-keyword-arg
    def test_no_shots_per_call_if_user_has_shots_qfunc_arg(self):
        """Tests that the per-call shots overwriting is suspended
        if user has a shots argument, but a warning is raised."""
        dev = qml.device("default.qubit.legacy", wires=[0, 1], shots=10)

        def ansatz0(a, shots):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=0))

        # assert that warning is still raised
        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):
            circuit = QNode(ansatz0, dev)

        assert len(circuit(0.8, 1)) == 10
        assert circuit.qtape.operations[0].wires.labels == (1,)

        dev = qml.device("default.qubit.legacy", wires=2, shots=10)

        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):

            @qnode(dev)
            def ansatz1(a, shots):
                qml.RX(a, wires=shots)
                return qml.sample(qml.PauliZ(wires=0))

        assert len(ansatz1(0.8, shots=0)) == 10
        assert ansatz1.qtape.operations[0].wires.labels == (0,)

    # pylint: disable=unexpected-keyword-arg
    def test_shots_setting_does_not_mutate_device(self):
        """Tests that per-call shots setting does not change the number of shots in the device."""

        dev = qml.device("default.qubit.legacy", wires=1, shots=3)

        @qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        assert dev.shots == qml.measurements.Shots(3)
        res = circuit(0.8, shots=2)
        assert len(res) == 2
        assert dev.shots == qml.measurements.Shots(3)

    def test_warning_finite_shots_dev(self):
        """Tests that a warning is raised when caching is used with finite shots."""
        dev = qml.device("default.qubit.legacy", wires=1, shots=5)

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
        dev = qml.device("default.qubit.legacy", wires=1, shots=5)

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
        dev = qml.device("default.qubit.legacy", wires=1, shots=5)

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
        dev = qml.device("default.qubit.legacy", wires=1)

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
        dev = qml.device("default.qubit.legacy", wires=1, shots=5)

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
        dev = qml.device("default.qubit.legacy", wires=2, shots=5)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.PauliZ(0))

        qn = QNode(func, dev)

        # No override
        _ = qn(0.1, 0.2)
        assert qn.tape.shots.total_shots == 5

        # Override
        _ = qn(0.1, 0.2, shots=shots)
        assert qn.tape.shots.total_shots == total_shots
        assert qn.tape.shots.shot_vector == shot_vector

        # Decorator syntax
        @qnode(dev)
        def qn2(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.PauliZ(0))

        # No override
        _ = qn2(0.1, 0.2)
        assert qn2.tape.shots.total_shots == 5

        # Override
        _ = qn2(0.1, 0.2, shots=shots)
        assert qn2.tape.shots.total_shots == total_shots
        assert qn2.tape.shots.shot_vector == shot_vector


class TestTransformProgramIntegration:
    def test_transform_program_modifies_circuit(self):
        """Test qnode integration with a transform that turns the circuit into just a pauli x."""
        dev = qml.device("default.qubit.legacy", wires=1)

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

        dev = qml.device("default.qubit.legacy", wires=2)

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

        dev = qml.device("default.qubit.legacy", wires=2)

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

        dev = qml.device("default.qubit.legacy", wires=2)

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
        dev = qml.device("default.qubit.legacy", wires=1)

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

        x = np.array(0.5)
        circuit(x)

        tape = spy.call_args[0][0][0]
        assert len(tape.operations) == 1
        assert tape.operations[0].name == "RX"
        assert np.allclose(tape.operations[0].parameters, 3 * x)

    @pytest.mark.autograd
    def test_no_gradient_expansion(self, mocker):
        """Test that an unsupported operation with defined gradient recipe is
        not expanded"""
        dev = qml.device("default.qubit.legacy", wires=1)

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
        dev = qml.device("default.qubit.legacy", wires=1)

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
        dev = qml.device("default.qubit.legacy", wires=3, shots=None)

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
        dev = qml.device("default.qubit.legacy", wires=3, shots=50000)

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

    @pytest.mark.usefixtures("new_opmath_only")
    @pytest.mark.parametrize("grouping", [True, False])
    def test_multiple_hamiltonian_expansion_finite_shots(self, grouping):
        """Test that multiple Hamiltonians works correctly (sum_expand should be used)"""

        dev = qml.device("default.qubit.legacy", wires=3, shots=50000)

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

    def test_device_expansion_strategy(self, mocker):
        """Test that the device expansion strategy performs the device
        decomposition at construction time, and not at execution time"""
        dev = qml.device("default.qubit.legacy", wires=2)
        x = pnp.array(0.5, requires_grad=True)

        with pytest.warns(
            qml.PennyLaneDeprecationWarning,
            match="'expansion_strategy' attribute is deprecated",
        ):

            @qnode(dev, diff_method="parameter-shift", expansion_strategy="device")
            def circuit(x):
                qml.SingleExcitation(x, wires=[0, 1])
                return qml.expval(qml.PauliX(0))

        assert circuit.expansion_strategy == "device"
        assert circuit.execute_kwargs["expand_fn"] is None

        spy_expand = mocker.spy(circuit.device.target_device, "expand_fn")

        circuit.construct([x], {})
        assert len(circuit.tape.operations) > 0
        spy_expand.assert_called_once()
        circuit(x)

        assert len(spy_expand.call_args_list) == 3

        qml.grad(circuit)(x)
        assert len(spy_expand.call_args_list) == 9

    def test_expansion_multiple_qwc_observables(self, mocker):
        """Test that the QNode correctly expands tapes that return
        multiple measurements of commuting observables"""
        dev = qml.device("default.qubit.legacy", wires=2)
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

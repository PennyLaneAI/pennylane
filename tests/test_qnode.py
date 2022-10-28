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
import warnings
from collections import defaultdict

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane import QNode
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane.tape import QuantumTape


def dummyfunc():
    return None


class TestValidation:
    """Tests for QNode creation and validation"""

    def test_invalid_interface(self):
        """Test that an exception is raised for an invalid interface"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"
        expected_error = rf"Unknown interface {test_interface}\. Interface must be one of"

        with pytest.raises(qml.QuantumFunctionError, match=expected_error):
            QNode(dummyfunc, dev, interface="something")

    def test_changing_invalid_interface(self):
        """Test that an exception is raised for an invalid interface
        on a pre-existing QNode"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"

        @qnode(dev)
        def circuit(x):
            qml.RX(wires=0)
            return qml.probs(wires=0)

        expected_error = rf"Unknown interface {test_interface}\. Interface must be one of"

        with pytest.raises(qml.QuantumFunctionError, match=expected_error):
            circuit.interface = test_interface

    @pytest.mark.torch
    def test_valid_interface(self):
        """Test that changing to a valid interface works as expected, and the
        diff method is updated as required."""
        import torch

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, interface="autograd", diff_method="best")
        def circuit(x):
            qml.RX(wires=0)
            return qml.probs(wires=0)

        assert circuit.device.short_name == "default.qubit.autograd"
        assert circuit.gradient_fn == "backprop"

        circuit.interface = "torch"
        assert circuit.device.short_name == "default.qubit.torch"
        assert circuit.gradient_fn == "backprop"

    def test_invalid_device(self):
        """Test that an exception is raised for an invalid device"""
        with pytest.raises(qml.QuantumFunctionError, match="Invalid device"):
            QNode(dummyfunc, None)

    def test_validate_device_method(self, monkeypatch):
        """Test that the method for validating the device diff method
        tape works as expected"""
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="does not provide a native method for computing the jacobian",
        ):
            QNode._validate_device_method(dev)

        monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)
        method, diff_options, device = QNode._validate_device_method(dev)

        assert method == "device"
        assert device is dev

    def test_validate_backprop_method_invalid_device(self):
        """Test that the method for validating the backprop diff method
        tape raises an exception if the device does not support backprop."""
        dev = qml.device("default.gaussian", wires=1)

        with pytest.raises(qml.QuantumFunctionError, match="does not support native computations"):
            QNode._validate_backprop_method(dev, None)

    def test_validate_backprop_method_invalid_interface(self, monkeypatch):
        """Test that the method for validating the backprop diff method
        tape raises an exception if the wrong interface is provided"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"

        monkeypatch.setitem(dev._capabilities, "passthru_interface", test_interface)

        with pytest.raises(qml.QuantumFunctionError, match=f"when using the {test_interface}"):
            QNode._validate_backprop_method(dev, None)

    def test_validate_backprop_method(self, monkeypatch):
        """Test that the method for validating the backprop diff method
        tape works as expected"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"
        monkeypatch.setitem(dev._capabilities, "passthru_interface", test_interface)

        method, diff_options, device = QNode._validate_backprop_method(dev, "something")

        assert method == "backprop"
        assert device is dev

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("accepted_name, official_name", qml.interfaces.INTERFACE_MAP.items())
    def test_validate_backprop_method_all_interface_names(self, accepted_name, official_name):
        """Test that backprop devices are mapped for all possible interface names."""
        if accepted_name in {None, "auto", "scipy"}:
            pytest.skip(f"None is not a backprop interface.")

        dev = qml.device("default.qubit", wires=1)

        diff_method, _, new_dev = QNode._validate_backprop_method(dev, accepted_name)

        assert diff_method == "backprop"
        assert new_dev.capabilities().get("passthru_interface") == official_name

    def test_validate_backprop_child_method(self, monkeypatch):
        """Test that the method for validating the backprop diff method
        tape works as expected if a child device supports backprop"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"

        orig_capabilities = dev.capabilities().copy()
        orig_capabilities["passthru_devices"] = {test_interface: "default.gaussian"}
        monkeypatch.setattr(dev, "capabilities", lambda: orig_capabilities)

        method, diff_options, device = QNode._validate_backprop_method(dev, test_interface)

        assert method == "backprop"
        assert isinstance(device, qml.devices.DefaultGaussian)

    def test_validate_backprop_child_method_wrong_interface(self, monkeypatch):
        """Test that the method for validating the backprop diff method
        tape raises an error if a child device supports backprop but using a different interface"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"

        orig_capabilities = dev.capabilities().copy()
        orig_capabilities["passthru_devices"] = {test_interface: "default.gaussian"}
        monkeypatch.setattr(dev, "capabilities", lambda: orig_capabilities)

        with pytest.raises(
            qml.QuantumFunctionError, match=r"when using the \['something'\] interface"
        ):
            QNode._validate_backprop_method(dev, "another_interface")

    @pytest.mark.autograd
    @pytest.mark.parametrize("device_string", ("default.qubit", "default.qubit.autograd"))
    def test_validate_backprop_finite_shots(self, device_string):
        """Test that a device with finite shots cannot be used with backpropagation."""
        dev = qml.device(device_string, wires=1, shots=100)

        with pytest.raises(qml.QuantumFunctionError, match=r"Backpropagation is only supported"):
            QNode._validate_backprop_method(dev, "autograd")

    @pytest.mark.autograd
    def test_parameter_shift_qubit_device(self):
        """Test that the _validate_parameter_shift method
        returns the correct gradient transform for qubit devices."""
        dev = qml.device("default.qubit", wires=1)
        gradient_fn = QNode._validate_parameter_shift(dev)
        assert gradient_fn[0] is qml.gradients.param_shift

    @pytest.mark.autograd
    def test_parameter_shift_cv_device(self):
        """Test that the _validate_parameter_shift method
        returns the correct gradient transform for cv devices."""
        dev = qml.device("default.gaussian", wires=1)
        gradient_fn = QNode._validate_parameter_shift(dev)
        assert gradient_fn[0] is qml.gradients.param_shift_cv
        assert gradient_fn[1] == {"dev": dev}

    def test_parameter_shift_tape_unknown_model(self, monkeypatch):
        """Test that an unknown model raises an exception"""

        def capabilities(cls):
            capabilities = cls._capabilities
            capabilities.update(model="None")
            return capabilities

        monkeypatch.setattr(qml.devices.DefaultQubit, "capabilities", capabilities)
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="does not support the parameter-shift rule"
        ):
            QNode._validate_parameter_shift(dev)

    @pytest.mark.autograd
    def test_best_method_is_device(self, monkeypatch):
        """Test that the method for determining the best diff method
        for a given device and interface returns the device"""
        dev = qml.device("default.qubit", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)

        # basic check if the device provides a Jacobian
        res = QNode.get_best_method(dev, "another_interface")
        assert res == ("device", {}, dev)

        # device is returned even if backpropagation is possible
        res = QNode.get_best_method(dev, "some_interface")
        assert res == ("device", {}, dev)

    def test_best_method_is_backprop(self, monkeypatch):
        """Test that the method for determining the best diff method
        for a given device and interface returns backpropagation"""
        dev = qml.device("default.qubit", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)

        # backprop is returned when the interfaces match and Jacobian is not provided
        res = QNode.get_best_method(dev, "some_interface")
        assert res == ("backprop", {}, dev)

    def test_best_method_is_param_shift(self, monkeypatch):
        """Test that the method for determining the best diff method
        for a given device and interface returns the parameter shift rule"""
        dev = qml.device("default.qubit", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)

        # parameter shift is returned when Jacobian is not provided and
        # the backprop interfaces do not match
        res = QNode.get_best_method(dev, "another_interface")
        assert res == (qml.gradients.param_shift, {}, dev)

    def test_best_method_is_finite_diff(self, monkeypatch):
        """Test that the method for determining the best diff method
        for a given device and interface returns finite differences"""
        dev = qml.device("default.qubit", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)

        def capabilities(cls):
            capabilities = cls._capabilities
            capabilities.update(model="None")
            return capabilities

        # finite differences is the fallback when we know nothing about the device
        monkeypatch.setattr(qml.devices.DefaultQubit, "capabilities", capabilities)
        res = QNode.get_best_method(dev, "another_interface")
        assert res == (qml.gradients.finite_diff, {}, dev)

    def test_best_method_str_is_device(self, monkeypatch):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'device'"""
        dev = qml.device("default.qubit", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)

        # basic check if the device provides a Jacobian
        res = QNode.best_method_str(dev, "another_interface")
        assert res == "device"

        # device is returned even if backpropagation is possible
        res = QNode.best_method_str(dev, "some_interface")
        assert res == "device"

    def test_best_method_str_is_backprop(self, monkeypatch):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'backprop'"""
        dev = qml.device("default.qubit", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)

        # backprop is returned when the interfaces match and Jacobian is not provided
        res = QNode.best_method_str(dev, "some_interface")
        assert res == "backprop"

    def test_best_method_str_is_param_shift(self, monkeypatch):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'parameter-shift'"""
        dev = qml.device("default.qubit", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)

        # parameter shift is returned when Jacobian is not provided and
        # the backprop interfaces do not match
        res = QNode.best_method_str(dev, "another_interface")
        assert res == "parameter-shift"

    def test_best_method_str_is_finite_diff(self, monkeypatch):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'finite-diff'"""
        dev = qml.device("default.qubit", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)

        def capabilities(cls):
            capabilities = cls._capabilities
            capabilities.update(model="None")
            return capabilities

        # finite differences is the fallback when we know nothing about the device
        monkeypatch.setattr(qml.devices.DefaultQubit, "capabilities", capabilities)
        res = QNode.best_method_str(dev, "another_interface")
        assert res == "finite-diff"

    def test_diff_method(self, mocker):
        """Test that a user-supplied diff method correctly returns the right
        diff method."""
        dev = qml.device("default.qubit", wires=1)

        mock_best = mocker.patch("pennylane.QNode.get_best_method")
        mock_best.return_value = ("best", {}, dev)

        mock_backprop = mocker.patch("pennylane.QNode._validate_backprop_method")
        mock_backprop.return_value = ("backprop", {}, dev)

        mock_device = mocker.patch("pennylane.QNode._validate_device_method")
        mock_device.return_value = ("device", {}, dev)

        qn = QNode(dummyfunc, dev, diff_method="best")
        assert qn.diff_method == "best"
        assert qn.gradient_fn == "best"

        qn = QNode(dummyfunc, dev, diff_method="backprop")
        assert qn.diff_method == "backprop"
        assert qn.gradient_fn == "backprop"
        mock_backprop.assert_called_once()

        qn = QNode(dummyfunc, dev, diff_method="device")
        assert qn.diff_method == "device"
        assert qn.gradient_fn == "device"
        mock_device.assert_called_once()

        qn = QNode(dummyfunc, dev, diff_method="finite-diff")
        assert qn.diff_method == "finite-diff"
        assert qn.gradient_fn is qml.gradients.finite_diff

        qn = QNode(dummyfunc, dev, diff_method="parameter-shift")
        assert qn.diff_method == "parameter-shift"
        assert qn.gradient_fn is qml.gradients.param_shift

        # check that get_best_method was only ever called once
        mock_best.assert_called_once()

    @pytest.mark.autograd
    def test_gradient_transform(self, mocker):
        """Test passing a gradient transform directly to a QNode"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(qml.gradients.finite_difference, "finite_diff_coeffs")

        @qnode(dev, diff_method=qml.gradients.finite_diff)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.gradient_fn is qml.gradients.finite_diff

        qml.grad(circuit)(pnp.array(0.5, requires_grad=True))
        spy.assert_called()

    def test_unknown_diff_method_string(self):
        """Test that an exception is raised for an unknown differentiation method string"""
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="Differentiation method hello not recognized"
        ):
            QNode(dummyfunc, dev, diff_method="hello")

    def test_unknown_diff_method_type(self):
        """Test that an exception is raised for an unknown differentiation method type"""
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Differentiation method 5 must be a gradient transform or a string",
        ):
            QNode(dummyfunc, dev, diff_method=5)

    def test_validate_adjoint_invalid_device(self):
        """Test if a ValueError is raised when an invalid device is provided to
        _validate_adjoint_method"""

        dev = qml.device("default.gaussian", wires=1)

        with pytest.raises(ValueError, match="The default.gaussian device does not"):
            QNode._validate_adjoint_method(dev)

    def test_validate_adjoint_finite_shots(self):
        """Test that a UserWarning is raised when device has finite shots"""

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):
            QNode._validate_adjoint_method(dev)

    def test_adjoint_finite_shots(self):
        """Tests that UserWarning is raised with the adjoint differentiation method
        on QNode construction when the device has finite shots
        """

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):

            @qnode(dev, diff_method="adjoint")
            def circ():
                return qml.expval(qml.PauliZ(0))

    @pytest.mark.autograd
    def test_sparse_diffmethod_error(self):
        """Test that an error is raised when the observable is SparseHamiltonian and the
        differentiation method is not parameter-shift."""
        dev = qml.device("default.qubit", wires=2, shots=None)

        @qnode(dev, diff_method="backprop")
        def circuit(param):
            qml.RX(param, wires=0)
            return qml.expval(qml.SparseHamiltonian(csr_matrix(np.eye(4)), [0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="SparseHamiltonian observable must be"
            " used with the parameter-shift differentiation method",
        ):
            qml.grad(circuit, argnum=0)([0.5])

    def test_qnode_print(self):
        """Test that printing a QNode object yields the right information."""
        dev = qml.device("default.qubit", wires=1)

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qn = QNode(func, dev)

        assert (
            qn.__repr__()
            == "<QNode: wires=1, device='default.qubit.autograd', interface='autograd', diff_method='best'>"
        )

    @pytest.mark.autograd
    def test_diff_method_none(self, tol):
        """Test that diff_method=None creates a QNode with no interface, and no
        device swapping."""
        dev = qml.device("default.qubit", wires=1)

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


class TestTapeConstruction:
    """Tests for the tape construction"""

    def test_basic_tape_construction(self, tol):
        """Test that a quantum tape is properly constructed"""
        dev = qml.device("default.qubit", wires=2)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        qn = QNode(func, dev)

        x = pnp.array(0.12, requires_grad=True)
        y = pnp.array(0.54, requires_grad=True)

        res = qn(x, y)

        assert isinstance(qn.qtape, QuantumTape)
        assert len(qn.qtape.operations) == 3
        assert len(qn.qtape.observables) == 1
        assert qn.qtape.num_params == 2

        expected = qml.execute([qn.tape], dev, None)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # when called, a new quantum tape is constructed
        old_tape = qn.qtape
        res2 = qn(x, y)

        assert np.allclose(res, res2, atol=tol, rtol=0)
        assert qn.qtape is not old_tape

    def test_jacobian(self, tol):
        """Test the jacobian computation"""
        dev = qml.device("default.qubit", wires=2)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        qn = QNode(func, dev, diff_method="finite-diff", h=1e-8, approx_order=2)
        assert qn.gradient_kwargs["h"] == 1e-8
        assert qn.gradient_kwargs["approx_order"] == 2

        jac = qn.gradient_fn(qn)(
            pnp.array(0.45, requires_grad=True), pnp.array(0.1, requires_grad=True)
        )
        assert isinstance(jac, tuple) and len(jac) == 2
        assert jac[0].shape == (2, 2)
        assert jac[1].shape == (2, 2)

    def test_returning_non_measurements(self):
        """Test that an exception is raised if a non-measurement
        is returned from the QNode."""
        dev = qml.device("default.qubit", wires=2)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return 5

        qn = QNode(func, dev)

        with pytest.raises(
            qml.QuantumFunctionError, match="must return either a single measurement"
        ):
            qn(5, 1)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), 5

        qn = QNode(func, dev)

        with pytest.raises(
            qml.QuantumFunctionError, match="must return either a single measurement"
        ):
            qn(5, 1)

    def test_inconsistent_measurement_order(self):
        """Test that an exception is raised if measurements are returned in an
        order different to how they were queued on the tape"""
        dev = qml.device("default.qubit", wires=2)

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
        dev = qml.device("default.qubit", wires=2)

        def func(x, y):
            global op1, op2, op3, m1, m2
            op1 = qml.RX(x, wires=0)
            op2 = qml.RY(y, wires=1)
            op3 = qml.CNOT(wires=[0, 1])
            m1 = qml.expval(qml.PauliZ(0))
            m2 = qml.expval(qml.PauliX(1))
            return [m1, m2]

        qn = QNode(func, dev)
        qn(5, 1)  # evaluate the QNode
        assert qn.qtape.operations == [op1, op2, op3]
        assert qn.qtape.measurements == [m1, m2]

    @pytest.mark.xfail
    def test_multiple_observables_same_wire_expval(self, mocker):
        """Test that the QNode supports returning expectation values of observables that are on the
        same wire (provided that they are Pauli words and qubit-wise commuting)"""
        dev = qml.device("default.qubit", wires=3)

        w = np.random.random((2, 3, 3))

        @qnode(dev)
        def f(w):
            qml.templates.StronglyEntanglingLayers(w, wires=range(3))
            return (
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliX(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliX(2)),
            )

        spy = mocker.spy(qml.devices.DefaultQubit, "apply")
        res = f(w)
        spy.assert_called_once()

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliX(2)]
        qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs, dev)
        res_2 = qnodes(w)

        assert np.allclose(res, res_2)

    @pytest.mark.xfail
    def test_multiple_observables_same_wire_mixed(self, mocker):
        """Test that the QNode supports returning observables that are on the
        same wire but with different return types (provided that the observables are Pauli words and
        qubit-wise commuting)"""
        dev = qml.device("default.qubit", wires=3)

        w = np.random.random((2, 3, 3))

        @qnode(dev)
        def f(w):
            qml.templates.StronglyEntanglingLayers(w, wires=range(3))
            return qml.expval(qml.PauliX(0)), qml.var(qml.PauliX(0) @ qml.PauliZ(1))

        spy = mocker.spy(qml.devices.DefaultQubit, "apply")
        res = f(w)
        spy.assert_called_once()

        q1 = qml.map(qml.templates.StronglyEntanglingLayers, [qml.PauliX(0)], dev, measure="expval")
        q2 = qml.map(
            qml.templates.StronglyEntanglingLayers,
            [qml.PauliX(0) @ qml.PauliZ(1)],
            dev,
            measure="var",
        )

        res_2 = np.array([q1(w), q2(w)]).squeeze()

        assert np.allclose(res, res_2)

    def test_operator_all_wires(self, monkeypatch, tol):
        """Test that an operator that must act on all wires
        does, or raises an error."""
        monkeypatch.setattr(qml.RX, "num_wires", qml.operation.AllWires)

        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        qnode = QNode(circuit, dev)

        with pytest.raises(qml.QuantumFunctionError, match="Operator RX must act on all wires"):
            qnode(0.5)

        dev = qml.device("default.qubit", wires=1)
        qnode = QNode(circuit, dev)
        assert np.allclose(qnode(0.5), np.cos(0.5), atol=tol, rtol=0)


class TestDecorator:
    """Unit tests for the decorator"""

    def test_decorator(self, tol):
        """Test that the decorator correctly creates a QNode."""
        dev = qml.device("default.qubit", wires=2)

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

        assert isinstance(func.qtape, QuantumTape)
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

        dev = qml.device("default.qubit", wires=2)
        qn = QNode(func, dev, interface="autograd")

        for i in range(2):
            qn()

        assert dev.num_executions == 2

        qn2 = QNode(func, dev, interface="autograd")
        for i in range(3):
            qn2()

        assert dev.num_executions == 5

    @pytest.mark.tf
    @pytest.mark.parametrize("interface", ["tf", "auto"])
    def test_correct_number_of_executions_tf(self, interface):
        """Test that number of executions are tracked in the tf interface."""
        import tensorflow as tf

        def func():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        qn = QNode(func, dev, interface=interface)
        for i in range(2):
            qn()

        assert dev.num_executions == 2

        qn2 = QNode(func, dev, interface=interface)
        for i in range(3):
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

        dev = qml.device("default.qubit", wires=2)
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
        dev = qml.device("default.qubit", wires=2)

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
        assert cache != {}

    def test_num_exec_caching_device_swap_two_exec(self):
        """Tests that if we swapped the original device (e.g., when
        diff_method='backprop') then the number of executions recorded is
        correct even with multiple QNode evaluations."""
        dev = qml.device("default.qubit", wires=2)

        cache = {}

        @qml.qnode(dev, diff_method="backprop", cache=cache)
        def circuit():
            qml.RY(0.345, wires=0)
            return qml.expval(qml.PauliZ(0))

        for _ in range(15):
            circuit()

        @qml.qnode(dev, diff_method="backprop", cache=cache)
        def circuit():
            qml.RZ(0.345, wires=0)
            return qml.expval(qml.PauliZ(0))

        for _ in range(15):
            circuit()

        # Although we've evaluated the QNode several times, due to caching,
        # there were two device executions recorded
        assert dev.num_executions == 2
        assert cache != {}

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff"])
    def test_single_expectation_value_with_argnum_one(self, diff_method, tol):
        """Tests correct output shape and evaluation for a QNode
        with a single expval output where only one parameter is chosen to
        estimate the jacobian.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """
        from pennylane import numpy as anp

        dev = qml.device("default.qubit", wires=2)

        x = anp.array(0.543, requires_grad=True)
        y = anp.array(-0.654, requires_grad=True)

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
        res = res
        expected = expected

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("first_par", np.linspace(0.15, np.pi - 0.3, 3))
    @pytest.mark.parametrize("sec_par", np.linspace(0.15, np.pi - 0.3, 3))
    @pytest.mark.parametrize(
        "return_type", [qml.expval(qml.PauliZ(1)), qml.var(qml.PauliZ(1)), qml.probs(wires=[1])]
    )
    def test_defer_meas_if_mcm_unsupported(self, first_par, sec_par, return_type):
        """Tests that the transform using the deferred measurement principle is
        applied if the device doesn't support mid-circuit measurements
        natively."""
        dev = qml.device("default.qubit", wires=2)

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
            return qml.apply(return_type)

        r1 = cry_qnode(first_par, sec_par)
        r2 = conditional_ry_qnode(first_par, sec_par)
        assert np.allclose(r1, r2)

    @pytest.mark.parametrize("basis_state", [[1, 0], [0, 1]])
    def test_sampling_with_mcm(self, basis_state):
        """Tests that a QNode with qml.sample and mid-circuit measurements
        returns the expected results."""
        dev = qml.device("default.qubit", wires=2, shots=1000)

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

        r1 = cry_qnode(first_par)
        r2 = conditional_ry_qnode(first_par)
        assert np.allclose(r1, r2)

    @pytest.mark.tf
    @pytest.mark.parametrize("interface", ["tf", "auto"])
    def test_conditional_ops_tensorflow(self, interface):
        """Test conditional operations with TensorFlow."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

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

        dev = qml.device("default.qubit", wires=2)

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
        dev = qml.device("default.qubit", wires=2)

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

    def test_already_measured_error_operation(self):
        """Test that attempting to apply an operation on a wires that has been
        measured raises an error."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode():
            qml.measure(1)
            qml.PauliX(1)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="wires have been measured already: {1}"):
            qnode()


class TestShots:
    """Unit tests for specifying shots per call."""

    def test_specify_shots_per_call_sample(self):
        """Tests that shots can be set per call for a sample return type."""
        dev = qml.device("default.qubit", wires=1, shots=10)

        @qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        assert len(circuit(0.8)) == 10
        assert len(circuit(0.8, shots=2)) == 2
        assert len(circuit(0.8, shots=3178)) == 3178
        assert len(circuit(0.8)) == 10

    def test_specify_shots_per_call_expval(self):
        """Tests that shots can be set per call for an expectation value.
        Note: this test has a vanishingly small probability to fail."""
        dev = qml.device("default.qubit", wires=1, shots=None)

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

    def test_no_shots_per_call_if_user_has_shots_qfunc_kwarg(self):
        """Tests that the per-call shots overwriting is suspended if user
        has a shots keyword argument, but a warning is raised."""

        dev = qml.device("default.qubit", wires=2, shots=10)

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

    def test_no_shots_per_call_if_user_has_shots_qfunc_arg(self):
        """Tests that the per-call shots overwriting is suspended
        if user has a shots argument, but a warning is raised."""
        dev = qml.device("default.qubit", wires=[0, 1], shots=10)

        def circuit(a, shots):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=0))

        # assert that warning is still raised
        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):
            circuit = QNode(circuit, dev)

        assert len(circuit(0.8, 1)) == 10
        assert circuit.qtape.operations[0].wires.labels == (1,)

        dev = qml.device("default.qubit", wires=2, shots=10)

        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):

            @qnode(dev)
            def circuit(a, shots):
                qml.RX(a, wires=shots)
                return qml.sample(qml.PauliZ(wires=0))

        assert len(circuit(0.8, shots=0)) == 10
        assert circuit.qtape.operations[0].wires.labels == (0,)

    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_shots_setting_does_not_mutate_device(self, diff_method):
        """Tests that per-call shots setting does not change the number of shots in the device."""

        dev = qml.device("default.qubit", wires=1, shots=3)

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
        dev = qml.device("default.qubit", wires=1, shots=5)

        @qml.qnode(dev, cache={})
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # no warning on the first execution
        circuit(0.3)
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            circuit(0.3)

    def test_warning_finite_shots_override(self):
        """Tests that a warning is raised when caching is used with finite shots."""
        dev = qml.device("default.qubit", wires=1)

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
        dev = qml.device("default.qubit", wires=1, shots=5)

        with qml.tape.QuantumTape() as tape:
            qml.RZ(0.3, wires=0)
            qml.expval(qml.PauliZ(0))

        # no warning on the first execution
        cache = {}
        qml.execute([tape], dev, None, cache=cache)
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            qml.execute([tape], dev, None, cache=cache)

    def test_no_warning_infinite_shots(self):
        """Tests that no warning is raised when caching is used with infinite shots."""
        dev = qml.device("default.qubit", wires=1)

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
        dev = qml.device("default.qubit", wires=1, shots=5)

        @qml.qnode(dev, cache=True)
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.probs(wires=0)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Cached execution with finite shots detected")
            qml.jacobian(circuit, argnum=0)(0.3)


@pytest.mark.xfail
class TestSpecs:
    """Tests for the qnode property specs"""

    def test_specs_error(self):
        """Tests an error is raised if the tape is not constructed."""

        dev = qml.device("default.qubit", wires=4)

        @qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match=r"The QNode specifications"):
            circuit.specs

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 10), ("parameter-shift", 12), ("adjoint", 11)]
    )
    def test_specs(self, diff_method, len_info):
        """Tests the specs property with backprop, parameter-shift and adjoint diff_method"""

        dev = qml.device("default.qubit", wires=4)

        @qnode(dev, diff_method=diff_method)
        def circuit(x, y):
            qml.RX(x[0], wires=0)
            qml.Toffoli(wires=(0, 1, 2))
            qml.CRY(x[1], wires=(0, 1))
            qml.Rot(x[2], x[3], y, wires=2)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        x = pnp.array([0.05, 0.1, 0.2, 0.3], requires_grad=True)
        y = pnp.array(0.1, requires_grad=False)

        res = circuit(x, y)

        info = circuit.specs

        assert len(info) == len_info

        assert info["gate_sizes"] == defaultdict(int, {1: 2, 3: 1, 2: 1})
        assert info["gate_types"] == defaultdict(int, {"RX": 1, "Toffoli": 1, "CRY": 1, "Rot": 1})
        assert info["num_operations"] == 4
        assert info["num_observables"] == 2
        assert info["num_diagonalizing_gates"] == 1
        assert info["num_used_wires"] == 3
        assert info["depth"] == 3
        assert info["num_device_wires"] == 4

        assert info["diff_method"] == diff_method

        if diff_method == "parameter-shift":
            assert info["num_parameter_shift_executions"] == 7

        if diff_method != "backprop":
            assert info["device_name"] == "default.qubit"
            assert info["num_trainable_params"] == 4
        else:
            assert info["device_name"] == "default.qubit.autograd"


class TestTapeExpansion:
    """Test that tape expansion within the QNode works correctly"""

    @pytest.mark.parametrize(
        "diff_method,mode",
        [("parameter-shift", "backward"), ("adjoint", "forward"), ("adjoint", "backward")],
    )
    def test_device_expansion(self, diff_method, mode, mocker):
        """Test expansion of an unsupported operation on the device"""
        dev = qml.device("default.qubit", wires=1)

        class UnsupportedOp(qml.operation.Operation):
            num_wires = 1

            def expand(self):
                with qml.tape.QuantumTape() as tape:
                    qml.RX(3 * self.data[0], wires=self.wires)
                return tape

        @qnode(dev, diff_method=diff_method, mode=mode)
        def circuit(x):
            UnsupportedOp(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        if diff_method == "adjoint" and mode == "forward":
            spy = mocker.spy(circuit.device, "execute_and_gradients")
        else:
            spy = mocker.spy(circuit.device, "batch_execute")

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
        dev = qml.device("default.qubit", wires=1)

        class UnsupportedOp(qml.operation.Operation):
            num_wires = 1

            grad_method = "A"
            grad_recipe = ([[3 / 2, 1, np.pi / 6], [-3 / 2, 1, -np.pi / 6]],)

            def expand(self):
                with qml.tape.QuantumTape() as tape:
                    qml.RX(3 * self.data[0], wires=self.wires)
                return tape

        @qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            UnsupportedOp(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = pnp.array(0.5, requires_grad=True)
        spy = mocker.spy(circuit.gradient_fn, "transform_fn")
        qml.grad(circuit)(x)

        # check that the gradient recipe was applied *prior* to
        # device expansion
        input_tape = spy.call_args[0][0]
        assert len(input_tape.operations) == 1
        assert input_tape.operations[0].name == "UnsupportedOp"
        assert input_tape.operations[0].data[0] == x

        shifted_tape1, shifted_tape2 = spy.spy_return[0]

        assert len(shifted_tape1.operations) == 1
        assert shifted_tape1.operations[0].name == "UnsupportedOp"

        assert len(shifted_tape2.operations) == 1
        assert shifted_tape2.operations[0].name == "UnsupportedOp"

        # check second derivative
        assert np.allclose(qml.grad(qml.grad(circuit))(x), -9 * np.cos(3 * x))

    @pytest.mark.autograd
    def test_gradient_expansion(self, mocker):
        """Test that a *supported* operation with no gradient recipe is
        expanded when applying the gradient transform, but not for execution."""
        dev = qml.device("default.qubit", wires=1)

        class PhaseShift(qml.PhaseShift):
            grad_method = None

            def expand(self):
                with qml.tape.QuantumTape() as tape:
                    qml.RY(3 * self.data[0], wires=self.wires)
                return tape

        @qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            return qml.expval(qml.PauliX(0))

        spy = mocker.spy(circuit.device, "batch_execute")
        x = pnp.array(0.5, requires_grad=True)
        circuit(x)

        tape = spy.call_args[0][0][0]

        spy = mocker.spy(circuit.gradient_fn, "transform_fn")
        res = qml.grad(circuit)(x)

        input_tape = spy.call_args[0][0]
        assert len(input_tape.operations) == 2
        assert input_tape.operations[1].name == "RY"
        assert input_tape.operations[1].data[0] == 3 * x

        shifted_tape1, shifted_tape2 = spy.spy_return[0]

        assert len(shifted_tape1.operations) == 2
        assert shifted_tape1.operations[1].name == "RY"

        assert len(shifted_tape2.operations) == 2
        assert shifted_tape2.operations[1].name == "RY"

        assert np.allclose(res, -3 * np.sin(3 * x))

        # test second order derivatives
        res = qml.grad(qml.grad(circuit))(x)
        assert np.allclose(res, -9 * np.cos(3 * x))

    def test_hamiltonian_expansion_analytic(self):
        """Test result if there are non-commuting groups and the number of shots is None"""
        dev = qml.device("default.qubit", wires=3, shots=None)

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
        dev = qml.device("default.qubit", wires=3, shots=50000)

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        c = np.array([-0.6543, 0.24, 0.54])
        H = qml.Hamiltonian(c, obs)
        H.compute_grouping()

        assert len(H.grouping_indices) == 2

        @qnode(dev)
        def circuit():
            return qml.expval(H)

        spy = mocker.spy(qml.transforms, "hamiltonian_expand")
        res = circuit()
        assert np.allclose(res, c[2], atol=0.1)

        spy.assert_called()
        tapes, fn = spy.spy_return

        assert len(tapes) == 2

    def test_invalid_hamiltonian_expansion_finite_shots(self, mocker):
        """Test that an error is raised if multiple expectations are requested
        when using finite shots"""
        dev = qml.device("default.qubit", wires=3, shots=50000)

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        c = np.array([-0.6543, 0.24, 0.54])
        H = qml.Hamiltonian(c, obs)
        H.compute_grouping()

        assert len(H.grouping_indices) == 2

        @qnode(dev)
        def circuit():
            return qml.expval(H), qml.expval(H)

        with pytest.raises(
            ValueError, match="Can only return the expectation of a single Hamiltonian"
        ):
            circuit()

    def test_device_expansion_strategy(self, mocker):
        """Test that the device expansion strategy performs the device
        decomposition at construction time, and not at execution time"""
        dev = qml.device("default.qubit", wires=2)
        x = pnp.array(0.5, requires_grad=True)

        @qnode(dev, diff_method="parameter-shift", expansion_strategy="device")
        def circuit(x):
            qml.SingleExcitation(x, wires=[0, 1])
            return qml.expval(qml.PauliX(0))

        assert circuit.expansion_strategy == "device"
        assert circuit.execute_kwargs["expand_fn"] is None

        spy_expand = mocker.spy(circuit.device, "expand_fn")

        circuit.construct([x], {})
        assert len(circuit.tape.operations) > 0
        spy_expand.assert_called_once()

        circuit(x)
        assert len(spy_expand.call_args_list) == 2

        qml.grad(circuit)(x)
        assert len(spy_expand.call_args_list) == 3

    def test_expansion_multiple_qwc_observables(self, mocker):
        """Test that the QNode correctly expands tapes that return
        multiple measurements of commuting observables"""
        dev = qml.device("default.qubit", wires=2)
        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliY(1)]

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return [qml.expval(o) for o in obs]

        spy_expand = mocker.spy(circuit.device, "expand_fn")
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

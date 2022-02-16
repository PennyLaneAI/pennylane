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
import pytest
import numpy as np
from collections import defaultdict

import pennylane as qml
from pennylane import numpy as pnp
from pennylane import QNodeCollection
from pennylane.qnode_old import qnode, QNode
from pennylane.transforms import draw
from pennylane.tape import JacobianTape, QubitParamShiftTape, CVParamShiftTape


def dummyfunc():
    return None


class TestValidation:
    """Tests for QNode creation and validation"""

    def test_invalid_interface(self):
        """Test that an exception is raised for an invalid interface"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"
        expected_error = (
            rf"Unknown interface {test_interface}\. Interface must be "
            r"one of \['autograd', 'torch', 'tf', 'jax'\]\."
        )

        with pytest.raises(qml.QuantumFunctionError, match=expected_error):
            QNode(dummyfunc, dev, interface="something")

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
            QNode._validate_device_method(dev, None)

        monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)
        tape_class, interface, device, diff_options = QNode._validate_device_method(
            dev, "interface"
        )
        method = diff_options["method"]

        assert tape_class is JacobianTape
        assert method == "device"
        assert interface == "interface"
        assert device is dev

    @pytest.mark.parametrize("interface", ("autograd", "torch", "tensorflow", "jax"))
    def test_validate_backprop_method_finite_shots(self, interface):
        """Tests that an error is raised for backpropagation with finite shots."""

        dev = qml.device("default.qubit", wires=1, shots=3)

        with pytest.raises(qml.QuantumFunctionError, match="Devices with finite shots"):
            QNode._validate_backprop_method(dev, interface)

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

        tape_class, interface, device, diff_options = QNode._validate_backprop_method(
            dev, test_interface
        )
        method = diff_options["method"]

        assert tape_class is JacobianTape
        assert method == "backprop"
        assert interface == "something"
        assert device is dev

    def test_validate_backprop_child_method(self, monkeypatch):
        """Test that the method for validating the backprop diff method
        tape works as expected if a child device supports backprop"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"

        orig_capabilities = dev.capabilities().copy()
        orig_capabilities["passthru_devices"] = {test_interface: "default.gaussian"}
        monkeypatch.setattr(dev, "capabilities", lambda: orig_capabilities)

        tape_class, interface, device, diff_options = QNode._validate_backprop_method(
            dev, test_interface
        )
        method = diff_options["method"]

        assert tape_class is JacobianTape
        assert method == "backprop"
        assert interface == "something"
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

    def test_parameter_shift_tape_qubit_device(self):
        """Test that the get_parameter_shift_method method correctly and
        returns the correct tape for qubit devices."""
        dev = qml.device("default.qubit", wires=1)
        tape_class = QNode._get_parameter_shift_tape(dev)
        assert tape_class is QubitParamShiftTape

    def test_parameter_shift_tape_cv_device(self):
        """Test that the get_parameter_shift_method method correctly and
        returns the correct tape for qubit devices."""
        dev = qml.device("default.gaussian", wires=1)
        tape_class = QNode._get_parameter_shift_tape(dev)
        assert tape_class is CVParamShiftTape

    def test_parameter_shift_tape_unknown_model(self, monkeypatch):
        """test that an unknown model raises an exception"""

        def capabilities(cls):
            capabilities = cls._capabilities
            capabilities.update(model="None")
            return capabilities

        monkeypatch.setattr(qml.devices.DefaultQubit, "capabilities", capabilities)
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="does not support the parameter-shift rule"
        ):
            QNode._get_parameter_shift_tape(dev)

    def test_best_method(self, monkeypatch):
        """Test that the method for determining the best diff method
        for a given device and interface works correctly"""
        dev = qml.device("default.qubit", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)

        # device is top priority
        res = QNode.get_best_method(dev, "another_interface")
        assert res == (JacobianTape, "another_interface", dev, {"method": "device"})

        # backprop is next priority
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)
        res = QNode.get_best_method(dev, "some_interface")
        assert res == (JacobianTape, "some_interface", dev, {"method": "backprop"})

        # The next fallback is parameter-shift.
        res = QNode.get_best_method(dev, "another_interface")
        assert res == (QubitParamShiftTape, "another_interface", dev, {"method": "best"})

        # finally, if both fail, finite differences is the fallback
        def capabilities(cls):
            capabilities = cls._capabilities
            capabilities.update(model="None")
            return capabilities

        monkeypatch.setattr(qml.devices.DefaultQubit, "capabilities", capabilities)
        res = QNode.get_best_method(dev, "another_interface")
        assert res == (JacobianTape, "another_interface", dev, {"method": "numeric"})

    def test_diff_method(self, mocker):
        """Test that a user-supplied diff-method correctly returns the right
        quantum tape, interface, and diff method."""
        dev = qml.device("default.qubit", wires=1)

        mock_best = mocker.patch("pennylane.qnode_old.QNode.get_best_method")
        mock_best.return_value = 1, 2, 3, {"method": "best"}

        mock_backprop = mocker.patch("pennylane.qnode_old.QNode._validate_backprop_method")
        mock_backprop.return_value = 4, 5, 6, {"method": "backprop"}

        mock_device = mocker.patch("pennylane.qnode_old.QNode._validate_device_method")
        mock_device.return_value = 7, 8, 9, {"method": "device"}

        qn = QNode(dummyfunc, dev, diff_method="best")
        assert qn._tape == mock_best.return_value[0]
        assert qn.interface == mock_best.return_value[1]
        assert qn.diff_options["method"] == mock_best.return_value[3]["method"]

        qn = QNode(dummyfunc, dev, diff_method="backprop")
        assert qn._tape == mock_backprop.return_value[0]
        assert qn.interface == mock_backprop.return_value[1]
        assert qn.diff_options["method"] == mock_backprop.return_value[3]["method"]
        mock_backprop.assert_called_once()

        qn = QNode(dummyfunc, dev, diff_method="device")
        assert qn._tape == mock_device.return_value[0]
        assert qn.interface == mock_device.return_value[1]
        assert qn.diff_options["method"] == mock_device.return_value[3]["method"]
        mock_device.assert_called_once()

        qn = QNode(dummyfunc, dev, diff_method="finite-diff")
        assert qn._tape == JacobianTape
        assert qn.diff_options["method"] == "numeric"

        qn = QNode(dummyfunc, dev, diff_method="parameter-shift")
        assert qn._tape == QubitParamShiftTape
        assert qn.diff_options["method"] == "analytic"

        # check that get_best_method was only ever called once
        mock_best.assert_called_once()

    def test_unknown_diff_method(self):
        """Test that an exception is raised for an unknown differentiation method"""
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(
            qml.QuantumFunctionError, match="Differentiation method hello not recognized"
        ):
            QNode(dummyfunc, dev, diff_method="hello")

    def test_validate_adjoint_invalid_device(self):
        """Test if a ValueError is raised when an invalid device is provided to
        _validate_adjoint_method"""

        dev = qml.device("default.gaussian", wires=1)

        with pytest.raises(ValueError, match="The default.gaussian device does not"):
            QNode._validate_adjoint_method(dev, "tf")

    def test_validate_adjoint_finite_shots(self):
        """Test that a UserWarning is raised when device has finite shots"""

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):
            QNode._validate_adjoint_method(dev, "autograd")

    def test_adjoint_finite_shots(self):
        """Tests that UserWarning is raised with the adjoint differentiation method
        on QNode construction when the device has finite shots
        """

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.warns(
            UserWarning, match="Requested adjoint differentiation to be computed with finite shots."
        ):

            @qml.qnode_old.qnode(dev, diff_method="adjoint")
            def circ():
                return qml.expval(qml.PauliZ(0))

    def test_validate_reversible_finite_shots(self):
        """Test that a UserWarning is raised when validating the reversible differentiation method
        and using a device that has finite shots
        """

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.warns(
            UserWarning,
            match="Requested reversible differentiation to be computed with finite shots.",
        ):
            QNode._validate_reversible_method(dev, "autograd")

    def test_reversible_finite_shots(self):
        """Tests that UserWarning is raised with the reversible differentiation method
        on QNode construction when the device has finite shots
        """

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.warns(
            UserWarning,
            match="Requested reversible differentiation to be computed with finite shots.",
        ):

            @qml.qnode_old.qnode(dev, diff_method="reversible")
            def circ():
                return qml.expval(qml.PauliZ(0))

    def test_qnode_print(self):
        """Test that printing a QNode object yields the right information."""
        dev = qml.device("default.qubit", wires=1)

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qn = qml.qnode_old.QNode(func, dev, diff_method="finite-diff")

        assert (
            qn.__repr__()
            == "<QNode: wires=1, device='default.qubit', interface='autograd', diff_method='finite-diff'>"
        )
        assert qn.diff_method_change == False

    def test_qnode_best_diff_method_backprop(self):
        """Test that selected "best" diff_method is correctly set to 'backprop'."""
        dev = qml.device("default.qubit", wires=1)

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qn = qml.qnode_old.QNode(func, dev)

        assert qn.diff_method == "backprop"
        assert qn.diff_method_change

    def test_qnode_best_diff_method_parameter_shift(self):
        """Test that selected "best" diff_method is correctly set to 'parameter-shift'."""
        dev = qml.device("default.mixed", wires=1)

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qn = qml.qnode_old.QNode(func, dev)

        assert qn.diff_method == "parameter-shift"
        assert qn.diff_method_change

    def test_qnode_best_diff_method_device(self, monkeypatch):
        """Test that selected "best" diff_method is correctly set to 'device'."""
        dev = qml.device("default.qubit", wires=1)

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # Force the "best" method to be "device"
        monkeypatch.setitem(dev._capabilities, "passthru_interface", "some_interface")
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)
        qn = qml.qnode_old.QNode(func, dev)
        assert qn.diff_method == "device"
        assert qn.diff_method_change

    def test_qnode_best_diff_method_finite_diff(self, monkeypatch):
        """Test that selected "best" diff_method is correctly set to 'finite-diff'."""
        dev = qml.device("default.qubit", wires=1)

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        def capabilities(cls):
            capabilities = cls._capabilities
            capabilities.update(model="None")
            return capabilities

        # Force the "best" method to be "finite-diff"
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)
        monkeypatch.setattr(qml.devices.DefaultQubit, "capabilities", capabilities)
        qn = qml.qnode_old.QNode(func, dev)
        assert qn.diff_method == "finite-diff"
        assert qn.diff_method_change

    def test_qnode_best_diff_method_finite_fallback(self):
        """Test that selected "best" diff_method is correctly set to 'finite-diff'
        in cases where other methods are not available."""

        # Custom operation which has grad_method="finite_diff"
        class MyRX(qml.operation.Operation):
            num_wires = 1
            is_composable_rotation = True
            basis = "X"
            grad_method = "F"

            @staticmethod
            def compute_matrix(*params):
                return qml.RX.compute_matrix(*params)

        dev = qml.device("default.mixed", wires=3, shots=None)
        dev.operations.add("MyRX")

        def circuit(x):
            MyRX(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        qnode = qml.qnode_old.QNode(circuit, dev, diff_method="best")

        # Before execution correctly show 'parameter-shift'
        assert qnode.diff_method == "parameter-shift"

        par = qml.numpy.array(0.3)
        qml.grad(qnode)(par)

        # After execution correctly show 'finite-diff'
        assert qnode.diff_method == "finite-diff"

    @pytest.mark.parametrize(
        "method",
        [
            "best",
            "parameter-shift",
            "finite-diff",
            "reversible",
            "adjoint",
            "backprop",
        ],
    )
    def test_to_tf(self, method, mocker):
        """Test if interface change is working"""
        tf = pytest.importorskip("tensorflow")
        dev = qml.device("default.qubit", wires=1)

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # Test if interface change works with different diff_methods
        qn = qml.qnode_old.QNode(func, dev, interface="autograd", diff_method=method)
        spy = mocker.spy(qn, "_get_best_diff_method")
        qn.to_tf()
        if method == "best":
            spy.assert_called_once()

    @pytest.mark.parametrize(
        "method",
        [
            "best",
            "parameter-shift",
            "finite-diff",
            "reversible",
            "adjoint",
            "backprop",
        ],
    )
    def test_to_autograd(self, method, mocker):
        """Test if interface change is working"""
        dev = qml.device("default.qubit", wires=1)
        tf = pytest.importorskip("tensorflow")

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # Test if interface change works with different diff_methods
        qn = qml.qnode_old.QNode(func, dev, interface="tf", diff_method=method)
        spy = mocker.spy(qn, "_get_best_diff_method")
        qn.to_autograd()
        if method == "best":
            spy.assert_called_once()

    @pytest.mark.parametrize(
        "method",
        [
            "best",
            "parameter-shift",
            "finite-diff",
            "reversible",
            "adjoint",
            "backprop",
        ],
    )
    def test_to_torch(self, method, mocker):
        """Test if interface change is working"""
        dev = qml.device("default.qubit", wires=1)
        torch = pytest.importorskip("torch")

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # Test if interface change works with different diff_methods
        qn = qml.qnode_old.QNode(func, dev, interface="autograd", diff_method=method)
        spy = mocker.spy(qn, "_get_best_diff_method")
        qn.to_torch()
        if method == "best":
            spy.assert_called_once()

    @pytest.mark.parametrize(
        "method",
        [
            "best",
            "parameter-shift",
            "finite-diff",
            "reversible",
            "adjoint",
            "backprop",
        ],
    )
    def test_to_jax(self, method, mocker):
        """Test if interface change is working"""
        dev = qml.device("default.qubit", wires=1)
        jax = pytest.importorskip("jax")

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # Test if interface change works with different diff_methods
        qn = qml.qnode_old.QNode(func, dev, interface="autograd", diff_method=method)
        spy = mocker.spy(qn, "_get_best_diff_method")
        qn.to_jax()
        if method == "best":
            spy.assert_called_once()

    @pytest.mark.parametrize(
        "par", [None, 1, 1.1, np.array(1.2), pnp.array(1.3, requires_grad=True)]
    )
    def test_diff_method_none(self, par):
        """Test if diff_method=None works as intended."""
        dev = qml.device("default.qubit", wires=1)

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qn = qml.qnode_old.QNode(func, dev, diff_method=None)
        assert qn.interface is None

        grad = qml.grad(qn)

        # Raise error in cases 1 and 5, as non-trainable parameters do not trigger differentiation.
        # Raise warning in cases 1-4 as there a no trainable parameters.
        # Case 1: No input
        # Case 2: int input
        # Case 3: float input
        # Case 4: numpy input
        # Case 5: differentiable tensor input
        if par is None:
            with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
                with pytest.raises(TypeError) as exp:
                    grad()
        elif hasattr(par, "requires_grad"):
            with pytest.raises(TypeError) as exp:
                grad(par)
        else:
            with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
                grad(par)

    def test_diff_method_none_no_qnode_param(self):
        """Test if diff_method=None works as intended."""
        dev = qml.device("default.qubit", wires=1)

        def func():
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        qn = qml.qnode_old.QNode(func, dev, diff_method=None)
        assert qn.interface is None

        grad = qml.grad(qn)

        # No differentiation required. No error raised.
        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            grad()

    def test_unrecognized_keyword_arguments_validation(self):
        """Tests that a UserWarning is raised when unrecognized keyword arguments are provided."""

        # use two unrecognized methods, to confirm that multiple warnings are raised
        unrecognized_one = "test_method_one"
        unrecognized_two = "test_method_two"
        warning_text = (
            " is unrecognized, and will not be included in your computation. "
            "Please review the QNode class or qnode decorator for the list of available "
            "keyword variables."
        )

        expected_warnings = {
            (UserWarning, f"'{unrecognized_one}'{warning_text}"),
            (UserWarning, f"'{unrecognized_two}'{warning_text}"),
        }

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.warns(UserWarning) as warning_list:

            QNode(dummyfunc, dev, test_method_one=1, test_method_two=2)

        warnings = {(warning.category, warning.message.args[0]) for warning in warning_list}
        assert warnings == expected_warnings

    def test_unrecognized_keyword_arguments_validation_decorator(self):
        """Tests that a UserWarning is raised when unrecognized keyword arguments are provided."""

        # use two unrecognized methods, to confirm that multiple warnings are raised
        unrecognized_one = "test_method_one"
        unrecognized_two = "test_method_two"
        warning_text = (
            " is unrecognized, and will not be included in your computation. "
            "Please review the QNode class or qnode decorator for the list of available "
            "keyword variables."
        )

        expected_warnings = {
            (UserWarning, f"'{unrecognized_one}'{warning_text}"),
            (UserWarning, f"'{unrecognized_two}'{warning_text}"),
        }

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.warns(UserWarning) as warning_list:

            @qml.qnode_old.qnode(dev, test_method_one=1, test_method_two=2)
            def circ():
                return qml.expval(qml.PauliZ(0))

        warnings = {(warning.category, warning.message.args[0]) for warning in warning_list}
        assert warnings == expected_warnings


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

        x = 0.12
        y = 0.54

        res = qn(x, y)

        assert isinstance(qn.qtape, JacobianTape)
        assert len(qn.qtape.operations) == 3
        assert len(qn.qtape.observables) == 1
        assert qn.qtape.num_params == 2

        expected = qn.qtape.execute(dev)
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

        qn = QNode(func, dev, h=1e-8, order=2)
        assert qn.diff_options["h"] == 1e-8
        assert qn.diff_options["order"] == 2

        x = 0.12
        y = 0.54

        res = qn(x, y)
        jac = qn.qtape.jacobian(dev, params=[0.45, 0.1])

        assert jac.shape == (4, 2)

    def test_diff_method_expansion(self, monkeypatch, mocker):
        """Test that a QNode with tape expansion during construction
        preserves the differentiation method."""

        class MyDev(qml.devices.DefaultQubit):
            """Dummy device that supports device Jacobians"""

            @classmethod
            def capabilities(cls):
                capabilities = super().capabilities().copy()
                capabilities.update(
                    provides_jacobian=True,
                )
                return capabilities

            def jacobian(self, *args, **kwargs):
                return np.zeros((2, 4))

        dev = MyDev(wires=2)

        def func(x, y):
            # the U2 operation is not supported on default.qubit
            # and is decomposed.
            qml.U2(x, y, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0)

        qn = QNode(func, dev, diff_method="device", h=1e-8, order=2)

        assert qn.diff_options["method"] == "device"
        assert qn.diff_options["h"] == 1e-8
        assert qn.diff_options["order"] == 2

        x = pnp.array(0.12, requires_grad=True)
        y = pnp.array(0.54, requires_grad=True)

        spy = mocker.spy(JacobianTape, "expand")
        res = qn(x, y)

        spy.assert_called_once()
        assert qn.qtape.jacobian_options["method"] == "device"
        assert qn.qtape.jacobian_options["h"] == 1e-8
        assert qn.qtape.jacobian_options["order"] == 2

        spy = mocker.spy(JacobianTape, "jacobian")
        jac = qml.jacobian(qn)(x, y)

        assert spy.call_args_list[0][1]["method"] == "device"

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
        """Test evaluation exceeds as expected if measurements are returned in the
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

    def test_draw_transform(self):
        """Test circuit drawing"""

        x = pnp.array(0.1, requires_grad=True)
        y = pnp.array([0.2, 0.3], requires_grad=True)
        z = pnp.array(0.4, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="autograd")
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        result = draw(circuit)(p1=x, p3=z)
        expected = "0: ──RX(0.10)──RX(0.40)─╭C─┤ ╭<Z@X>\n" "1: ──RY(0.06)───────────╰X─┤ ╰<Z@X>"

        assert result == expected

    def test_drawing(self):
        """Test circuit drawing"""

        x = pnp.array(0.1, requires_grad=True)
        y = pnp.array([0.2, 0.3], requires_grad=True)
        z = pnp.array(0.4, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="autograd")
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        circuit(p1=x, p3=z)

        with pytest.warns(UserWarning, match="The QNode.draw method has been deprecated."):
            result = circuit.draw()
        expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──╭┤ ⟨Z ⊗ X⟩ 
 1: ──RY(0.06)───────────╰X──╰┤ ⟨Z ⊗ X⟩ 
"""

        assert result == expected

    def test_drawing_ascii(self):
        """Test circuit drawing when using ASCII characters"""
        from pennylane import numpy as anp

        x = anp.array(0.1, requires_grad=True)
        y = anp.array([0.2, 0.3], requires_grad=True)
        z = anp.array(0.4, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="autograd")
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        circuit(p1=x, p3=z)

        with pytest.warns(UserWarning, match="The QNode.draw method has been deprecated."):
            result = circuit.draw(charset="ascii")
        expected = """\
 0: --RX(0.1)---RX(0.4)--+C--+| <Z @ X> 
 1: --RY(0.06)-----------+X--+| <Z @ X> 
"""

        assert result == expected

    def test_drawing_exception(self):
        """Test that an error is raised if a QNode is drawn prior to
        construction."""
        from pennylane import numpy as anp

        x = anp.array(0.1, requires_grad=True)
        y = anp.array([0.2, 0.3], requires_grad=True)
        z = anp.array(0.4, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="autograd")
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        with pytest.raises(qml.QuantumFunctionError, match="can only be drawn after"):
            circuit.draw()

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


class TestDecorator:
    """Unittests for the decorator"""

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

        x = 0.12
        y = 0.54

        res = func(x, y)

        assert isinstance(func.qtape, JacobianTape)
        assert len(func.qtape.operations) == 3
        assert len(func.qtape.observables) == 1
        assert func.qtape.num_params == 2

        expected = func.qtape.execute(dev)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # when called, a new quantum tape is constructed
        old_tape = func.qtape
        res2 = func(x, y)

        assert np.allclose(res, res2, atol=tol, rtol=0)
        assert func.qtape is not old_tape


@pytest.mark.usefixtures("skip_if_no_dask_support")
class TestQNodeCollection:
    """Unittests for the QNodeCollection"""

    def test_multi_thread(self):
        """Test that multi-threaded queuing works correctly"""
        n_qubits = 4
        n_batches = 5
        dev = qml.device("default.qubit", wires=n_qubits)

        def circuit(inputs, weights):
            for index, input in enumerate(inputs):
                qml.RY(input, wires=index)
            for index in range(n_qubits - 1):
                qml.CNOT(wires=(index, index + 1))
            for index, weight in enumerate(weights):
                qml.RX(weight, wires=index)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_qubits)}

        try:
            qnode = QNodeCollection([QNode(circuit, dev) for _ in range(n_batches)])
        except Exception as e:
            pytest.fail("QNodeCollection cannot be instantiated")
        x = np.random.rand(n_qubits).astype(np.float64)
        p = np.random.rand(weight_shapes["weights"]).astype(np.float64)
        try:
            for _ in range(10):
                qnode(x, p, parallel=True)
        except:
            pytest.fail("Multi-threading on QuantumTape failed")


class TestIntegration:
    """Integration tests."""

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

    def test_correct_number_of_executions_tf(self):
        """Test that number of executions are tracked in the tf interface."""
        tf = pytest.importorskip("tf")

        def func():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        qn = QNode(func, dev, interface="tf")
        for i in range(2):
            qn()

        assert dev.num_executions == 2

        qn2 = QNode(func, dev, interface="tf")
        for i in range(3):
            qn2()

        assert dev.num_executions == 5

        # qubit of different interface
        qn3 = QNode(func, dev, interface="autograd")
        qn3()

        assert dev.num_executions == 6

    def test_correct_number_of_executions_torch(self):
        """Test that number of executions are tracked in the torch interface."""
        torch = pytest.importorskip("torch")

        def func():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        qn = QNode(func, dev, interface="torch")
        for i in range(2):
            qn()

        assert dev.num_executions == 2

        qn2 = QNode(func, dev, interface="torch")
        for i in range(3):
            qn2()

        assert dev.num_executions == 5

        # qubit of different interface
        qn3 = QNode(func, dev, interface="autograd")
        qn3()

        assert dev.num_executions == 6

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff", "reversible"])
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

        @qml.qnode_old.qnode(
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


class TestMutability:
    """Test for QNode immutability"""

    def test_mutable(self, mocker, tol):
        """Test that a QNode which has structure dependent
        on trainable arguments is reconstructed with
        every call, and remains differentiable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode_old.qnode(dev, mutable=True)
        def circuit(x):
            if x < 0:
                qml.RY(x, wires=0)
            else:
                qml.RZ(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = 0.5
        spy = mocker.spy(circuit, "construct")
        res = circuit(x)
        spy.assert_called_once_with((x,), {})
        assert len(spy.call_args_list) == 1
        assert circuit.qtape.operations[0].name == "RZ"
        assert circuit.qtape.operations[0].data == [x]
        np.testing.assert_allclose(res, 1, atol=tol, rtol=0)

        # calling the qnode with new arguments reconstructs the tape
        x = -0.5
        res = circuit(x)
        spy.assert_called_with((x,), {})
        assert len(spy.call_args_list) == 2
        assert circuit.qtape.operations[0].name == "RY"
        assert circuit.qtape.operations[0].data == [x]
        np.testing.assert_allclose(res, np.cos(x), atol=tol, rtol=0)

        # test differentiability
        grad = qml.grad(circuit, argnum=0)(0.5)
        np.testing.assert_allclose(grad, 0, atol=tol, rtol=0)

        grad = qml.grad(circuit, argnum=0)(-0.5)
        np.testing.assert_allclose(grad, -np.sin(-0.5), atol=tol, rtol=0)

    def test_immutable(self, mocker, tol):
        """Test that a QNode which has structure dependent
        on trainable arguments is *not* reconstructed with
        every call when mutable=False"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode_old.qnode(dev, mutable=False)
        def circuit(x):
            if x < 0:
                qml.RY(x, wires=0)
            else:
                qml.RZ(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = 0.5
        spy = mocker.spy(circuit, "construct")
        res = circuit(x)
        spy.assert_called_once_with((x,), {})
        assert len(spy.call_args_list) == 1
        assert circuit.qtape.operations[0].name == "RZ"
        assert circuit.qtape.operations[0].data == [x]
        np.testing.assert_allclose(res, 1, atol=tol, rtol=0)

        # calling the qnode with new arguments does not reconstruct the tape
        x = -0.5
        res = circuit(x)
        spy.assert_called_once_with((0.5,), {})
        assert len(spy.call_args_list) == 1
        assert circuit.qtape.operations[0].name == "RZ"
        assert circuit.qtape.operations[0].data == [0.5]
        np.testing.assert_allclose(res, 1, atol=tol, rtol=0)

        # test differentiability. The circuit will assume an RZ gate
        with pytest.warns(UserWarning, match="Output seems independent of input"):
            grad = qml.grad(circuit, argnum=0)(-0.5)
        np.testing.assert_allclose(grad, 0, atol=tol, rtol=0)


class TestShots:
    """Unittests for specifying shots per call."""

    def test_specify_shots_per_call_sample(self):
        """Tests that shots can be set per call for a sample return type."""
        dev = qml.device("default.qubit", wires=1, shots=10)

        @qml.qnode_old.qnode(dev)
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

        @qml.qnode_old.qnode(dev)
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
            circuit = qml.qnode_old.QNode(circuit, dev)

        assert len(circuit(0.8)) == 10
        assert circuit.qtape.operations[0].wires.labels == (0,)

        assert len(circuit(0.8, shots=1)) == 10
        assert circuit.qtape.operations[0].wires.labels == (1,)

        assert len(circuit(0.8, shots=0)) == 10
        assert circuit.qtape.operations[0].wires.labels == (0,)

    def test_no_shots_per_call_if_user_has_shots_qfunc_arg(self):
        """Tests that the per-call shots overwriting is suspended
        if user has a shots argument, but a warning is raised."""

        # Todo: use standard creation of qnode below for both asserts once we do not parse args to tensors any more
        dev = qml.device("default.qubit", wires=[qml.numpy.array(0), qml.numpy.array(1)], shots=10)

        def circuit(a, shots):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=qml.numpy.array(0)))

        # assert that warning is still raised
        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):
            circuit = qml.qnode_old.QNode(circuit, dev)

        assert len(circuit(0.8, 1)) == 10
        assert circuit.qtape.operations[0].wires.labels == (1,)

        dev = qml.device("default.qubit", wires=2, shots=10)

        @qml.qnode_old.qnode(dev)
        def circuit(a, shots):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=0))

        assert len(circuit(0.8, shots=0)) == 10
        assert circuit.qtape.operations[0].wires.labels == (0,)

    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_shots_setting_does_not_mutate_device(self, diff_method):
        """Tests that per-call shots setting does not change the number of shots in the device."""

        dev = qml.device("default.qubit", wires=1, shots=3)

        @qml.qnode_old.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        assert dev.shots == 3
        res = circuit(0.8, shots=2)
        assert len(res) == 2
        assert dev.shots == 3


class TestSpecs:
    """Tests for the qnode property specs"""

    def test_specs_error(self):
        """Tests an error is raised if the tape is not constructed."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode_old.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match=r"The QNode specifications"):
            circuit.specs

    @pytest.mark.parametrize(
        "diff_method, len_info", [("backprop", 10), ("parameter-shift", 12), ("adjoint", 11)]
    )
    def test_specs(self, diff_method, len_info):
        """Tests the specs property with backprop"""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode_old.qnode(dev, diff_method=diff_method)
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


def test_finitediff_float32(tol):
    """Tests that float32 parameters do not effect order 1 finite-diff results.

    Checks bugfix.  Problem occured with StronglyEntanglingLayers, but not simpler circuits.
    """

    n_wires = 2
    n_layers = 2

    shape = qml.templates.StronglyEntanglingLayers.shape(n_wires=n_wires, n_layers=n_layers)

    rng = pnp.random.default_rng(seed=42)
    params = rng.random(shape, requires_grad=True)
    params_f32 = pnp.array(params, dtype=np.float32, requires_grad=True)

    dev = qml.device("default.qubit", n_wires)

    @qml.qnode_old.qnode(dev, diff_method="finite-diff", order=1)
    def circuit(params):
        qml.templates.StronglyEntanglingLayers(params, wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

    grad64 = qml.grad(circuit)(params)
    grad32 = qml.grad(circuit)(params_f32)

    assert np.allclose(grad64, grad32, atol=tol, rtol=0)


class TestDrawMethod:
    """Tests for the deprecated qnode.draw() method"""

    def test_method_deprecation(self):
        """Test that the qnode.draw() method raises a deprecation warning"""

        x = np.array(0.1)
        y = np.array([0.2, 0.3])
        z = np.array(0.4)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode_old.qnode(dev, interface="autograd")
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        circuit(p1=x, p3=z)

        with pytest.warns(UserWarning, match=r"The QNode\.draw method has been deprecated"):
            result = circuit.draw()

        expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──╭┤ ⟨Z ⊗ X⟩ 
 1: ──RY(0.06)───────────╰X──╰┤ ⟨Z ⊗ X⟩ 
"""

        assert result == expected

    def test_invalid_wires(self):
        """Test that an exception is raised if a wire in the wire
        ordering does not exist on the device"""
        dev = qml.device("default.qubit", wires=["a", -1, "q2"])

        @qml.qnode_old.qnode(dev)
        def circuit():
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=["a", "q2"])
            qml.RX(0.2, wires="a")
            return qml.expval(qml.PauliX(wires="q2"))

        circuit()

        with pytest.raises(ValueError, match="contains wires not contained on the device"):
            res = circuit.draw(wire_order=["q2", 5])

    def test_tape_not_constructed(self):
        """Test that an exception is raised if the tape has not been constructed"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode_old.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliX(wires=0))

        with pytest.raises(
            qml.QuantumFunctionError, match="after its quantum tape has been constructed"
        ):
            res = circuit.draw()

    def test_show_all_wires_error(self):
        """Test that show_all_wires will raise an error if the provided wire
        order does not contain all wires on the device"""

        dev = qml.device("default.qubit", wires=[-1, "a", "q2", 0])

        @qml.qnode_old.qnode(dev)
        def circuit():
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=[-1, "q2"])
            return qml.expval(qml.PauliX(wires="q2"))

        circuit()

        with pytest.raises(ValueError, match="must contain all wires"):
            circuit.draw(show_all_wires=True, wire_order=[-1, "a"])

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
"""Unit tests for the QNode"""
import pytest
import numpy as np

import pennylane as qml
from pennylane.tape import QuantumTape, QNode, qnode, QubitParamShiftTape, CVParamShiftTape
from pennylane.tape.measure import MeasurementProcess


class TestValidation:
    """Tests for QNode creation and validation"""

    def test_invalid_interface(self):
        """Test that an exception is raised for an invalid interface"""
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(qml.QuantumFunctionError, match="Unknown interface"):
            QNode(None, dev, interface="something")

    def test_invalid_device(self):
        """Test that an exception is raised for an invalid device"""
        with pytest.raises(qml.QuantumFunctionError, match="Invalid device"):
            QNode(None, None)

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
        tape_class, interface, method = QNode._validate_device_method(dev, "interface")

        assert tape_class is QuantumTape
        assert method == "device"
        assert interface == "interface"

    def test_validate_backprop_method_invalid_device(self):
        """Test that the method for validating the backprop diff method
        tape raises an exception if the device does not support backprop."""
        dev = qml.device("default.qubit", wires=1)

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

        tape_class, interface, method = QNode._validate_backprop_method(dev, test_interface)

        assert tape_class is QuantumTape
        assert method == "backprop"
        assert interface == None

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

        # backprop is given priority
        res = QNode.get_best_method(dev, "some_interface")
        assert res == (QuantumTape, None, "backprop")

        # device is the next priority
        res = QNode.get_best_method(dev, "another_interface")
        assert res == (QuantumTape, "another_interface", "device")

        # The next fallback is parameter-shift.
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)
        res = QNode.get_best_method(dev, "another_interface")
        assert res == (QubitParamShiftTape, "another_interface", "best")

        # finally, if both fail, finite differences is the fallback
        def capabilities(cls):
            capabilities = cls._capabilities
            capabilities.update(model="None")
            return capabilities

        monkeypatch.setattr(qml.devices.DefaultQubit, "capabilities", capabilities)
        res = QNode.get_best_method(dev, "another_interface")
        assert res == (QuantumTape, "another_interface", "numeric")

    def test_diff_method(self, mocker):
        """Test that a user-supplied diff-method correctly returns the right
        quantum tape, interface, and diff method."""
        dev = qml.device("default.qubit", wires=1)

        mock_best = mocker.patch("pennylane.tape.QNode.get_best_method")
        mock_best.return_value = 1, 2, 3

        mock_backprop = mocker.patch("pennylane.tape.QNode._validate_backprop_method")
        mock_backprop.return_value = 4, 5, 6

        mock_device = mocker.patch("pennylane.tape.QNode._validate_device_method")
        mock_device.return_value = 7, 8, 9

        qn = QNode(None, dev, diff_method="best")
        assert qn._tape == mock_best.return_value[0]
        assert qn.interface == mock_best.return_value[1]
        assert qn.diff_options["method"] == mock_best.return_value[2]

        qn = QNode(None, dev, diff_method="backprop")
        assert qn._tape == mock_backprop.return_value[0]
        assert qn.interface == mock_backprop.return_value[1]
        assert qn.diff_options["method"] == mock_backprop.return_value[2]
        mock_backprop.assert_called_once()

        qn = QNode(None, dev, diff_method="device")
        assert qn._tape == mock_device.return_value[0]
        assert qn.interface == mock_device.return_value[1]
        assert qn.diff_options["method"] == mock_device.return_value[2]
        mock_device.assert_called_once()

        qn = QNode(None, dev, diff_method="finite-diff")
        assert qn._tape == QuantumTape
        assert qn.diff_options["method"] == "numeric"

        qn = QNode(None, dev, diff_method="parameter-shift")
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
            QNode(None, dev, diff_method="hello")


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

        assert isinstance(qn.qtape, QuantumTape)
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


class TestTFInterface:
    """Unittests for applying the tensorflow interface"""

    def test_import_error(self, mocker):
        """Test that an exception is caught on import error"""
        tf = pytest.importorskip("tensorflow", minversion="2.1")
        mock = mocker.patch("pennylane.tape.interfaces.tf.TFInterface.apply")
        mock.side_effect = ImportError()

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        qn = QNode(func, dev, interface="tf")

        with pytest.raises(
            qml.QuantumFunctionError,
            match="TensorFlow not found. Please install the latest version of TensorFlow to enable the 'tf' interface",
        ):
            qn(0.1, 0.1)


class TestTorchInterface:
    """Unittests for applying the torch interface"""

    torch = pytest.importorskip("torch", minversion="1.3")

    def test_import_error(self, mocker):
        """Test that an exception is caught on import error"""
        mock = mocker.patch("pennylane.tape.interfaces.torch.TorchInterface.apply")
        mock.side_effect = ImportError()

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        qn = QNode(func, dev, interface="torch")

        with pytest.raises(
            qml.QuantumFunctionError,
            match="PyTorch not found. Please install the latest version of PyTorch to enable the 'torch' interface",
        ):
            qn(0.1, 0.1)


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

        assert isinstance(func.qtape, QuantumTape)
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

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
from pennylane import QNodeCollection
from pennylane.tape import JacobianTape, QNode, draw, qnode, QubitParamShiftTape, CVParamShiftTape


def dummyfunc():
    return None


class TestValidation:
    """Tests for QNode creation and validation"""

    def test_invalid_interface(self):
        """Test that an exception is raised for an invalid interface"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"
        expected_error = (
            fr"Unknown interface {test_interface}\. Interface must be "
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
        tape_class, interface, device, diff_options = QNode._validate_device_method(dev, "interface")
        method = diff_options["method"]

        assert tape_class is JacobianTape
        assert method == "device"
        assert interface == "interface"
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

        tape_class, interface, device, diff_options = QNode._validate_backprop_method(dev, test_interface)
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

        tape_class, interface, device, diff_options = QNode._validate_backprop_method(dev, test_interface)
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

        with pytest.raises(qml.QuantumFunctionError, match=r"when using the \['something'\] interface"):
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

        mock_best = mocker.patch("pennylane.tape.QNode.get_best_method")
        mock_best.return_value = 1, 2, 3, {"method": "best"}

        mock_backprop = mocker.patch("pennylane.tape.QNode._validate_backprop_method")
        mock_backprop.return_value = 4, 5, 6, {"method": "backprop"}

        mock_device = mocker.patch("pennylane.tape.QNode._validate_device_method")
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

        x = 0.12
        y = 0.54

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

        result = draw(circuit)(p1=x, p3=z)
        expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──╭┤ ⟨Z ⊗ X⟩ 
 1: ──RY(0.06)───────────╰X──╰┤ ⟨Z ⊗ X⟩ 
"""

        assert result == expected

    def test_draw_transform_ascii(self):
        """Test circuit drawing when using ASCII characters"""
        from pennylane import numpy as anp

        x = anp.array(0.1, requires_grad=True)
        y = anp.array([0.2, 0.3], requires_grad=True)
        z = anp.array(0.4, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="autograd")
        def circuit(p1, p2, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        result = draw(circuit, charset="ascii")(p1=x, p2=y, p3=z)
        expected = """\
 0: --RX(0.1)---RX(0.4)--+C--+| <Z @ X> 
 1: --RY(0.06)-----------+X--+| <Z @ X> 
"""

        assert result == expected

    def test_draw_transform_raises(self):
        qml.disable_tape()
        dev = qml.device("default.qubit", wires=2)
        @qml.qnode(dev, interface="autograd")
        def circuit(p1, p2, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        with pytest.raises(ValueError, match="only works when tape mode is enabled"):
            result = draw(circuit, charset="ascii")

        qml.enable_tape()

    def test_drawing(self):
        """Test circuit drawing"""
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
        qn = QNode(func, dev, interface="tf", diff_method="parameter-shift")

        with pytest.raises(
            qml.QuantumFunctionError,
            match="TensorFlow not found. Please install the latest version of TensorFlow to enable the 'tf' interface",
        ):
            qn(0.1, 0.1)

    def test_drawing(self):
        """Test circuit drawing when using the TF interface"""
        tf = pytest.importorskip("tensorflow", minversion="2.1")

        x = tf.Variable(0.1, dtype=tf.float64)
        y = tf.Variable([0.2, 0.3], dtype=tf.float64)
        z = tf.Variable(0.4, dtype=tf.float64)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="tf")
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        circuit(p1=x, p3=z)

        result = circuit.draw()
        expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──╭┤ State 
 1: ──RY(0.06)───────────╰X──╰┤ State 
"""

        assert result == expected


class TestTorchInterface:
    """Unittests for applying the torch interface"""

    def test_import_error(self, mocker):
        """Test that an exception is caught on import error"""
        torch = pytest.importorskip("torch", minversion="1.3")
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

    def test_drawing(self):
        """Test circuit drawing when using the torch interface"""
        torch = pytest.importorskip("torch", minversion="1.3")

        x = torch.tensor(0.1, requires_grad=True)
        y = torch.tensor([0.2, 0.3], requires_grad=True)
        z = torch.tensor(0.4, requires_grad=True)

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, interface="torch")
        def circuit(p1, p2=y, **kwargs):
            qml.RX(p1, wires=0)
            qml.RY(p2[0] * p2[1], wires=1)
            qml.RX(kwargs["p3"], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0), qml.var(qml.PauliZ(1))

        circuit(p1=x, p3=z)

        result = circuit.draw()
        expected = """\
 0: ──RX(0.1)───RX(0.4)──╭C──┤ Probs  
 1: ──RY(0.06)───────────╰X──┤ Var[Z] 
"""

        assert result == expected


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


class TestQNodeCollection:
    """Unittests for the QNodeCollection"""

    def test_multi_thread(self):
        """Test that multi-threaded queuing in tape mode works correctly"""
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
        qml.enable_tape()

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


class TestMutability:
    """Test for QNode immutability"""

    def test_mutable(self, mocker, tol):
        """Test that a QNode which has structure dependent
        on trainable arguments is reconstructed with
        every call, and remains differentiable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, mutable=True)
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
        grad = qml.grad(circuit)(0.5)
        np.testing.assert_allclose(grad, 0, atol=tol, rtol=0)

        grad = qml.grad(circuit)(-0.5)
        np.testing.assert_allclose(grad, -np.sin(-0.5), atol=tol, rtol=0)

    def test_immutable(self, mocker, tol):
        """Test that a QNode which has structure dependent
        on trainable arguments is *not* reconstructed with
        every call when mutable=False"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, mutable=False)
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
        grad = qml.grad(circuit)(-0.5)
        np.testing.assert_allclose(grad, 0, atol=tol, rtol=0)


class TestShots:
    """Unittests for specifying shots per call."""

    def test_specify_shots_per_call(self):
        """Tests that shots can be set per call."""
        dev = qml.device('default.qubit', wires=1, shots=10)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        assert len(circuit(0.8)) == 10
        assert len(circuit(0.8, shots=2)) == 2
        assert len(circuit(0.8, shots=3178)) == 3178
        assert len(circuit(0.8)) == 10

    def test_no_shots_per_call_if_user_has_shots_qfunc_kwarg(self):
        """Tests that the per-call shots overwriting is suspended if user
        has a shots keyword argument, but a warning is raised."""

        dev = qml.device('default.qubit', wires=2, shots=10)

        def circuit(a, shots=0):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=0))

        with pytest.warns(DeprecationWarning, match="The 'shots' argument name is reserved for overriding"):
            circuit = qml.QNode(circuit, dev)

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
        dev = qml.device('default.qubit', wires=[qml.numpy.array(0), qml.numpy.array(1)], shots=10)

        def circuit(a, shots):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=qml.numpy.array(0)))

        # assert that warning is still raised
        with pytest.warns(DeprecationWarning, match="The 'shots' argument name is reserved for overriding"):
            circuit = qml.QNode(circuit, dev)

        assert len(circuit(0.8, 1)) == 10
        assert circuit.qtape.operations[0].wires.labels == (1,)

        dev = qml.device('default.qubit', wires=2, shots=10)

        @qml.qnode(dev)
        def circuit(a, shots):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=0))

        assert len(circuit(0.8, shots=0)) == 10
        assert circuit.qtape.operations[0].wires.labels == (0,)

    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift"])
    def test_shots_setting_does_not_mutate_device(self, diff_method):
        """Tests that per-call shots setting does not change the number of shots in the device."""

        dev = qml.device('default.qubit', wires=1, shots=3)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        assert dev.shots == 3
        res = circuit(0.8, shots=2)
        assert len(res) == 2
        assert dev.shots == 3

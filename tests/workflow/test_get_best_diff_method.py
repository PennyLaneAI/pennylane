# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the `get_best_diff_method` function"""

import pytest

import pennylane as qml
from pennylane.workflow import get_best_diff_method


def dummy_cv_func(x):
    """A dummy CV function with continuous-variable operations."""
    qml.Displacement(x, 0.1, wires=0)
    return qml.expval(qml.X(0))


def dummyfunc():
    """dummy func."""
    return None


# pylint: disable=unused-argument
class CustomDevice(qml.devices.Device):
    """A null device that just returns 0."""

    def __repr__(self):
        return "CustomDevice"

    def execute(self, circuits, execution_config=None):
        return (0,)


class CustomDeviceWithDiffMethod(qml.devices.Device):
    """A device that defines a derivative."""

    def execute(self, circuits, execution_config=None):
        return 0

    def compute_derivatives(self, circuits, execution_config=None):
        """Device defines its own method to compute derivatives"""
        return 0


class TestValidation:
    """Tests for QNode creation and validation"""

    @pytest.mark.jax
    def test_best_method_is_device(self):
        """Test that the method for determining the best diff method
        for a device that is a child of qml.devices.Device and has a
        compute_derivatives method defined returns 'device'"""

        dev = CustomDeviceWithDiffMethod()
        qn_jax = qml.QNode(dummyfunc, dev, "jax")
        qn_none = qml.QNode(dummyfunc, dev)

        res = get_best_diff_method(qn_jax)()
        assert res == "device"

        res = get_best_diff_method(qn_none)()
        assert res == "device"

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["jax", "torch", "autograd"])
    def test_best_method_is_backprop(self, interface):
        """Test that the method for determining the best diff method
        for the default.qubit device and a valid interface returns back-propagation"""

        dev = qml.device("default.qubit", wires=1)
        qn = qml.QNode(dummyfunc, dev, interface)

        # backprop is returned when the interface is an allowed interface for the device and Jacobian is not provided
        res = get_best_diff_method(qn)()
        assert res == "backprop"

    def test_best_method_is_param_shift(self):
        """Test that the method for determining the best diff method
        for a given device and interface returns the parameter shift rule if
        'device' and 'backprop' don't work"""

        # null device has no info - fall back on parameter-shift
        dev = CustomDevice()
        qn = qml.QNode(dummyfunc, dev)

        res = get_best_diff_method(qn)()
        assert res == "parameter-shift"

        # no interface - fall back on parameter-shift
        dev2 = qml.device("default.qubit", wires=1)
        qn = qml.QNode(dummyfunc, dev2)
        res2 = get_best_diff_method(qml.set_shots(qn, shots=50))()
        assert res2 == "parameter-shift"

    def test_best_method_is_param_shift_cv(self):
        """Tests that the method returns 'parameter-shift' when CV operations are in the QNode."""

        dev = qml.device("default.gaussian", wires=1)
        qn = qml.QNode(dummy_cv_func, dev, interface=None)

        res = get_best_diff_method(qn)(0.5)
        assert res == "parameter-shift"

    def test_best_method_with_transforms(self):
        """Test that transforms and execution parameters affect the supported differentiation method."""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        x = qml.numpy.array(0.5)

        original_method = get_best_diff_method(circuit)(x)
        metric_tensor_method = get_best_diff_method(qml.metric_tensor(circuit))(x)

        assert original_method == "adjoint"

        assert metric_tensor_method == "parameter-shift"

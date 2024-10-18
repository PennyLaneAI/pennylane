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
"""Unit tests for the `qml.workflow.get_best_diff_method`"""

import pytest

import pennylane as qml


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


# pylint: disable=too-many-public-methods
class TestValidation:
    """Tests for QNode creation and validation"""

    # pylint: disable=protected-access
    @pytest.mark.autograd
    def test_best_method_is_device(self):
        """Test that the method for determining the best diff method
        for a device that is a child of qml.devices.Device and has a
        compute_derivatives method defined returns 'device'"""

        dev = CustomDeviceWithDiffMethod()
        qn_jax = qml.QNode(dummyfunc, dev, "jax")
        qn_none = qml.QNode(dummyfunc, dev, None)

        res = qml.workflow.get_best_diff_method(qn_jax)()
        assert res == "device"

        res = qml.workflow.get_best_diff_method(qn_none)()
        assert res == "device"

    # pylint: disable=protected-access
    @pytest.mark.parametrize("interface", ["jax", "tensorflow", "torch", "autograd"])
    def test_best_method_is_backprop(self, interface):
        """Test that the method for determining the best diff method
        for the default.qubit device and a valid interface returns backpropagation"""

        dev = qml.device("default.qubit", wires=1)
        qn = qml.QNode(dummyfunc, dev, interface)

        # backprop is returned when the interface is an allowed interface for the device and Jacobian is not provided
        res = qml.workflow.get_best_diff_method(qn)()
        assert res == "backprop"

    # pylint: disable=protected-access
    def test_best_method_is_param_shift(self):
        """Test that the method for determining the best diff method
        for a given device and interface returns the parameter shift rule if
        'device' and 'backprop' don't work"""

        # null device has no info - fall back on parameter-shift
        dev = CustomDevice()
        qn = qml.QNode(dummyfunc, dev)

        res = qml.workflow.get_best_diff_method(qn)()
        assert res == qml.gradients.param_shift

        # no interface - fall back on parameter-shift
        dev2 = qml.device("default.qubit", wires=1)
        qn = qml.QNode(dummyfunc, dev2)
        res2 = qml.workflow.get_best_diff_method(qn)(shots=50)
        assert res2 == qml.gradients.param_shift

    # pylint: disable=protected-access
    def test_best_method_str_is_device(self, monkeypatch):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'device'"""
        dev = qml.device("default.mixed", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", None)
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", True)

        # backprop is returned when the interface is an allowed interface for the device and Jacobian is not provided
        qn = qml.QNode(dummyfunc, dev, interface=None)
        res = qml.workflow.get_best_diff_method(qn)(return_as_str=True)
        assert res == "device"

        # device is returned even if backpropagation is possible
        qn = qml.QNode(dummyfunc, dev, interface=None)
        res = qml.workflow.get_best_diff_method(qn)(return_as_str=True)
        assert res == "device"

    # pylint: disable=protected-access
    def test_best_method_str_is_backprop(self, monkeypatch):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'backprop'"""
        dev = qml.device("default.mixed", wires=1)
        monkeypatch.setitem(dev._capabilities, "passthru_interface", None)
        monkeypatch.setitem(dev._capabilities, "provides_jacobian", False)

        # backprop is returned when the interfaces match and Jacobian is not provided
        qn = qml.QNode(dummyfunc, dev, interface=None)
        res = qml.workflow.get_best_diff_method(qn)(return_as_str=True)
        assert res == "backprop"

    # pylint: disable=protected-access
    def test_best_method_str_is_param_shift(self):
        """Test that the method for determining the best diff method string
        for a given device and interface returns 'parameter-shift'"""
        dev = qml.device("default.mixed", wires=1, shots=50)

        # parameter shift is returned when Jacobian is not provided and
        # the backprop interfaces do not match
        qn = qml.QNode(dummyfunc, dev, interface=None)
        res = qml.workflow.get_best_diff_method(qn)(return_as_str=True)
        assert res == "parameter-shift"

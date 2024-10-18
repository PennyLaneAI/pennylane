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
"""Unit tests for the `qml.workflow.get_gradient_fn`"""

import pytest

import pennylane as qml


# pylint: disable=unused-argument
class CustomDevice(qml.devices.Device):
    """A null device that just returns 0."""

    def __repr__(self):
        return "CustomDevice"

    def execute(self, circuits, execution_config=None):
        return (0,)


# pylint: disable=unused-argument
class DerivativeDevice(qml.devices.Device):
    """A device that says it supports device derivatives."""

    def execute(self, circuits, execution_config=None):
        return 0

    def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
        return execution_config.gradient_method == "device"


# pylint: disable=unused-argument
class BackpropDevice(qml.devices.Device):
    """A device that says it supports backpropagation."""

    def execute(self, circuits, execution_config=None):
        return 0

    def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
        return execution_config.gradient_method == "backprop"


class TestNewDeviceIntegration:
    """Basic tests for integration of the new device interface and the QNode."""

    dev = CustomDevice()

    def test_get_gradient_fn_custom_device(self):
        """Test get_gradient_fn is parameter for best for null device."""
        gradient_fn = qml.workflow.get_gradient_fn(self.dev, "best")
        assert gradient_fn is qml.gradients.param_shift

    def test_get_gradient_fn_default_qubit(self):
        """Tests the get_gradient_fn is backprop for best for default qubit2."""
        dev = qml.devices.DefaultQubit()
        gradient_fn = qml.workflow.get_gradient_fn(dev, "best")
        assert gradient_fn == "backprop"

    def test_get_gradient_fn_custom_dev_adjoint(self):
        """Test that an error is raised if adjoint is requested for a device that does not support it."""
        with pytest.raises(
            qml.QuantumFunctionError, match=r"Device CustomDevice does not support adjoint"
        ):
            qml.workflow.get_gradient_fn(self.dev, "adjoint")

    def test_error_for_backprop_with_custom_device(self):
        """Test that an error is raised when backprop is requested for a device that does not support it."""
        with pytest.raises(
            qml.QuantumFunctionError, match=r"Device CustomDevice does not support backprop"
        ):
            qml.workflow.get_gradient_fn(self.dev, "backprop")

    def test_custom_device_that_supports_backprop(self):
        """Test that a custom device and designate that it supports backprop derivatives."""

        dev = BackpropDevice()
        gradient_fn = qml.workflow.get_gradient_fn(dev, diff_method="backprop")
        assert gradient_fn == "backprop"

    def test_custom_device_with_device_derivative(self):
        """Test that a custom device can specify that it supports device derivatives."""

        dev = DerivativeDevice()
        gradient_fn = qml.workflow.get_gradient_fn(dev, "device")
        assert gradient_fn == "device"

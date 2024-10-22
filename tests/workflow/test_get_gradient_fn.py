# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the `qml.workflow.get_gradient_fn`"""

import pytest

import pennylane as qml
from pennylane.workflow import get_gradient_fn
from pennylane.workflow.construct_batch import expand_fn_transform


def dummyfunc():
    """Dummy function for QNode"""
    return None


# pylint: disable=unused-argument
class CustomDeviceWithDiffMethod(qml.devices.Device):
    """A device that defines its own derivative."""

    def execute(self, circuits, execution_config=None):
        return 0

    def compute_derivatives(self, circuits, execution_config=None):
        """Device defines its own method to compute derivatives"""
        return 0


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


class TestCustomDeviceIntegration:
    """Basic tests for integration of the new device interface and the QNode."""

    dev = CustomDevice()

    def test_get_gradient_fn_custom_device(self):
        """Test get_gradient_fn is parameter for best for null device."""
        gradient_fn = get_gradient_fn(self.dev, "best")
        assert gradient_fn == "parameter-shift"

    def test_get_gradient_fn_custom_dev_adjoint(self):
        """Test that an error is raised if adjoint is requested for a device that does not support it."""
        with pytest.raises(
            qml.QuantumFunctionError, match=r"Device CustomDevice does not support adjoint"
        ):
            get_gradient_fn(self.dev, "adjoint")

    def test_error_for_backprop_with_custom_device(self):
        """Test that an error is raised when backprop is requested for a device that does not support it."""
        with pytest.raises(
            qml.QuantumFunctionError, match=r"Device CustomDevice does not support backprop"
        ):
            get_gradient_fn(self.dev, "backprop")

    def test_custom_device_that_supports_backprop(self):
        """Test that a custom device and designate that it supports backprop derivatives."""
        dev = BackpropDevice()
        gradient_fn = get_gradient_fn(dev, diff_method="backprop")
        assert gradient_fn == "backprop"

    def test_custom_device_with_device_derivative(self):
        """Test that a custom device can specify that it supports device derivatives."""
        dev = DerivativeDevice()
        gradient_fn = get_gradient_fn(dev, "device")
        assert gradient_fn == "device"


class TestGetGradientFn:
    """Tests for the `get_gradient_fn` function"""

    def test_diff_method_is_none(self):
        """Tests whether None is returned if user doesn't provide a diff_method."""
        dev = qml.device("default.qubit", wires=1)
        gradient_fn = get_gradient_fn(dev, diff_method=None)
        assert gradient_fn is None

    def test_error_is_raised_for_invalid_diff_method(self):
        """Tests whether error is raised if diff_method is not valid."""
        dev = qml.device("default.qubit", wires=1)
        with pytest.raises(
            qml.QuantumFunctionError,
            match="Differentiation method 123 must be a gradient transform or a string.",
        ):
            get_gradient_fn(dev, diff_method=123)

    def test_diff_method_is_transform_dispatcher(self):
        """Tests that the diff method is simply returned"""
        dev = qml.device("default.mixed", wires=1)
        transform_dispatcher = expand_fn_transform(dev.expand_fn)
        assert isinstance(transform_dispatcher, qml.transforms.core.TransformDispatcher)
        gradient_fn = get_gradient_fn(dev, diff_method=transform_dispatcher)
        assert gradient_fn is transform_dispatcher

    def test_best_method_backprop(self):
        """Test that get_gradient_fn returns 'backprop' for default.qubit and 'best' diff_method"""
        dev = qml.device("default.qubit", wires=1)
        gradient_fn = get_gradient_fn(dev, diff_method="best")
        assert gradient_fn == "backprop"

    def test_best_method_device_defined(self):
        """Test that get_gradient_fn returns 'device' for a custom device with its own diff method"""
        dev = CustomDeviceWithDiffMethod(wires=1)
        gradient_fn = get_gradient_fn(dev, diff_method="device")
        assert gradient_fn == "device"

    def test_finite_diff_method(self):
        """Test that get_gradient_fn returns 'finite-diff' for the 'finite-diff' method"""
        dev = qml.device("default.qubit", wires=1)
        gradient_fn = get_gradient_fn(dev, diff_method="finite-diff")
        assert gradient_fn == qml.gradients.finite_diff

    def test_spsa_method(self):
        """Test that get_gradient_fn returns 'spsa' for the 'spsa' method"""
        dev = qml.device("default.qubit", wires=1)
        gradient_fn = get_gradient_fn(dev, diff_method="spsa")
        assert gradient_fn == qml.gradients.spsa_grad

    def test_hadamard_method(self):
        """Test that get_gradient_fn returns 'hadamard' for the 'hadamard' method"""
        dev = qml.device("default.qubit", wires=1)
        gradient_fn = get_gradient_fn(dev, diff_method="hadamard")
        assert gradient_fn == qml.gradients.hadamard_grad

    def test_param_shift_method(self):
        """Test that get_gradient_fn returns 'parameter-shift' for the 'parameter-shift' method"""
        dev = qml.device("default.qubit", wires=1)
        gradient_fn = get_gradient_fn(dev, diff_method="parameter-shift")
        assert gradient_fn == qml.gradients.param_shift

    def test_param_shift_method_with_cv_ops(self):
        """Test that get_gradient_fn returns 'parameter-shift-cv' when CV operations are present on tape"""
        dev = qml.device("default.gaussian", wires=1)
        tape = qml.tape.QuantumScript([qml.Displacement(0.5, 0.0, wires=0)])
        gradient_fn = get_gradient_fn(dev, diff_method="parameter-shift", tape=tape)
        assert gradient_fn == qml.gradients.param_shift_cv

    def test_invalid_diff_method(self):
        """Test that get_gradient_fn raises an error for invalid diff method"""
        dev = qml.device("default.qubit", wires=1)
        with pytest.raises(
            qml.QuantumFunctionError, match="Differentiation method invalid-method not recognized"
        ):
            get_gradient_fn(dev, diff_method="invalid-method")

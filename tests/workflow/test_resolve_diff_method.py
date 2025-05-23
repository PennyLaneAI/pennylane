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

"""Unit tests for the `qml.workflow.resolution._resolve_diff_method` helper function"""

import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.exceptions import QuantumFunctionError
from pennylane.workflow import _resolve_diff_method


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


# pylint: disable=unused-argument, too-few-public-methods
class CustomDevice(qml.devices.Device):
    """A null device that just returns 0."""

    def execute(self, circuits, execution_config=None):
        return 0


# pylint: disable=unused-argument
class DerivativeDevice(qml.devices.Device):
    """A device that says it supports device derivatives."""

    def execute(self, circuits, execution_config=None):
        return 0

    def supports_derivatives(self, execution_config=None, circuit=None):
        return execution_config.gradient_method == "device"


# pylint: disable=unused-argument
class BackpropDevice(qml.devices.Device):
    """A device that says it supports backpropagation."""

    def execute(self, circuits, execution_config=None):
        return 0

    def supports_derivatives(self, execution_config=None, circuit=None):
        return execution_config.gradient_method == "backprop"


class TestCustomDeviceIntegration:
    """Basic tests for integration of the new device interface and the QNode."""

    dev = CustomDevice()

    def test_custom_device(self):
        """Test best method for custom device"""
        config = ExecutionConfig(gradient_method="best")
        resolved_config = _resolve_diff_method(config, self.dev)
        assert resolved_config.gradient_method is qml.gradients.param_shift

    def test_gradient_fn_custom_dev_adjoint(self):
        """Test that an error is raised if adjoint is requested for a device that does not support it."""
        config = ExecutionConfig(gradient_method="adjoint")
        with pytest.raises(
            QuantumFunctionError,
            match=r"does not support adjoint with requested circuit",
        ):
            _resolve_diff_method(config, self.dev)

    def test_error_for_backprop_with_custom_device(self):
        """Test that an error is raised when backprop is requested for a device that does not support it."""
        config = ExecutionConfig(gradient_method="backprop")
        with pytest.raises(
            QuantumFunctionError,
            match=r"does not support backprop with requested circuit",
        ):
            _resolve_diff_method(config, self.dev)

    def test_custom_device_that_supports_backprop(self):
        """Test that a custom device and designate that it supports backprop derivatives."""
        dev = BackpropDevice()
        config = ExecutionConfig(gradient_method="backprop")
        resolved_config = _resolve_diff_method(config, dev)
        assert resolved_config.gradient_method == "backprop"

    def test_custom_device_with_device_derivative(self):
        """Test that a custom device can specify that it supports device derivatives."""
        dev = DerivativeDevice()
        config = ExecutionConfig(gradient_method="device")
        resolved_config = _resolve_diff_method(config, dev)
        assert resolved_config.gradient_method == "device"


class TestResolveDiffMethod:
    """Tests for the `_resolve_diff_method` function"""

    def test_diff_method_is_none(self):
        """Test that the configuration remains unchanged if the diff_method is None"""
        dev = qml.device("default.qubit", wires=1)
        initial_config = ExecutionConfig(gradient_method=None)
        resolved_config = _resolve_diff_method(initial_config, dev)
        assert resolved_config == initial_config

    def test_transform_dispatcher_as_diff_method(self):
        """Test when diff_method is of type TransformDispatcher"""
        dev = qml.device("default.mixed", wires=1)
        initial_config = ExecutionConfig(gradient_method=qml.gradients.param_shift)
        resolved_config = _resolve_diff_method(initial_config, dev)
        assert resolved_config.gradient_method is qml.gradients.param_shift

    def test_best_method_backprop(self):
        """Test that 'backprop' is chosen for 'best' diff_method on default.qubit"""
        dev = qml.device("default.qubit", wires=1)
        initial_config = ExecutionConfig(gradient_method="best")
        resolved_config = _resolve_diff_method(initial_config, dev)
        assert resolved_config.gradient_method == "backprop"

    def test_best_method_device_defined(self):
        """Test that 'device' is chosen for a custom device with device-specific gradients"""
        dev = CustomDeviceWithDiffMethod(wires=1)
        initial_config = ExecutionConfig(gradient_method="device")
        resolved_config = _resolve_diff_method(initial_config, dev)
        assert resolved_config.gradient_method == "device"

    def test_finite_diff_method(self):
        """Test that the 'finite-diff' method is correctly resolved"""
        dev = qml.device("default.qubit", wires=1)
        initial_config = ExecutionConfig(gradient_method="finite-diff")
        resolved_config = _resolve_diff_method(initial_config, dev)
        assert resolved_config.gradient_method is qml.gradients.finite_diff

    def test_param_shift_method_with_cv_ops(self):
        """Test that 'parameter-shift-cv' is used when CV operations are present."""
        dev = qml.device("default.gaussian", wires=1)
        tape = qml.tape.QuantumScript([qml.Displacement(0.5, 0.0, wires=0)])
        initial_config = ExecutionConfig(gradient_method="parameter-shift")
        resolved_config = _resolve_diff_method(initial_config, dev, tape=tape)
        assert resolved_config.gradient_method is qml.gradients.param_shift_cv

        dev = qml.device("default.gaussian", wires=1)
        tape = qml.tape.QuantumScript([qml.Identity(wires=0)])
        initial_config = ExecutionConfig(gradient_method="parameter-shift")
        resolved_config = _resolve_diff_method(initial_config, dev, tape=tape)
        assert resolved_config.gradient_method is qml.gradients.param_shift

    def test_custom_device_that_supports_backprop(self):
        """Test that a custom device supports backprop derivatives."""
        dev = BackpropDevice()
        initial_config = ExecutionConfig(gradient_method="backprop")
        resolved_config = _resolve_diff_method(initial_config, dev)
        assert resolved_config.gradient_method == "backprop"

    def test_device_derivative_support(self):
        """Test that a device supporting its own derivatives works correctly."""
        dev = DerivativeDevice()
        initial_config = ExecutionConfig(gradient_method="device")
        resolved_config = _resolve_diff_method(initial_config, dev)
        assert resolved_config.gradient_method == "device"

    def test_invalid_diff_method(self):
        """Test that an invalid diff method raises an error."""
        dev = qml.device("default.qubit", wires=1)
        initial_config = ExecutionConfig(gradient_method="invalid-method")
        with pytest.raises(
            QuantumFunctionError,
            match="Differentiation method invalid-method not recognized",
        ):
            _resolve_diff_method(initial_config, dev)

    @pytest.mark.parametrize(
        "diff_method, mode",
        [
            ("hadamard", "standard"),
            ("reversed-hadamard", "reversed"),
            ("direct-hadamard", "direct"),
            ("reversed-direct-hadamard", "reversed-direct"),
        ],
    )
    def test_hadamard_methods(self, diff_method, mode):
        """Test that we can resolve the hadamard methods correctly."""

        dev = qml.device("default.qubit")
        initial_config = ExecutionConfig(gradient_method=diff_method)
        processed = _resolve_diff_method(initial_config, dev)
        assert processed.gradient_method == qml.gradients.hadamard_grad
        assert processed.gradient_keyword_arguments == {"mode": mode, "device_wires": None}

    def test_hadamard_device_wires(self):
        """Test that device wires are added to the gradient keyword args."""

        dev = qml.device("default.qubit", wires=("a", "b"))
        initial_config = ExecutionConfig(gradient_method="hadamard")
        processed = _resolve_diff_method(initial_config, dev)
        assert processed.gradient_method == qml.gradients.hadamard_grad
        assert processed.gradient_keyword_arguments == {
            "mode": "standard",
            "device_wires": dev.wires,
        }

    @pytest.mark.parametrize(
        "diff_method", ("reversed-hadamard", "direct-hadamard", "reversed-direct-hadamard")
    )
    def test_error_if_specific_hadamard_variant_and_mode(self, diff_method):

        dev = qml.device("default.qubit")

        initial_config = ExecutionConfig(
            gradient_method=diff_method, gradient_keyword_arguments={"mode": "reversed"}
        )
        with pytest.raises(ValueError, match="cannot be provided with a 'mode'"):
            _resolve_diff_method(initial_config, dev)

    def test_specify_hadamard_and_mode(self):
        dev = qml.device("default.qubit")
        initial_config = ExecutionConfig(
            gradient_method="hadamard", gradient_keyword_arguments={"mode": "reversed"}
        )
        processed = _resolve_diff_method(initial_config, dev)
        assert processed.gradient_method == qml.gradients.hadamard_grad
        assert processed.gradient_keyword_arguments == {"mode": "reversed", "device_wires": None}

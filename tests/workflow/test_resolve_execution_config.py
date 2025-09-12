# Copyright 2018-2024 Xanadu Quantum Technologies Inc.
#
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

"""Unit tests for the `_resolve_execution_config` helper function in PennyLane."""
# pylint: disable=redefined-outer-name
import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig, MCMConfig
from pennylane.exceptions import QuantumFunctionError
from pennylane.workflow.resolution import _resolve_execution_config


def test_resolve_execution_config_with_gradient_method():
    """Test resolving an ExecutionConfig with a specified gradient method."""
    execution_config = ExecutionConfig(gradient_method="best", interface=None)
    device = qml.device("default.qubit")

    empty_tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])

    resolved_config = _resolve_execution_config(execution_config, device, [empty_tape])

    assert resolved_config.gradient_method == "backprop"


def test_metric_tensor_lightning_edge_case():
    """Test resolving an ExecutionConfig with the metric tensor transform on a lightning device."""
    execution_config = ExecutionConfig(
        gradient_method="best",
    )
    device = qml.device("lightning.qubit", wires=2)

    empty_tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])

    resolved_config = _resolve_execution_config(execution_config, device, [empty_tape])

    assert resolved_config.gradient_method == "adjoint"


def test_param_shift_cv_kwargs():
    """Test resolving an ExecutionConfig with parameter-shift and validating gradient keyword arguments."""
    dev = qml.device("default.gaussian", wires=1)
    tape = qml.tape.QuantumScript([qml.Displacement(0.5, 0.0, wires=0)])
    execution_config = ExecutionConfig(gradient_method="parameter-shift")

    resolved_config = _resolve_execution_config(execution_config, dev, [tape])

    assert resolved_config.gradient_keyword_arguments["dev"] == dev


def test_mcm_config_validation():
    """Test validation of MCMConfig within an ExecutionConfig."""
    mcm_config = MCMConfig(postselect_mode="hw-like")
    execution_config = ExecutionConfig(mcm_config=mcm_config, interface=None)
    device = qml.device("default.qubit")

    empty_tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])

    resolved_config = _resolve_execution_config(execution_config, device, [empty_tape])

    expected_mcm_config = MCMConfig(mcm_method="deferred", postselect_mode=None)

    assert resolved_config.mcm_config == expected_mcm_config


@pytest.mark.parametrize("mcm_method", [None, "one-shot"])
@pytest.mark.parametrize("postselect_mode", [None, "hw-like"])
@pytest.mark.jax
def test_jax_interface(mcm_method, postselect_mode):
    """Test resolving ExecutionConfig with JAX interface and different MCMConfig settings."""
    mcm_config = MCMConfig(mcm_method=mcm_method, postselect_mode=postselect_mode)
    execution_config = ExecutionConfig(mcm_config=mcm_config, interface="jax")
    device = qml.device("default.qubit")

    tape_with_finite_shots = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=100)

    resolved_config = _resolve_execution_config(execution_config, device, [tape_with_finite_shots])

    # since finite shots, mcm_method always one-shot
    expected_mcm_config = MCMConfig("one-shot", postselect_mode="pad-invalid-samples")

    assert resolved_config.mcm_config == expected_mcm_config


@pytest.mark.jax
def test_jax_jit_interface():
    """Test resolving ExecutionConfig with JAX-JIT interface and deferred MCMConfig method."""
    mcm_config = MCMConfig(mcm_method="deferred")
    execution_config = ExecutionConfig(mcm_config=mcm_config, interface="jax-jit")
    device = qml.device("default.qubit")

    empty_tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])

    resolved_config = _resolve_execution_config(execution_config, device, [empty_tape])

    expected_mcm_config = MCMConfig(mcm_method="deferred", postselect_mode="fill-shots")

    assert resolved_config.mcm_config == expected_mcm_config


# pylint: disable=unused-argument
def test_no_device_vjp_if_not_supported():
    """Test that an error is raised for device_vjp=True if the device does not support it."""

    class DummyDev(qml.devices.Device):

        def execute(self, circuits, execution_config=qml.devices.ExecutionConfig()):
            return 0

        def supports_derivatives(self, execution_config=None, circuit=None):
            return execution_config and execution_config.gradient_method == "vjp_grad"

        def supports_vjp(self, execution_config=None, circuit=None) -> bool:
            return execution_config and execution_config.gradient_method == "vjp_grad"

    config_vjp_grad = ExecutionConfig(use_device_jacobian_product=True, gradient_method="vjp_grad")
    tape = qml.tape.QuantumScript()
    # no error
    _ = _resolve_execution_config(config_vjp_grad, DummyDev(), (tape,))

    config_parameter_shift = ExecutionConfig(
        use_device_jacobian_product=True, gradient_method="parameter-shift"
    )
    with pytest.raises(QuantumFunctionError, match="device_vjp=True is not supported"):
        _resolve_execution_config(config_parameter_shift, DummyDev(), (tape,))

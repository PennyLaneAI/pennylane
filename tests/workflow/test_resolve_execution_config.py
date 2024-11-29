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
from pennylane.transforms.core import TransformProgram
from pennylane.workflow.resolution import _resolve_execution_config


def test_resolve_execution_config_with_gradient_method():
    """Test resolving an ExecutionConfig with a specified gradient method."""
    execution_config = ExecutionConfig(gradient_method="best", interface=None)
    device = qml.device("default.qubit")

    empty_tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])
    empty_tp = TransformProgram()

    resolved_config = _resolve_execution_config(execution_config, device, [empty_tape], empty_tp)

    assert resolved_config.gradient_method == "backprop"


def test_metric_tensor_lightning_edge_case():
    """Test resolving an ExecutionConfig with the metric tensor transform on a lightning device."""
    execution_config = ExecutionConfig(
        gradient_method="best",
    )
    device = qml.device("lightning.qubit", wires=2)

    empty_tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])
    metric_tensor_tp = TransformProgram([qml.metric_tensor])

    resolved_config = _resolve_execution_config(
        execution_config, device, [empty_tape], metric_tensor_tp
    )

    assert resolved_config.gradient_method is qml.gradients.param_shift


def test_param_shift_cv_kwargs():
    """Test resolving an ExecutionConfig with parameter-shift and validating gradient keyword arguments."""
    dev = qml.device("default.gaussian", wires=1)
    tape = qml.tape.QuantumScript([qml.Displacement(0.5, 0.0, wires=0)])
    execution_config = ExecutionConfig(gradient_method="parameter-shift")
    empty_tp = TransformProgram()

    resolved_config = _resolve_execution_config(execution_config, dev, [tape], empty_tp)

    assert resolved_config.gradient_keyword_arguments["dev"] == dev


def test_mcm_config_validation():
    """Test validation of MCMConfig within an ExecutionConfig."""
    mcm_config = MCMConfig(postselect_mode="hw-like")
    execution_config = ExecutionConfig(mcm_config=mcm_config, interface=None)
    device = qml.device("default.qubit")

    empty_tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])
    empty_tp = TransformProgram()

    resolved_config = _resolve_execution_config(execution_config, device, [empty_tape], empty_tp)

    expected_mcm_config = MCMConfig(postselect_mode=None)

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
    empty_tp = TransformProgram()

    resolved_config = _resolve_execution_config(
        execution_config, device, [tape_with_finite_shots], empty_tp
    )

    expected_mcm_config = MCMConfig(mcm_method, postselect_mode="pad-invalid-samples")

    assert resolved_config.mcm_config == expected_mcm_config


@pytest.mark.jax
def test_jax_jit_interface():
    """Test resolving ExecutionConfig with JAX-JIT interface and deferred MCMConfig method."""
    mcm_config = MCMConfig(mcm_method="deferred")
    execution_config = ExecutionConfig(mcm_config=mcm_config, interface="jax-jit")
    device = qml.device("default.qubit")

    empty_tape = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])
    empty_tp = TransformProgram()

    resolved_config = _resolve_execution_config(execution_config, device, [empty_tape], empty_tp)

    expected_mcm_config = MCMConfig(mcm_method="deferred", postselect_mode="fill-shots")

    assert resolved_config.mcm_config == expected_mcm_config

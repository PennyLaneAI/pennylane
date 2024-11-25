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


@pytest.fixture
def mock_lightning_device():
    """Fixture for creating a mock device with 'lightning' in its name."""
    return qml.device("lightning.qubit", wires=2)


@pytest.fixture
def mock_device():
    """Fixture for creating a generic mock device."""
    return qml.device("default.qubit")


@pytest.fixture
def mock_tape_with_finite_shots():
    """Fixture for creating a batch of quantum tapes with finite shots."""
    return [qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=100)]


@pytest.fixture
def mock_empty_tape():
    """Fixture for creating a batch of empty quantum tapes."""
    return [qml.tape.QuantumScript([], [qml.expval(qml.Z(0))])]


@pytest.fixture
def mock_metric_tensor_tp():
    """Fixture for creating a transform program with the metric tensor transform."""
    return TransformProgram([qml.metric_tensor])


@pytest.fixture
def mock_empty_tp():
    """Fixture for creating an empty transform program."""
    return TransformProgram()


def test_resolve_execution_config_with_gradient_method(mock_device, mock_empty_tape, mock_empty_tp):
    """Test resolving an ExecutionConfig with a specified gradient method."""
    execution_config = ExecutionConfig(gradient_method="best", interface=None)

    resolved_config = _resolve_execution_config(
        execution_config, mock_device, mock_empty_tape, mock_empty_tp
    )

    assert resolved_config.gradient_method == "backprop"


def test_metric_tensor_lightning_edge_case(
    mock_lightning_device, mock_empty_tape, mock_metric_tensor_tp
):
    """Test resolving an ExecutionConfig with the metric tensor transform on a lightning device."""
    execution_config = ExecutionConfig(
        gradient_method="best",
    )

    resolved_config = _resolve_execution_config(
        execution_config, mock_lightning_device, mock_empty_tape, mock_metric_tensor_tp
    )

    assert resolved_config.gradient_method is qml.gradients.param_shift


def test_param_shift_cv_kwargs(mock_empty_tp):
    """Test resolving an ExecutionConfig with parameter-shift and validating gradient keyword arguments."""
    dev = qml.device("default.gaussian", wires=1)
    tape = qml.tape.QuantumScript([qml.Displacement(0.5, 0.0, wires=0)])
    execution_config = ExecutionConfig(gradient_method="parameter-shift")

    resolved_config = _resolve_execution_config(execution_config, dev, [tape], mock_empty_tp)

    assert resolved_config.gradient_keyword_arguments["dev"] == dev


def test_mcm_config_validation(mock_device, mock_empty_tape, mock_empty_tp):
    """Test validation of MCMConfig within an ExecutionConfig."""
    mcm_config = MCMConfig(postselect_mode="hw-like")
    execution_config = ExecutionConfig(mcm_config=mcm_config, interface=None)
    resolved_config = _resolve_execution_config(
        execution_config, mock_device, mock_empty_tape, mock_empty_tp
    )

    expected_mcm_config = MCMConfig(postselect_mode=None)

    assert resolved_config.mcm_config == expected_mcm_config


@pytest.mark.parametrize("mcm_method", [None, "one-shot"])
@pytest.mark.parametrize("postselect_mode", [None, "hw-like"])
@pytest.mark.jax
def test_jax_interface(
    mcm_method, postselect_mode, mock_device, mock_tape_with_finite_shots, mock_empty_tp
):
    """Test resolving ExecutionConfig with JAX interface and different MCMConfig settings."""
    mcm_config = MCMConfig(mcm_method=mcm_method, postselect_mode=postselect_mode)
    execution_config = ExecutionConfig(mcm_config=mcm_config, interface="jax")
    resolved_config = _resolve_execution_config(
        execution_config, mock_device, mock_tape_with_finite_shots, mock_empty_tp
    )

    expected_mcm_config = MCMConfig(mcm_method, postselect_mode="pad-invalid-samples")

    assert resolved_config.mcm_config == expected_mcm_config


@pytest.mark.jax
def test_jax_jit_interface(mock_device, mock_empty_tape, mock_empty_tp):
    """Test resolving ExecutionConfig with JAX-JIT interface and deferred MCMConfig method."""
    mcm_config = MCMConfig(mcm_method="deferred")
    execution_config = ExecutionConfig(mcm_config=mcm_config, interface="jax-jit")
    resolved_config = _resolve_execution_config(
        execution_config, mock_device, mock_empty_tape, mock_empty_tp
    )

    expected_mcm_config = MCMConfig(mcm_method="deferred", postselect_mode="fill-shots")

    assert resolved_config.mcm_config == expected_mcm_config

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
"""Unit tests for the `qml.workflow.resolution._setup_transform_program` helper function"""

from dataclasses import replace
from unittest.mock import MagicMock

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.transforms.core import TransformProgram
from pennylane.workflow._setup_transform_program import (
    _prune_dynamic_transform,
    _setup_transform_program,
)


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


def mock_user_transform(tape):
    """Mock user transform function"""
    return [tape], null_postprocessing


@qml.transform
def device_transform(tape):
    """Mock user transform function"""
    return [tape], null_postprocessing


def test_gradient_expand_transform():
    """Test if gradient expand transform is added to the full_transform_program."""
    config = ExecutionConfig(gradient_method=qml.gradients.param_shift)

    device = qml.device("default.qubit")

    full_tp, _ = _setup_transform_program(device, config)

    assert repr(full_tp) == "TransformProgram(_expand_transform_param_shift)"


def test_device_transform_program():
    """Test that the device transform is correctly placed in the transform program."""
    config = ExecutionConfig(use_device_gradient=True)

    container = qml.transforms.core.TransformContainer(device_transform)
    device_tp = qml.transforms.core.TransformProgram((container,))
    device = qml.device("default.qubit")
    device.preprocess_transforms = MagicMock(return_value=device_tp)

    full_tp, inner_tp = _setup_transform_program(device, config)

    assert repr(full_tp) == "TransformProgram(device_transform)"
    assert inner_tp.is_empty()

    config = replace(config, use_device_gradient=False)

    full_tp, inner_tp = _setup_transform_program(device, config)

    assert full_tp.is_empty()
    assert repr(inner_tp) == "TransformProgram(device_transform)"


def test_prune_dynamic_transform():
    """Tests that the helper function prune dynamic transform works."""

    program1 = qml.transforms.core.TransformProgram(
        [
            qml.transforms.dynamic_one_shot,
            qml.transforms.split_non_commuting,
            qml.transforms.dynamic_one_shot,
        ]
    )
    program2 = qml.transforms.core.TransformProgram(
        [
            qml.transforms.dynamic_one_shot,
            qml.transforms.split_non_commuting,
        ]
    )

    _prune_dynamic_transform(program1, program2)
    assert len(program1) == 1
    assert len(program2) == 2


def test_prune_dynamic_transform_with_mcm():
    """Tests that the helper function prune dynamic transform works with mcm"""

    program1 = qml.transforms.core.TransformProgram(
        [
            qml.transforms.dynamic_one_shot,
            qml.transforms.split_non_commuting,
            qml.devices.preprocess.mid_circuit_measurements,
        ]
    )
    program2 = qml.transforms.core.TransformProgram(
        [
            qml.transforms.dynamic_one_shot,
            qml.transforms.split_non_commuting,
        ]
    )

    _prune_dynamic_transform(program1, program2)
    assert len(program1) == 2
    assert qml.devices.preprocess.mid_circuit_measurements in program1
    assert len(program2) == 1


def test_interface_data_not_supported():
    """Test that convert_to_numpy_parameters transform is correctly added."""
    config = ExecutionConfig(interface="autograd", gradient_method="adjoint")
    device = qml.device("default.qubit")

    full_tp, inner_tp = _setup_transform_program(device, config)

    assert full_tp.is_empty()
    assert qml.transforms.convert_to_numpy_parameters in inner_tp


def test_interface_data_supported():
    """Test that convert_to_numpy_parameters transform is not added for these cases."""
    config = ExecutionConfig(interface="autograd", gradient_method="backprop")

    device = qml.device("default.mixed", wires=1)

    _, inner_tp = _setup_transform_program(device, config)

    assert qml.transforms.convert_to_numpy_parameters not in inner_tp

    config = ExecutionConfig(interface="autograd", gradient_method="backprop")

    device = qml.device("default.qubit")

    _, inner_tp = _setup_transform_program(device, config)

    assert qml.transforms.convert_to_numpy_parameters not in inner_tp

    config = ExecutionConfig(interface=None, gradient_method="backprop")

    device = qml.device("default.qubit")

    _, inner_tp = _setup_transform_program(device, config)

    assert qml.transforms.convert_to_numpy_parameters not in inner_tp

    config = ExecutionConfig(
        convert_to_numpy=False, interface="jax", gradient_method=qml.gradients.param_shift
    )

    _, inner_tp = _setup_transform_program(device, config)
    assert qml.transforms.convert_to_numpy_parameters not in inner_tp


def test_cache_handling():
    """Test that caching is handled correctly."""
    config = ExecutionConfig()
    device = qml.device("default.qubit")
    device.preprocess_transforms = MagicMock(return_value=TransformProgram())

    full_tp, inner_tp = _setup_transform_program(device, config, cache=True)

    assert repr(inner_tp) == "TransformProgram(_cache_transform)"
    assert full_tp.is_empty()

    full_tp, inner_tp = _setup_transform_program(device, config, cache=False)

    assert full_tp.is_empty()
    assert inner_tp.is_empty()

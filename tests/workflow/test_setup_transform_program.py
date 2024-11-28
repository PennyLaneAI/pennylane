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
# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock

import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.transforms.core import TransformProgram
from pennylane.workflow._setup_transform_program import (
    _prune_dynamic_transform,
    _setup_transform_program,
)


@pytest.fixture
def mock_device():
    """Creates a mock device."""
    device = MagicMock()
    device.short_name = "mock_device"
    return device


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


def mock_expand_transform(tape):
    """Mock expand transform function"""
    return [tape], null_postprocessing


def mock_user_transform(tape):
    """Mock user transform function"""
    return [tape], null_postprocessing


def mock_device_transform(tape):
    """Mock user transform function"""
    return [tape], null_postprocessing


def test_gradient_expand_transform(mock_device):
    """Test if gradient expand transform is added to the full_transform_program."""
    config = ExecutionConfig()
    config.gradient_method = MagicMock(expand_transform=mock_expand_transform)

    mock_device.preprocess_transforms_transforms = MagicMock(return_value=TransformProgram())

    container = qml.transforms.core.TransformContainer(mock_user_transform)
    user_tp = qml.transforms.core.TransformProgram((container,))
    full_tp, inner_tp = _setup_transform_program(user_tp, mock_device, config)

    assert len(full_tp) == 2
    assert repr(full_tp) == "TransformProgram(mock_user_transform, mock_expand_transform)"
    assert inner_tp.is_empty()  # Should not add anything to inner program


def test_device_transform_program(mock_device):
    """Test that the device transform is correctly placed in the transform program."""
    config = ExecutionConfig()
    config.use_device_gradient = True

    container = qml.transforms.core.TransformContainer(mock_device_transform)
    device_tp = qml.transforms.core.TransformProgram((container,))
    mock_device.preprocess_transforms = MagicMock(return_value=device_tp)

    user_transform_program = TransformProgram()
    full_tp, inner_tp = _setup_transform_program(user_transform_program, mock_device, config)

    assert len(full_tp) == 1
    assert repr(full_tp) == "TransformProgram(mock_device_transform)"

    assert inner_tp.is_empty()  # Should not add anything to inner program

    config.use_device_gradient = False

    full_tp, inner_tp = _setup_transform_program(user_transform_program, mock_device, config)

    assert full_tp.is_empty()  # Should not add anything to outer program

    assert len(inner_tp) == 1
    assert repr(inner_tp) == "TransformProgram(mock_device_transform)"


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


def test_interface_data_not_supported(mock_device):
    """Test that convert_to_numpy_parameters transform is correctly added."""
    config = ExecutionConfig()
    config.interface = "autograd"
    config.gradient_method = "adjoint"

    user_transform_program = TransformProgram()
    full_tp, inner_tp = _setup_transform_program(user_transform_program, mock_device, config)

    assert len(inner_tp) == 1
    assert repr(inner_tp) == "TransformProgram(convert_to_numpy_parameters)"

    assert full_tp.is_empty()


def test_interface_data_supported(mock_device):
    """Test that convert_to_numpy_parameters transform is not added for these cases."""
    config = ExecutionConfig()

    config.interface = None
    config.gradient_method = "adjoint"
    mock_device.short_name = "mock_device"

    user_transform_program = TransformProgram()
    full_tp, inner_tp = _setup_transform_program(user_transform_program, mock_device, config)

    assert inner_tp.is_empty()
    assert full_tp.is_empty()

    config = ExecutionConfig()

    config.interface = "autograd"
    config.gradient_method = "backprop"
    mock_device.short_name = "mock_device"

    user_transform_program = TransformProgram()
    full_tp, inner_tp = _setup_transform_program(user_transform_program, mock_device, config)

    assert inner_tp.is_empty()
    assert full_tp.is_empty()

    config = ExecutionConfig()

    config.interface = "autograd"
    config.gradient_method = None
    mock_device.short_name = "default.mixed"

    user_transform_program = TransformProgram()
    full_tp, inner_tp = _setup_transform_program(user_transform_program, mock_device, config)

    assert inner_tp.is_empty()
    assert full_tp.is_empty()


def test_cache_handling(mock_device):
    """Test that caching is handled correctly."""
    config = ExecutionConfig()
    mock_device.preprocess_transforms = MagicMock(return_value=TransformProgram())

    user_transform_program = TransformProgram()
    full_tp, inner_tp = _setup_transform_program(
        user_transform_program, mock_device, config, cache=True
    )

    assert len(inner_tp) == 1
    assert repr(inner_tp) == "TransformProgram(_cache_transform)"

    assert full_tp.is_empty()

    full_tp, inner_tp = _setup_transform_program(
        user_transform_program, mock_device, config, cache=False
    )

    assert full_tp.is_empty()
    assert inner_tp.is_empty()

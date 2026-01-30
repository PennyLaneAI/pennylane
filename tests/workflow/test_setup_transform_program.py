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
from pennylane.transforms.core import CompilePipeline
from pennylane.workflow._setup_transform_program import _setup_transform_program


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

    assert repr(full_tp) == "CompilePipeline(_expand_transform_param_shift)"


def test_device_transform_program():
    """Test that the device transform is correctly placed in the transform program."""
    config = ExecutionConfig(use_device_gradient=True)

    container = qml.transforms.core.BoundTransform(device_transform)
    device_tp = qml.CompilePipeline(container)
    device = qml.device("default.qubit")
    device.preprocess_transforms = MagicMock(return_value=device_tp)

    full_tp, inner_tp = _setup_transform_program(device, config)

    assert repr(full_tp) == "CompilePipeline(device_transform)"
    assert not inner_tp

    config = replace(config, use_device_gradient=False)

    full_tp, inner_tp = _setup_transform_program(device, config)

    assert not full_tp
    assert repr(inner_tp) == "CompilePipeline(device_transform)"


def test_interface_data_not_supported():
    """Test that convert_to_numpy_parameters transform is correctly added."""
    config = ExecutionConfig(interface="autograd", gradient_method="adjoint")
    device = qml.device("default.qubit")

    full_tp, inner_tp = _setup_transform_program(device, config)

    assert not full_tp
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
    device.preprocess_transforms = MagicMock(return_value=CompilePipeline())

    full_tp, inner_tp = _setup_transform_program(device, config, cache=True)

    assert repr(inner_tp) == "CompilePipeline(_cache_transform)"
    assert not full_tp

    full_tp, inner_tp = _setup_transform_program(device, config, cache=False)

    assert not full_tp
    assert not inner_tp

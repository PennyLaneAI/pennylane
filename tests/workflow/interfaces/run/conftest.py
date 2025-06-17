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
"""
Pytest configuration file for the test files in tests/workflow/interfaces/run.
"""
from dataclasses import replace

from param_shift_dev import ParamShiftDerivativesDevice

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.measurements import Shots


def atol_for_shots(shots):
    """Return higher tolerance if finite shots."""
    return 1e-2 if shots else 1e-6


def get_device(device_name, seed):
    if device_name == "param_shift.qubit":
        return ParamShiftDerivativesDevice(seed=seed)
    return qml.device(device_name, seed=seed)


test_matrix = [
    # 0
    [
        "default.qubit",
        replace(
            ExecutionConfig(),
            gradient_method=qml.gradients.param_shift,
        ),
        Shots((100000, 100000)),
    ],
    # 1
    [
        "default.qubit",
        replace(
            ExecutionConfig(),
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(100000),
    ],
    # 2
    [
        "default.qubit",
        replace(
            ExecutionConfig(),
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(None),
    ],
    # 3
    [
        "default.qubit",
        replace(
            ExecutionConfig(),
            gradient_method="backprop",
        ),
        Shots(None),
    ],
    # 4
    [
        "default.qubit",
        replace(
            ExecutionConfig(),
            gradient_method="adjoint",
            use_device_jacobian_product=True,
        ),
        Shots(None),
    ],
    # 5
    [
        "default.qubit",
        replace(
            ExecutionConfig(),
            gradient_method="adjoint",
        ),
        Shots(None),
    ],
    # 6
    [
        "reference.qubit",
        replace(
            ExecutionConfig(),
            gradient_method=qml.gradients.param_shift,
        ),
        Shots((100000, 100000)),
    ],
    # 7
    [
        "reference.qubit",
        replace(
            ExecutionConfig(),
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(100000),
    ],
    # 8
    [
        "reference.qubit",
        replace(
            ExecutionConfig(),
            gradient_method=qml.gradients.param_shift,
        ),
        Shots(None),
    ],
    # 9
    [
        "param_shift.qubit",
        replace(
            ExecutionConfig(),
            gradient_method="device",
            use_device_jacobian_product=False,
        ),
        Shots((100000, 100000)),
    ],
    # 10
    [
        "param_shift.qubit",
        replace(
            ExecutionConfig(),
            gradient_method="device",
            use_device_jacobian_product=False,
        ),
        Shots(100000),
    ],
    # 11
    [
        "param_shift.qubit",
        replace(
            ExecutionConfig(),
            gradient_method="device",
            use_device_jacobian_product=False,
        ),
        Shots(None),
    ],
]

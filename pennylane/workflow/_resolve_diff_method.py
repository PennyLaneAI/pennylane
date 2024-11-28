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
"""Contains a function for resolving the differentiation method in an initial execution config based on device and tape information.

"""

from dataclasses import replace
from typing import Literal, get_args

import pennylane as qml
from pennylane.logging import debug_logger
from pennylane.transforms.core import TransformDispatcher

SupportedDiffMethods = Literal[
    None,
    "best",
    "device",
    "backprop",
    "adjoint",
    "parameter-shift",
    "hadamard",
    "finite-diff",
    "spsa",
]


@debug_logger
def _resolve_diff_method(
    initial_config: "qml.devices.ExecutionConfig",
    device: "qml.devices.Device",
    tape: "qml.tape.QuantumTape" = None,
) -> "qml.devices.ExecutionConfig":
    """
    Resolves the differentiation method and updates the initial execution configuration accordingly.

    Args:
        initial_config (qml.devices.ExecutionConfig): The initial execution configuration.
        device (qml.devices.Device): A PennyLane device.
        tape (Optional[qml.tape.QuantumTape]): The circuit that will be differentiated. Should include shots information.

    Returns:
        qml.devices.ExecutionConfig: Updated execution configuration with the resolved differentiation method.
    """
    diff_method = initial_config.gradient_method
    updated_values = {"gradient_method": diff_method}

    if diff_method is None:
        return initial_config

    if device.supports_derivatives(initial_config, circuit=tape):
        new_config = device.setup_execution_config(initial_config)
        return new_config

    if diff_method in {"backprop", "adjoint", "device"}:
        raise qml.QuantumFunctionError(
            f"Device {device} does not support {diff_method} with requested circuit."
        )

    if diff_method in {"best", "parameter-shift"}:
        if tape and any(isinstance(op, qml.operation.CV) and op.name != "Identity" for op in tape):
            updated_values["gradient_method"] = qml.gradients.param_shift_cv
        else:
            updated_values["gradient_method"] = qml.gradients.param_shift

    else:
        gradient_transform_map = {
            "finite-diff": qml.gradients.finite_diff,
            "spsa": qml.gradients.spsa_grad,
            "hadamard": qml.gradients.hadamard_grad,
        }

        if diff_method in gradient_transform_map:
            updated_values["gradient_method"] = gradient_transform_map[diff_method]
        elif isinstance(diff_method, TransformDispatcher):
            updated_values["gradient_method"] = diff_method
        else:
            raise qml.QuantumFunctionError(
                f"Differentiation method {diff_method} not recognized. Allowed "
                f"options are {tuple(get_args(SupportedDiffMethods))}."
            )

    return replace(initial_config, **updated_values)

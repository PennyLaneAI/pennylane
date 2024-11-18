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
"""Contains a function for retrieving the gradient function for a given device or tape.

"""

from typing import get_args

import pennylane as qml
from pennylane.logging import debug_logger
from pennylane.transforms.core import TransformDispatcher
from pennylane.workflow.qnode import (
    SupportedDeviceAPIs,
    SupportedDiffMethods,
    _make_execution_config,
)


# pylint: disable=too-many-return-statements, unsupported-binary-operation, inconsistent-return-statements
@debug_logger
def _get_gradient_fn(
    device: SupportedDeviceAPIs,
    diff_method: "TransformDispatcher | SupportedDiffMethods" = "best",
    tape: "qml.tape.QuantumTape" = None,
):
    """Determines the differentiation method for a given device and diff method.

    Args:
        device (:class:`~.devices.Device`): PennyLane device
        diff_method (str or :class:`~.TransformDispatcher`): The requested method of differentiation. Defaults to ``"best"``.
            If a string, allowed options are ``"best"``, ``"backprop"``, ``"adjoint"``,
            ``"device"``, ``"parameter-shift"``, ``"hadamard"``, ``"finite-diff"``, or ``"spsa"``.
            Alternatively, a gradient transform can be provided.
        tape (Optional[.QuantumTape]): the circuit that will be differentiated. Should include shots information.

    Returns:
        str or :class:`~.TransformDispatcher` (the ``gradient_fn``)
    """

    if diff_method is None:
        return None

    config = _make_execution_config(None, diff_method)

    if device.supports_derivatives(config, circuit=tape):
        new_config = device.preprocess(config)[1]
        return new_config.gradient_method

    if diff_method in {"backprop", "adjoint", "device"}:  # device-only derivatives
        raise qml.QuantumFunctionError(
            f"Device {device} does not support {diff_method} with requested circuit."
        )

    if diff_method == "best":
        if tape and any(isinstance(o, qml.operation.CV) for o in tape):
            return qml.gradients.param_shift_cv

        return qml.gradients.param_shift

    if diff_method == "parameter-shift":
        if tape and any(isinstance(o, qml.operation.CV) and o.name != "Identity" for o in tape):
            return qml.gradients.param_shift_cv
        return qml.gradients.param_shift

    gradient_transform_map = {
        "finite-diff": qml.gradients.finite_diff,
        "spsa": qml.gradients.spsa_grad,
        "hadamard": qml.gradients.hadamard_grad,
    }

    if diff_method in gradient_transform_map:
        return gradient_transform_map[diff_method]

    if isinstance(diff_method, TransformDispatcher):
        return diff_method

    raise qml.QuantumFunctionError(
        f"Differentiation method {diff_method} not recognized. Allowed "
        f"options are {tuple(get_args(SupportedDiffMethods))}."
    )

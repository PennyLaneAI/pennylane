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
"""Contains a function for getting the gradient function for a given device or tape.

"""

from typing import Literal, Optional, Union, get_args

import pennylane as qml
from pennylane.transforms.core import TransformDispatcher
from pennylane.workflow.qnode import _make_execution_config

SupportedDeviceAPIs = Union["qml.devices.LegacyDevice", "qml.devices.Device"]

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


# pylint: disable=too-many-return-statements
def get_gradient_fn(
    device: SupportedDeviceAPIs,
    diff_method: Union[TransformDispatcher, SupportedDiffMethods] = "best",
    tape: Optional["qml.tape.QuantumTape"] = None,
):
    """Determine the best differentiation method, interface, and device
    for a requested device and diff method.

    Args:
        device (.device.Device): PennyLane device
        diff_method (str or .TransformDispatcher): The requested method of differentiation.
            If a string, allowed options are ``"best"``, ``"backprop"``, ``"adjoint"``,
            ``"device"``, ``"parameter-shift"``, ``"hadamard"``, ``"finite-diff"``, or ``"spsa"``.
            A gradient transform may also be passed here.
        tape (Optional[.QuantumTape]): the circuit that will be differentiated. Should include shots information.

    Returns:
        tuple[str or .TransformDispatcher, dict, .device.Device: Tuple containing the ``gradient_fn``,
        ``gradient_kwargs``, and the device to use when calling the execute function.
    """

    if diff_method is None:
        return None, {}, device

    config = _make_execution_config(None, diff_method)

    if device.supports_derivatives(config, circuit=tape):
        new_config = device.preprocess(config)[1]
        return new_config.gradient_method, {}, device

    if diff_method in {"backprop", "adjoint", "device"}:  # device-only derivatives
        raise qml.QuantumFunctionError(
            f"Device {device} does not support {diff_method} with requested circuit."
        )

    if diff_method == "best":
        qn = qml.QNode(lambda: None, device, diff_method=None)
        # pylint: disable=protected-access
        qn._tape = tape
        return qml.workflow.get_best_diff_method(qn)()

    if diff_method == "parameter-shift":
        if tape and any(isinstance(o, qml.operation.CV) and o.name != "Identity" for o in tape):
            return qml.gradients.param_shift_cv, {"dev": device}, device
        return qml.gradients.param_shift, {}, device

    if diff_method == "finite-diff":
        return qml.gradients.finite_diff, {}, device

    if diff_method == "spsa":
        return qml.gradients.spsa_grad, {}, device

    if diff_method == "hadamard":
        return qml.gradients.hadamard_grad, {}, device

    if isinstance(diff_method, str):
        raise qml.QuantumFunctionError(
            f"Differentiation method {diff_method} not recognized. Allowed "
            f"options are {tuple(get_args(SupportedDiffMethods))}."
        )

    if isinstance(diff_method, qml.transforms.core.TransformDispatcher):
        return diff_method, {}, device

    raise qml.QuantumFunctionError(
        f"Differentiation method {diff_method} must be a gradient transform or a string."
    )

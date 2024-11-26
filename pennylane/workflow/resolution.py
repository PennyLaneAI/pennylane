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
"""This module contains the necessary helper functions for setting up the workflow for execution.

"""

from dataclasses import replace
from typing import Literal, get_args

import pennylane as qml
from pennylane.logging import debug_logger
from pennylane.tape import QuantumScriptBatch
from pennylane.transforms.core import TransformDispatcher, TransformProgram

from .execution import _get_interface_name

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


def _resolve_mcm_config(
    mcm_config: "qml.devices.MCMConfig", interface: str, finite_shots: bool
) -> "qml.devices.MCMConfig":
    """Helper function to resolve the mid-circuit measurements configuration based on
    execution parameters"""
    updated_values = {}

    if not finite_shots:
        updated_values["postselect_mode"] = None
        if mcm_config.mcm_method == "one-shot":
            raise ValueError(
                "Cannot use the 'one-shot' method for mid-circuit measurements with analytic mode."
            )

    if mcm_config.mcm_method == "single-branch-statistics":
        raise ValueError("Cannot use mcm_method='single-branch-statistics' without qml.qjit.")

    if interface == "jax-jit" and mcm_config.mcm_method == "deferred":
        # This is a current limitation of defer_measurements. "hw-like" behaviour is
        # not yet accessible.
        if mcm_config.postselect_mode == "hw-like":
            raise ValueError(
                "Using postselect_mode='hw-like' is not supported with jax-jit when using "
                "mcm_method='deferred'."
            )
        updated_values["postselect_mode"] = "fill-shots"

    if (
        finite_shots
        and "jax" in interface
        and mcm_config.mcm_method in (None, "one-shot")
        and mcm_config.postselect_mode in (None, "hw-like")
    ):
        updated_values["postselect_mode"] = "pad-invalid-samples"

    return replace(mcm_config, **updated_values)


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
        new_config = device.preprocess(initial_config)[1]
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


# pylint: disable=protected-access
def _resolve_execution_config(
    execution_config: "qml.devices.ExecutionConfig",
    device: "qml.devices.Device",
    tapes: QuantumScriptBatch,
    transform_program: TransformProgram,
) -> "qml.devices.ExecutionConfig":
    """Resolves the execution configuration for non-device specific properties.

    Args:
        execution_config (qml.devices.ExecutionConfig): an execution config to be executed on the device
        device (qml.devices.Device): a Pennylane device
        tapes (QuantumScriptBatch): a batch of tapes
        transform_program (TransformProgram): a program of transformations to be applied to the tapes

    Returns:
        qml.devices.ExecutionConfig: resolved execution configuration
    """
    updated_values = {}
    updated_values["gradient_keyword_arguments"] = dict(execution_config.gradient_keyword_arguments)

    if (
        "lightning" in device.name
        and qml.metric_tensor in transform_program
        and execution_config.gradient_method == "best"
    ):
        execution_config = replace(execution_config, gradient_method=qml.gradients.param_shift)
    else:
        execution_config = _resolve_diff_method(execution_config, device, tape=tapes[0])

    if execution_config.gradient_method is qml.gradients.param_shift_cv:
        updated_values["gradient_keyword_arguments"]["dev"] = device

    # Mid-circuit measurement configuration validation
    # If the user specifies `interface=None`, regular execution considers it numpy, but the mcm
    # workflow still needs to know if jax-jit is used
    interface = _get_interface_name(tapes, execution_config.interface)
    finite_shots = any(tape.shots for tape in tapes)
    mcm_interface = (
        _get_interface_name(tapes, "auto") if execution_config.interface is None else interface
    )
    mcm_config = _resolve_mcm_config(execution_config.mcm_config, mcm_interface, finite_shots)

    updated_values["mcm_config"] = mcm_config

    execution_config = device.preprocess(execution_config)[1]

    return replace(execution_config, **updated_values)

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
"""This module contains the necessary helper functions for setting up the workflow for execution."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from importlib.metadata import version
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Literal, get_args
from warnings import warn

from packaging.version import Version

import pennylane as qml
from pennylane import math
from pennylane.exceptions import QuantumFunctionError
from pennylane.logging import debug_logger
from pennylane.math import Interface, get_interface
from pennylane.transforms.core import TransformDispatcher

SupportedDiffMethods = Literal[
    None,
    "best",
    "device",
    "backprop",
    "adjoint",
    "parameter-shift",
    "hadamard",
    "reversed-hadamard",
    "direct-hadamard",
    "reversed-direct-hadamard",
    "finite-diff",
    "spsa",
]

if TYPE_CHECKING:
    from pennylane.devices.device_api import Device
    from pennylane.devices.execution_config import ExecutionConfig, MCMConfig
    from pennylane.tape import QuantumScript, QuantumScriptBatch


def _get_jax_interface_name() -> Interface:
    """Check if we are in a jitting context by creating a dummy array and seeing if it's
    abstract.
    """
    x = math.asarray([0], like="jax")
    return Interface.JAX_JIT if math.is_abstract(x) else Interface.JAX


def _validate_jax_version() -> None:
    """Checks if the installed version of JAX is supported. If an unsupported version of
    JAX is installed, a ``RuntimeWarning`` is raised."""
    if not find_spec("jax"):
        return

    jax_version = version("jax")
    min_jax_version = "0.0.0"  # place holders than can be updated as needed
    max_jax_version = "1.0.0"
    if Version(jax_version) < Version(min_jax_version):  # pragma: no cover
        warn(
            f"PennyLane is currently not compatible with versions of less than {min_jax_version}. "
            f"You have version {jax_version} installed.",
            RuntimeWarning,
        )
    if Version(max_jax_version) < Version(jax_version):  # pragma: no cover
        warn(
            f"PennyLane is currently not compatible with versions of greater than {max_jax_version}. "
            f"You have version {jax_version} installed.",
            RuntimeWarning,
        )


# pylint: disable=import-outside-toplevel
# TensorFlow tests were disabled during deprecation
def _use_tensorflow_autograph() -> bool:  # pragma: no cover
    """Checks if TensorFlow is in graph mode, allowing Autograph for optimized execution"""
    try:  # pragma: no cover
        import tensorflow as tf
    except ImportError as e:  # pragma: no cover
        raise QuantumFunctionError(  # pragma: no cover
            "tensorflow not found. Please install the latest "  # pragma: no cover
            "version of tensorflow supported by Pennylane "  # pragma: no cover
            "to enable the 'tensorflow' interface."  # pragma: no cover
        ) from e  # pragma: no cover

    return not tf.executing_eagerly()


def _resolve_interface(interface: str | Interface, tapes: QuantumScriptBatch) -> Interface:
    """Helper function to resolve an interface based on a set of tapes.

    Args:
        interface (str, Interface): Original interface to use as reference.
        tapes (list[.QuantumScript]): Quantum tapes

    Returns:
        Interface: resolved interface
    """
    interface = Interface(interface)

    if interface == Interface.AUTO:
        params = []
        for tape in tapes:
            params.extend(tape.get_parameters(trainable_only=False))
        interface = get_interface(*params)
        try:
            interface = Interface(interface)
        except ValueError:
            # If the interface is not recognized, default to numpy, like networkx
            interface = Interface.NUMPY

    if interface in (Interface.JAX, Interface.JAX_JIT):
        _validate_jax_version()

    if (
        interface == Interface.TF and _use_tensorflow_autograph()
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        interface = Interface.TF_AUTOGRAPH
    if interface == Interface.JAX:
        # pylint: disable=unused-import
        try:  # pragma: no cover
            import jax
        except ImportError as e:  # pragma: no cover
            raise QuantumFunctionError(  # pragma: no cover
                "jax not found. Please install the latest "  # pragma: no cover
                "version of jax to enable the 'jax' interface."  # pragma: no cover
            ) from e  # pragma: no cover

        interface = _get_jax_interface_name()

    return interface


def _resolve_mcm_config(
    mcm_config: MCMConfig, interface: Interface, finite_shots: bool
) -> MCMConfig:
    """Helper function to resolve the mid-circuit measurements configuration based on
    execution parameters"""
    updated_values: dict[str, Any] = {}

    if not finite_shots:
        updated_values["postselect_mode"] = None
        if mcm_config.mcm_method == "one-shot":
            raise ValueError(
                "Cannot use the 'one-shot' method for mid-circuit measurements with analytic mode."
            )

    if mcm_config.mcm_method == "single-branch-statistics":
        raise ValueError(
            "Cannot use mcm_method='single-branch-statistics' without qml.qjit or capture enabled."
        )

    if interface == Interface.JAX_JIT and mcm_config.mcm_method == "deferred":
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
        and interface in {Interface.JAX, Interface.JAX_JIT}
        and mcm_config.mcm_method in (None, "one-shot")
        and mcm_config.postselect_mode in (None, "hw-like")
    ):
        updated_values["postselect_mode"] = "pad-invalid-samples"

    return replace(mcm_config, **updated_values)


def _resolve_hadamard(initial_config: ExecutionConfig, device: Device) -> ExecutionConfig:
    diff_method = initial_config.gradient_method
    updated_values = {"gradient_method": diff_method}
    if diff_method != "hadamard" and "mode" in initial_config.gradient_keyword_arguments:
        raise ValueError(
            f"diff_method={diff_method} cannot be provided with a 'mode' in the gradient_kwargs."
        )

    hadamard_mode_map = {
        "hadamard": "standard",
        "reversed-hadamard": "reversed",
        "direct-hadamard": "direct",
        "reversed-direct-hadamard": "reversed-direct",
    }
    gradient_kwargs = copy(initial_config.gradient_keyword_arguments)
    if "mode" not in gradient_kwargs:
        gradient_kwargs["mode"] = hadamard_mode_map[diff_method]

    if "device_wires" not in gradient_kwargs and "aux_wire" not in gradient_kwargs:
        gradient_kwargs["device_wires"] = device.wires
    updated_values["gradient_keyword_arguments"] = gradient_kwargs
    updated_values["gradient_method"] = qml.gradients.hadamard_grad
    return replace(initial_config, **updated_values)


@debug_logger
def _resolve_diff_method(
    initial_config: ExecutionConfig,
    device: Device,
    tape: QuantumScript | None = None,
) -> ExecutionConfig:
    """
    Resolves the differentiation method and updates the initial execution configuration accordingly.

    Args:
        initial_config (ExecutionConfig): The initial execution configuration.
        device (Device): A PennyLane device.
        tape (Optional[qml.tape.QuantumTape]): The circuit that will be differentiated. Should include shots information.

    Returns:
        ExecutionConfig: Updated execution configuration with the resolved differentiation method.
    """
    diff_method = initial_config.gradient_method
    updated_values = {"gradient_method": diff_method}

    if diff_method is None:
        return initial_config

    if device.supports_derivatives(initial_config, circuit=tape):
        new_config = device.setup_execution_config(initial_config, tape)
        return new_config

    if diff_method in {"backprop", "adjoint", "device"}:
        raise QuantumFunctionError(
            f"Device {device} does not support {diff_method} with requested circuit."
        )

    if "hadamard" in str(diff_method):
        return _resolve_hadamard(initial_config, device)

    if diff_method in {"best", "parameter-shift"}:
        if tape and any(isinstance(op, qml.operation.CV) and op.name != "Identity" for op in tape):
            updated_values["gradient_method"] = qml.gradients.param_shift_cv
            updated_values["gradient_keyword_arguments"] = dict(
                initial_config.gradient_keyword_arguments
            )
            updated_values["gradient_keyword_arguments"]["dev"] = device
        else:
            updated_values["gradient_method"] = qml.gradients.param_shift

    else:
        gradient_transform_map = {
            "finite-diff": qml.gradients.finite_diff,
            "spsa": qml.gradients.spsa_grad,
        }

        if diff_method in gradient_transform_map:
            updated_values["gradient_method"] = gradient_transform_map[diff_method]
        elif isinstance(diff_method, TransformDispatcher):
            updated_values["gradient_method"] = diff_method
        else:
            raise QuantumFunctionError(
                f"Differentiation method {diff_method} not recognized. Allowed "
                f"options are {tuple(get_args(SupportedDiffMethods))}."
            )

    return replace(initial_config, **updated_values)


def _resolve_execution_config(
    execution_config: ExecutionConfig,
    device: Device,
    tapes: QuantumScriptBatch,
) -> ExecutionConfig:
    """Resolves the execution configuration for non-device specific properties.

    Args:
        execution_config (ExecutionConfig): an execution config to be executed on the device
        device (Device): a Pennylane device
        tapes (QuantumScriptBatch): a batch of tapes

    Returns:
        ExecutionConfig: resolved execution configuration
    """
    updated_values: dict[str, Any] = {}

    if execution_config.interface in {Interface.JAX, Interface.JAX_JIT} and not callable(
        execution_config.gradient_method
    ):
        updated_values["grad_on_execution"] = False
    execution_config = _resolve_diff_method(execution_config, device, tape=tapes[0])

    if execution_config.use_device_jacobian_product and not device.supports_vjp(
        execution_config, tapes[0]
    ):
        raise QuantumFunctionError(
            f"device_vjp=True is not supported for device {device},"
            f" diff_method {execution_config.gradient_method},"
            " and the provided circuit."
        )

    # Mid-circuit measurement configuration validation
    # If the user specifies `interface=None`, regular execution considers it numpy, but the mcm
    # workflow still needs to know if jax-jit is used
    interface = _resolve_interface(execution_config.interface, tapes)
    finite_shots = any(tape.shots for tape in tapes)
    mcm_interface = (
        _resolve_interface(Interface.AUTO, tapes)
        if execution_config.interface == Interface.NUMPY
        else interface
    )
    mcm_config = _resolve_mcm_config(execution_config.mcm_config, mcm_interface, finite_shots)

    updated_values["interface"] = interface
    updated_values["mcm_config"] = mcm_config
    execution_config = replace(execution_config, **updated_values)
    execution_config = device.setup_execution_config(execution_config, tapes[0])
    return execution_config

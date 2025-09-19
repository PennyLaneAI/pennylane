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
"""
This module contains a developer focused execution function for internal executions
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import pennylane as qml
from pennylane import math
from pennylane.exceptions import QuantumFunctionError
from pennylane.math import Interface
from pennylane.workflow import _cache_transform

from .jacobian_products import (
    DeviceDerivatives,
    DeviceJacobianProducts,
    JacobianProductCalculator,
    NoGradients,
    TransformJacobianProducts,
)

if TYPE_CHECKING:
    from pennylane.devices import Device, ExecutionConfig
    from pennylane.tape import QuantumScriptBatch
    from pennylane.transforms.core import TransformProgram
    from pennylane.typing import ResultBatch

    ExecuteFn = Callable[[QuantumScriptBatch], ResultBatch]


def _construct_tf_autograph_pipeline(
    config: ExecutionConfig,
    device: Device,
    inner_transform_program: TransformProgram,
):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
    """Handles the pipeline construction for the TF_AUTOGRAPH interface.

    This function determines the execution function (`execute_fn`) and gradient method specifically
    for the TensorFlow Autograph interface.

    Args:
        config (qml.devices.ExecutionConfig): resolved execution configuration
        device (qml.devices.Device): a Pennylane device
        inner_transform_program (qml.transforms.core.TransformProgram): the transformation applied to quantum tapes

    Returns:
        tuple: A tuple containing:
            - `execute_fn`: function to execute quantum tapes
            - `diff_method`: method for computing gradients
    """
    inner_execute = _make_inner_execute(device, inner_transform_program, config)

    def inner_execute_with_empty_jac(tapes, **_):
        return inner_execute(tapes), []

    execute_fn = inner_execute_with_empty_jac

    if config.use_device_gradient:

        if config.grad_on_execution:

            def wrap_execute_and_compute_derivatives(internal_tapes):
                """A partial function wrapping the execute_and_compute_derivatives method of the device."""
                numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(internal_tapes)
                return device.execute_and_compute_derivatives(numpy_tapes, config)

            execute_fn = wrap_execute_and_compute_derivatives
            diff_method = None

        else:

            def execution_with_dummy_jac(internal_tapes):
                """A wrapper around device.execute that returns an empty tuple for derivatives."""
                numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(internal_tapes)
                return device.execute(numpy_tapes, config), tuple()

            execute_fn = execution_with_dummy_jac

            def device_compute_derivatives(internal_tapes):
                """A partial function wrapping the compute_derivatives method of the device."""
                numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(internal_tapes)
                return device.compute_derivatives(numpy_tapes, config)

            diff_method = device_compute_derivatives

    elif config.grad_on_execution is True:
        raise ValueError("Gradient transforms cannot be used with grad_on_execution=True")

    else:
        diff_method = config.gradient_method

    return execute_fn, diff_method


def _construct_ml_execution_pipeline(
    config: ExecutionConfig,
    device: Device,
    inner_transform_program: TransformProgram,
) -> tuple[JacobianProductCalculator, ExecuteFn]:
    """Constructs the ML execution pipeline for all JPC interfaces.

    This function determines the execution function (`execute_fn`) and the Jacobian product
    class (`jpc`) required for gradient computations.

    Args:
        config (qml.devices.ExecutionConfig): resolved execution configuration
        device (qml.devices.Device): a Pennylane device
        inner_transform_program (qml.transforms.core.TransformProgram): the transformation applied to quantum tapes

    Returns:
        tuple: A tuple containing:
            - `jpc`: jacobian product class for computing gradients
            - `execute_fn`: the function to execute quantum tapes

    Raises:
        ValueError: If gradients are computed on execution (`grad_on_execution=True`).
    """
    inner_execute = _make_inner_execute(device, inner_transform_program, config)
    cache = _cache_transform in inner_transform_program

    execute_fn = inner_execute

    if config.gradient_method is None:
        return NoGradients(), execute_fn

    if config.use_device_jacobian_product:
        return DeviceJacobianProducts(device, config), execute_fn

    if config.use_device_gradient:
        jpc = DeviceDerivatives(device, config)
        if config.grad_on_execution:
            execute_fn = jpc.execute_and_cache_jacobian
        else:
            execute_fn = inner_execute
        return jpc, execute_fn

    if config.grad_on_execution is True:
        raise ValueError("Gradient transforms cannot be used with grad_on_execution=True")

    cache_full_jacobian = (config.interface == Interface.AUTOGRAD) and not cache
    jpc = TransformJacobianProducts(
        execute_fn,
        config.gradient_method,
        config.gradient_keyword_arguments,
        cache_full_jacobian,
    )
    for i in range(1, config.derivative_order):
        differentiable = i > 1
        ml_boundary_execute = _get_ml_boundary_execute(config, differentiable=differentiable)
        execute_fn = partial(
            ml_boundary_execute,
            execute_fn=execute_fn,
            jpc=jpc,
            device=device,
        )
        jpc = TransformJacobianProducts(
            execute_fn,
            config.gradient_method,
            config.gradient_keyword_arguments,
        )

    return jpc, execute_fn


# pylint: disable=import-outside-toplevel
def _get_ml_boundary_execute(
    resolved_execution_config: ExecutionConfig, differentiable=False
) -> Callable:
    """Imports and returns the function that handles the interface boundary for a given machine learning framework.

    Args:
        resolved_execution_config (ExecutionConfig): resolved execution configuration set-up for execution
        differentiable (bool): Specifies if the operation should be differentiable within the framework.
            Relevant for TensorFlow and similar interfaces. Defaults to ``False``.

    Returns:
        Callable: Execution function for the specified machine learning framework.

    Raises:
        pennylane.QuantumFunctionError: If the required package for the specified interface is not installed.
    """
    interface = resolved_execution_config.interface
    grad_on_execution = resolved_execution_config.grad_on_execution
    device_vjp = resolved_execution_config.use_device_jacobian_product
    try:
        match interface:
            case Interface.AUTOGRAD:
                from .interfaces.autograd import autograd_execute as ml_boundary

            case (
                Interface.TF_AUTOGRAPH
            ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
                from .interfaces.tensorflow_autograph import execute as ml_boundary

                ml_boundary = partial(ml_boundary, grad_on_execution=grad_on_execution)

            case (
                Interface.TF
            ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
                from .interfaces.tensorflow import tf_execute as full_ml_boundary

                ml_boundary = partial(full_ml_boundary, differentiable=differentiable)

            case Interface.TORCH:
                from .interfaces.torch import execute as ml_boundary

            case Interface.JAX_JIT if resolved_execution_config.convert_to_numpy:
                from .interfaces.jax_jit import jax_jit_jvp_execute as ml_boundary

            case _:  # interface is jax
                if device_vjp:
                    from .interfaces.jax_jit import jax_jit_vjp_execute as ml_boundary
                else:
                    from .interfaces.jax import jax_jvp_execute as ml_boundary

    except ImportError as e:  # pragma: no cover
        raise QuantumFunctionError(
            f"{interface} not found. Please install the latest "
            f"version of {interface} to enable the '{interface}' interface."
        ) from e

    return ml_boundary


def _make_inner_execute(device, inner_transform, execution_config=None) -> Callable:
    """Construct the function responsible for executing quantum tapes within a ML framework boundary
    for first-order derivatives.

    The process involves performing device-specific preprocessing on the tapes (for new devices) or
    device expansion (for legacy devices).

    For higher-order derivatives, this function will delegate to another ML framework execution.
    """

    def inner_execute(tapes: QuantumScriptBatch) -> ResultBatch:
        """Execution that occurs within a ML framework boundary.

        Closure Variables:
            inner_transform(TransformProgram): a transform to apply to a set of tapes
            expand_fn (Callable[[QuantumScript], QuantumScript]): A device preprocessing step
            device (qml.devices.Device): a Pennylane device
        """

        transformed_tapes, transform_post_processing = inner_transform(tapes)

        if transformed_tapes:
            results = device.execute(transformed_tapes, execution_config=execution_config)
        else:
            results = ()

        return transform_post_processing(results)

    return inner_execute


def run(
    tapes: QuantumScriptBatch,
    device: Device,
    config: ExecutionConfig,
    inner_transform_program: TransformProgram,
) -> ResultBatch:
    """Execute a batch of quantum scripts on a device with optional gradient computation.

    Args:
        tapes (qml.tape.QuantumScriptBatch): batch of quantum scripts
        device (qml.devices.Device): a Pennylane device
        config (qml.devices.ExecutionConfig): Resolved configuration detailing
            execution and differentiation settings.
        inner_transform_program (TransformProgram): The transformation program to apply
            to the quantum scripts before execution.

    Returns:
        ResultBatch: results of the execution
    """
    inner_execute = _make_inner_execute(device, inner_transform_program, config)

    # Exiting early if we do not need to deal with an interface boundary
    no_interface_boundary_required = (
        config.interface == Interface.NUMPY or config.gradient_method == "backprop"
    )
    if no_interface_boundary_required:
        results = inner_execute(tapes)
        return results

    # TODO: Prune once support for tf-autograph is dropped
    if (
        config.interface == Interface.TF_AUTOGRAPH
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)

        execute_fn, diff_method = _construct_tf_autograph_pipeline(
            config, device, inner_transform_program
        )

        ml_execute = _get_ml_boundary_execute(
            config,
            differentiable=config.derivative_order > 1,
        )
        results = ml_execute(
            tapes,
            device,
            execute_fn,
            diff_method,
            config.gradient_keyword_arguments,
            _n=1,
            max_diff=config.derivative_order,
        )

        return results

    jpc, execute_fn = _construct_ml_execution_pipeline(config, device, inner_transform_program)

    if config.interface == Interface.JAX_JIT and config.derivative_order > 1:
        # no need to use pure callbacks around execute_fn or the jpc when taking
        # higher order derivatives
        config = replace(config, interface=Interface.JAX)

    ml_execute = _get_ml_boundary_execute(
        config,
        differentiable=config.derivative_order > 1,
    )

    # trainable parameters can only be set on the first pass for jax
    # not higher order passes for higher order derivatives
    if config.interface in {Interface.JAX, Interface.JAX_JIT}:
        for tape in tapes:
            params = tape.get_parameters(trainable_only=False)
            tape.trainable_params = math.get_trainable_indices(params)

    results = ml_execute(tapes, execute_fn, jpc, device=device)
    return results

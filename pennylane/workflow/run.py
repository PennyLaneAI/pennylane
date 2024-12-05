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

from dataclasses import replace
from functools import partial
from typing import Callable

import pennylane as qml
from pennylane.math import Interface
from pennylane.transforms.core import TransformProgram
from pennylane.typing import ResultBatch
from pennylane.workflow import _cache_transform

from .jacobian_products import DeviceDerivatives, DeviceJacobianProducts, TransformJacobianProducts


def _construct_ml_execution_pipeline(
    config: "qml.devices.ExecutionConfig",
    device: "qml.devices.Device",
    inner_transform_program: "qml.transforms.core.TransformProgram",
):
    """Constructs the machine learning execution pipeline based on the configuration, device,
    and inner transformation program.

    This function determines the execution function (`execute_fn`) and the Jacobian product
    class (`jpc`) required for gradient computations. It adapts the execution logic based
    on the specified interface, gradient method, and device capabilities.

    Args:
        config (qml.devices.ExecutionConfig): resolved execution configuration
        device (qml.devices.Device): a Pennylane device
        inner_transform_program (qml.transforms.core.TransformProgram): the transformation applied to quantum tapes before execution

    Returns:
        tuple: A tuple containing:
            - `jpc`: Jacobian product class for computing gradients efficiently.
            - `execute_fn`: The function to execute quantum tapes within the
              machine learning framework boundary.
            - `diff_method`: Method for computing gradients, or None
              if not applicable.

    Raises:
        ValueError: If gradients are computed on execution (`grad_on_execution=True`)
    """
    inner_execute = _make_inner_execute(device, inner_transform_program, config)
    cache = _cache_transform in inner_transform_program
    diff_method = config.gradient_method

    # moved to its own explicit step so that it will be easier to remove
    def inner_execute_with_empty_jac(tapes, **_):
        return inner_execute(tapes), []

    execute_fn = inner_execute
    if config.interface == Interface.TF_AUTOGRAPH:
        execute_fn = inner_execute_with_empty_jac

    jpc = None

    if config.use_device_jacobian_product and config.interface != Interface.TF_AUTOGRAPH:
        jpc = DeviceJacobianProducts(device, config)

    elif config.use_device_gradient:
        jpc = DeviceDerivatives(device, config)

        if config.interface != Interface.TF_AUTOGRAPH:
            execute_fn = (
                jpc.execute_and_cache_jacobian if config.grad_on_execution else inner_execute
            )

        elif config.grad_on_execution:

            def wrap_execute_and_compute_derivatives(internal_tapes):
                """A partial function that wraps the execute_and_compute_derivatives
                method of the device.

                Closure Variables:
                    device: The device to execute on
                    resolved_execution_config: the ExecutionConfig that specifies how to perform the simulations.
                """
                numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(internal_tapes)

                return device.execute_and_compute_derivatives(numpy_tapes, config)

            execute_fn = wrap_execute_and_compute_derivatives

            diff_method = None

        else:

            def execution_with_dummy_jac(internal_tapes) -> tuple[ResultBatch, tuple]:
                """A wrapper around device.execute that adds an empty tuple instead of derivatives.

                Closure Variables:
                    device: the device to execute on
                    resolved_execution_config: the ExecutionConfig that specifies how to perform the simulations.
                """
                numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(internal_tapes)
                return device.execute(numpy_tapes, config), tuple()

            execute_fn = execution_with_dummy_jac

            def device_compute_derivatives(internal_tapes):
                """A partial function that wraps compute_derivatives method of the device.

                Closure Variables:
                    device: the device to execute on
                    resolved_execution_config: the ExecutionConfig that specifies how to take the derivative.
                """
                numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(internal_tapes)
                return device.compute_derivatives(numpy_tapes, config)

            diff_method = device_compute_derivatives

    elif config.grad_on_execution is True:
        # In "forward" mode, gradients are automatically handled
        # within execute_and_gradients, so providing a diff_method
        # in this case would have ambiguous behaviour.
        raise ValueError("Gradient transforms cannot be used with grad_on_execution=True")

    elif config.interface != Interface.TF_AUTOGRAPH:

        # See autograd.py submodule docstring for explanation for ``cache_full_jacobian``
        cache_full_jacobian = (config.interface == Interface.AUTOGRAD) and not cache

        # we can have higher order derivatives when the `inner_execute` used to take
        # transform gradients is itself differentiable
        # To make the inner execute itself differentiable, we make it an interface boundary with
        # its own jacobian product class
        # this mechanism unpacks the currently existing recursion
        jpc = TransformJacobianProducts(
            execute_fn,
            diff_method,
            config.gradient_keyword_arguments,
            cache_full_jacobian,
        )
        for i in range(1, config.derivative_order):
            differentiable = i > 1
            ml_boundary_execute = _get_ml_boundary_execute(
                config,
                differentiable=differentiable,
            )
            execute_fn = partial(
                ml_boundary_execute,
                execute_fn=execute_fn,
                jpc=jpc,
                device=device,
            )
            jpc = TransformJacobianProducts(
                execute_fn,
                diff_method,
                config.gradient_keyword_arguments,
            )

    return jpc, execute_fn, diff_method


# pylint: disable=import-outside-toplevel
def _get_ml_boundary_execute(
    resolved_execution_config: "qml.devices.ExecutionConfig", differentiable=False
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
        if interface == Interface.AUTOGRAD:
            from .interfaces.autograd import autograd_execute as ml_boundary

        elif interface == Interface.TF_AUTOGRAPH:
            from .interfaces.tensorflow_autograph import execute as ml_boundary

            ml_boundary = partial(ml_boundary, grad_on_execution=grad_on_execution)

        elif interface == Interface.TF:
            from .interfaces.tensorflow import tf_execute as full_ml_boundary

            ml_boundary = partial(full_ml_boundary, differentiable=differentiable)

        elif interface == Interface.TORCH:
            from .interfaces.torch import execute as ml_boundary

        elif interface == Interface.JAX_JIT:
            from .interfaces.jax_jit import jax_jit_jvp_execute as ml_boundary

        else:  # interface is jax
            if device_vjp:
                from .interfaces.jax_jit import jax_jit_vjp_execute as ml_boundary
            else:
                from .interfaces.jax import jax_jvp_execute as ml_boundary

    except ImportError as e:  # pragma: no cover
        raise qml.QuantumFunctionError(
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

    def inner_execute(tapes: "qml.tape.QuantumScriptBatch", **_) -> ResultBatch:
        """Execution that occurs within a ML framework boundary.

        Closure Variables:
            expand_fn (Callable[[QuantumTape], QuantumTape]): A device preprocessing step
            device (qml.devices.Device)
        """

        transformed_tapes, transform_post_processing = inner_transform(tapes)

        if transformed_tapes:
            results = device.execute(transformed_tapes, execution_config=execution_config)
        else:
            results = ()

        return transform_post_processing(results)

    return inner_execute


# pylint: disable=too-many-branches
def run(
    tapes: "qml.tape.QuantumScriptBatch",
    device: "qml.devices.Device",
    resolved_execution_config: "qml.devices.ExecutionConfig",
    inner_transform_program: TransformProgram,
) -> ResultBatch:
    """Execute a batch of quantum scripts on a device with optional gradient computation.

    Args:
        tapes (qml.tape.QuantumScriptBatch): batch of quantum scripts
        device (qml.devices.Device): a Pennylane device
        resolved_execution_config (qml.devices.ExecutionConfig): Configuration detailing
            execution and differentiation settings.
        inner_transform_program (TransformProgram): The transformation program to apply
            to the quantum scripts before execution.

    Returns:
        ResultBatch: results of the execution
    """
    inner_execute = _make_inner_execute(device, inner_transform_program, resolved_execution_config)

    # Exiting early if we do not need to deal with an interface boundary
    no_interface_boundary_required = (
        resolved_execution_config.interface == Interface.NUMPY
        or resolved_execution_config.gradient_method
        in {
            None,
            "backprop",
        }
    )
    if no_interface_boundary_required:
        results = inner_execute(tapes)
        return results

    jpc, execute_fn, diff_method = _construct_ml_execution_pipeline(
        resolved_execution_config, device, inner_transform_program
    )

    if (
        resolved_execution_config.interface == Interface.JAX_JIT
        and resolved_execution_config.derivative_order > 1
    ):
        # no need to use pure callbacks around execute_fn or the jpc when taking
        # higher order derivatives
        config = replace(config, interface=Interface.JAX)

    # trainable parameters can only be set on the first pass for jax
    # not higher order passes for higher order derivatives
    if resolved_execution_config.interface in {Interface.JAX, Interface.JAX_JIT}:
        for tape in tapes:
            params = tape.get_parameters(trainable_only=False)
            tape.trainable_params = qml.math.get_trainable_indices(params)

    ml_execute = _get_ml_boundary_execute(
        resolved_execution_config,
        differentiable=resolved_execution_config.derivative_order > 1,
    )

    if resolved_execution_config.interface != Interface.TF_AUTOGRAPH:
        results = ml_execute(tapes, execute_fn, jpc, device=device)
    else:
        results = ml_execute(  # pylint: disable=too-many-function-args, unexpected-keyword-arg
            tapes,
            device,
            execute_fn,
            diff_method,
            resolved_execution_config.gradient_keyword_arguments,
            _n=1,
            max_diff=resolved_execution_config.derivative_order,
        )

    return results

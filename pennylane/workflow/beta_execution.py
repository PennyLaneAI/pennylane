

from warnings import warn
from functools import partial
from typing import Optional, MutableMapping, Tuple, Callable
from dataclasses import replace

import pennylane as qml

from pennylane.transforms.core import TransformProgram

from .cache_transform import cache_transform
from .jacobian_products import JacobianProductCalculator, DeviceDerivatives, DeviceJacobianProducts, NullJPC, TransformJacobianProducts

BatchTape = Tuple[qml.tape.QuantumTape]
ExecuteFn = Callable[[BatchTape], qml.typing.ResultBatch]

def _use_tensorflow_autograph():
    import tensorflow as tf

    return not tf.executing_eagerly()

INTERFACE_MAP = {
    None: "Numpy",
    "auto": "auto",
    "autograd": "autograd",
    "numpy": "autograd",
    "scipy": "numpy",
    "jax": "jax",
    "jax-jit": "jax",
    "jax-python": "jax",
    "JAX": "jax",
    "torch": "torch",
    "pytorch": "torch",
    "tf": "tf",
    "tensorflow": "tf",
    "tensorflow-autograph": "tf",
    "tf-autograph": "tf",
}
"""dict[str, str]: maps an allowed interface specification to its canonical name."""


gradient_transform_map = {
    "parameter-shift": qml.gradients.param_shift,
    "finite-diff": qml.gradients.finite_diff,
    "spsa": qml.gradients.spsa_grad,
    "hadamard": qml.gradients.hadamard_grad
}



def null_ml_boundary(tapes: BatchTape, execute_fn: ExecuteFn, jpc: JacobianProductCalculator, device: qml.devices.Device=None) -> qml.typing.ResultBatch:
    return execute_fn(tapes)


def _get_ml_boundary_execute(execution_config, differentiable=False
) -> Callable:
    """Imports and returns the function that binds derivatives of the required ml framework.

    Args:
        interface (str): The designated ml framework.

        grad_on_execution (bool): whether or not the device derivatives are taken upon execution
    Returns:
        Callable

    Raises:
        pennylane.QuantumFunctionError if the required package is not installed.

    """
    if execution_config.interface is None or execution_config.gradient_method == "backprop":
        return null_ml_boundary

    mapped_interface = INTERFACE_MAP[execution_config.interface]
    try:
        if mapped_interface == "autograd":
            from .interfaces.autograd import autograd_execute as ml_boundary

        elif mapped_interface == "tf":
            if "autograph" in execution_config.interface:
                from .interfaces.tensorflow_autograph import execute as ml_boundary

                ml_boundary = partial(ml_boundary, grad_on_execution=execution_config.grad_on_execution)

            else:
                from .interfaces.tensorflow import tf_execute as full_ml_boundary

                ml_boundary = partial(full_ml_boundary, differentiable=differentiable)

        elif mapped_interface == "torch":
            from .interfaces.torch import execute as ml_boundary

        elif execution_config.interface == "jax-jit":
            if execution_config.use_device_jacobian_product:
                from .interfaces.jax_jit import jax_jit_vjp_execute as ml_boundary
            else:
                from .interfaces.jax_jit import jax_jit_jvp_execute as ml_boundary
        else:  # interface in {"jax", "jax-python", "JAX"}:
            if execution_config.use_device_jacobian_product:
                from .interfaces.jax_jit import jax_jit_vjp_execute as ml_boundary
            else:
                from .interfaces.jax import jax_jvp_execute as ml_boundary

    except ImportError as e:  # pragma: no-cover
        raise qml.QuantumFunctionError(
            f"{mapped_interface} not found. Please install the latest "
            f"version of {mapped_interface} to enable the '{mapped_interface}' interface."
        ) from e
    return ml_boundary


def _resolve_interface(tapes, interface: Optional[str]) -> Optional[str]:
    if interface == "auto":
        params = []
        for tape in tapes:
            params.extend(tape.get_parameters(trainable_only=False))
        interface = qml.math.get_interface(*params)
    if INTERFACE_MAP.get(interface, "") == "tf" and _use_tensorflow_autograph():
        interface = "tf-autograph"
        raise NotImplementedError
    if interface == "jax":
        try:  # pragma: no-cover
            from .interfaces.jax import get_jax_interface_name
        except ImportError as e:  # pragma: no-cover
            raise qml.QuantumFunctionError(  # pragma: no-cover
                "jax not found. Please install the latest "  # pragma: no-cover
                "version of jax to enable the 'jax' interface."  # pragma: no-cover
            ) from e  # pragma: no-cover

        interface = get_jax_interface_name(tapes)
        # Only need to calculate derivatives with jax when we know it will be executed later.
    return interface


def resolve_execution_config(tapes: BatchTape,
    device: qml.devices.Device,
    execution_config: qml.devices.ExecutionConfig) -> qml.devices.ExecutionConfig:

    interface = _resolve_interface(tapes, execution_config.interface)
    execution_config = replace(execution_config, interface=interface)

    if device.supports_derivatives(execution_config, circuit=tapes[0]):
        return device.preprocess(execution_config)[1]
    if execution_config.use_device_gradient or execution_config.use_device_jacobian_product:
        raise qml.QuantumFunctionError(f"device {device} does not support derivative method"
           f" {execution_config.gradient_method} with tape {tapes[0]}")

    execution_config = replace(execution_config, use_device_gradient=False, use_device_jacobian_product=False)
    if execution_config.gradient_method in {"best", "parameter-shift", qml.gradients.param_shift}:
        if tapes and any(isinstance(o, qml.operation.CV) for o in tapes[0]):
            gradient_method = qml.gradients.param_shift_cv
        else:
            gradient_method = qml.gradients.param_shift
    elif execution_config.gradient_method in gradient_transform_map:
        gradient_method = gradient_transform_map[execution_config.gradient_method]
    elif isinstance(execution_config.gradient_method, qml.transforms.core.TransformDispatcher):
        gradient_method = execution_config.gradient_method
    else:
        raise qml.QuantumFunctionError(f"Unrecognized gradient_method {execution_config.gradient_method}")

    if execution_config.grad_on_execution:
        raise qml.QuantumFunctionError("grad_on_execution=True cannot be used with gradient transforms.")

    return replace(execution_config, gradient_method = gradient_method)


def setup_transform_programs(user_transforms: TransformProgram,
    device: qml.devices.Device,
    execution_config: qml.devices.ExecutionConfig,
    cache: Optional[MutableMapping]) -> tuple[TransformProgram, TransformProgram]:

    outer_transform_program = TransformProgram(user_transforms)
    inner_transform_program = TransformProgram()

    if execution_config.use_device_gradient:
        outer_transform_program += device.preprocess(execution_config)[0]
    else:
        inner_transform_program += device.preprocess(execution_config)[0]

    if getattr(execution_config.gradient_method, "expand_transform", False):
        outer_transform_program.insert_front_transform(
            qml.transform(execution_config.gradient_method.expand_transform),
            **execution_config.gradient_keyword_arguments,
        )

    if execution_config.gradient_method != "backprop":
        inner_transform_program.add_transform(qml.transforms.convert_to_numpy_parameters)

    if cache is not None:
        inner_transform_program.add_transform(cache_transform, cache=cache)

    outer_transform_program.prune_dynamic_transform()
    inner_transform_program.prune_dynamic_transform()
    return outer_transform_program, inner_transform_program


def _get_jacobian_product_calculator(device: qml.devices.Device, execution_config: qml.devices.ExecutionConfig, inner_execute: ExecuteFn) -> JacobianProductCalculator:

    if execution_config.gradient_method in {None, "backprop"}:
        return NullJPC()

    if execution_config.use_device_jacobian_product:
        return DeviceJacobianProducts(device, execution_config)

    if execution_config.use_device_gradient:
        return DeviceDerivatives(device, execution_config)

    # we can have higher order derivatives when the `inner_execute` used to take
    # transform gradients is itself differentiable
    # To make the inner execute itself differentiable, we make it an interface boundary with
    # its own jacobian product class
    # this mechanism unpacks the currently existing recursion

    execute_fn = inner_execute
    jpc = TransformJacobianProducts(execute_fn, execution_config.gradient_method, execution_config.gradient_keyword_arguments)
    for i in range(1, execution_config.derivative_order):
        differentiable = i > 1
        ml_boundary_execute = _get_ml_boundary_execute(execution_config, differentiable=differentiable)
        execute_fn = partial(
            ml_boundary_execute,
            execute_fn=execute_fn,
            jpc=jpc,
            device=device,
        )
        jpc = TransformJacobianProducts(execute_fn, execution_config.gradient_method, execution_config.gradient_keyword_arguments)

    return jpc


def _make_inner_execute(device:qml.devices.Device, execution_config: qml.devices.ExecutionConfig, inner_transform_program: TransformProgram) -> ExecuteFn:
    def inner_execute(tapes :BatchTape) -> qml.typing.ResultBatch:
        new_batch, postprocessing = inner_transform_program(tapes)
        results = device.execute(new_batch, execution_config) if new_batch else ()
        return postprocessing(results)
    return inner_execute

def _warn_about_kwargs(kwargs):
    for key in kwargs:
        if key in {"gradient_fn", "interface", "grad_on_execution", "gradient_kwargs", "max_diff", "device_vjp"}:
            warn( f"{key} should now be specified via the execution config instead of as its own keyword argument.", qml.PennyLaneDeprecationWarning)
        elif key == "cachesize":
            warn(f"Please provide a cache with size {kwargs['cachesize']} instead of providing the cachesize itself.", qml.PennyLaneDeprecationWarning)
        elif key in {"override_shots", "expand_fn", "max_expansion", "device_batch_transform"}:
            warn((
                f"keyword argument {key} no longer supported.\n"
                "If you wish to customize the behavior of a device, please modify "
                "the device behavior via either defining a new device that inherits from "
                " the original, or defining a device wrapper that modifies the behavior of an "
                "existing device."
            ), qml.PennyLaneDeprecationWarning)
        else:
            raise ValueError(f"Unrecognized keyword argument {key}")

def execute(tapes: BatchTape,
    device: qml.devices.Device,
    config : qml.devices.ExecutionConfig = qml.devices.DefaultExecutionConfig,
    user_transform_program: Optional[TransformProgram]=None,
    cache: Optional[MutableMapping] = None,
    **deprecated_kwargs
    ) -> qml.typing.ResultBatch:
    _warn_about_kwargs(kwargs)
    if cache is not None and not isinstance(cache, MutableMapping):
        raise ValueError(f"cache must None or a MutableMapping. Got {cache}")
    user_transform_program = user_transform_program or TransformProgram()
    config = resolve_execution_config(tapes, device, config)
    outer_transform_program, inner_transform_program = setup_transform_programs(user_transform_program, device, config, cache)


    inner_execute_fn = _make_inner_execute(device, config, inner_transform_program)
    jpc = _get_jacobian_product_calculator(device, config, inner_execute_fn)
    ml_boundary = _get_ml_boundary_execute(config)

    if config.grad_on_execution and isinstance(jpc, DeviceDerivatives):
        inner_execute_fn = jpc.execute_and_cache_jacobian

    new_batch, outer_postprocessing = outer_transform_program(tapes)
    if outer_transform_program.is_informative:
        return outer_postprocessing(new_batch)
    results = ml_boundary(new_batch, inner_execute_fn, jpc, device=device)
    return outer_postprocessing(results)

    
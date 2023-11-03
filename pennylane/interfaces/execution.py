# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the cache_execute decoratator, for adding caching to a function
that executes multiple tapes on a device.

Also contains the general execute function, for exectuting tapes on
devices with autodifferentiation support.
"""

# pylint: disable=import-outside-toplevel,too-many-arguments,too-many-branches,not-callable
# pylint: disable=unused-argument,unnecessary-lambda-assignment,inconsistent-return-statements,
# pylint: disable=too-many-statements, invalid-unary-operand-type, function-redefined

import inspect
import warnings
from functools import wraps, partial
from typing import Callable, Sequence, Optional, Union, Tuple
import logging

from cachetools import LRUCache, Cache

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import ResultBatch

from .set_shots import set_shots

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

device_type = Union[qml.Device, "qml.devices.Device"]

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

#: list[str]: allowed interface strings
SUPPORTED_INTERFACES = list(INTERFACE_MAP)
"""list[str]: allowed interface strings"""


def _adjoint_jacobian_expansion(
    tapes: Sequence[QuantumTape], grad_on_execution: bool, interface: str, max_expansion: int
):
    """Performs adjoint jacobian specific expansion.  Expands so that every
    trainable operation has a generator.

    TODO: Let the device specify any gradient-specific expansion logic.  This
    function will be removed once the device-support pipeline is improved.
    """
    if grad_on_execution and INTERFACE_MAP[interface] == "jax":
        # qml.math.is_trainable doesn't work with jax on the forward pass
        non_trainable = qml.operation.has_nopar
    else:
        non_trainable = ~qml.operation.is_trainable

    stop_at = ~qml.operation.is_measurement & (
        non_trainable | qml.operation.has_gen  # pylint: disable=unsupported-binary-operation
    )
    for i, tape in enumerate(tapes):
        if any(not stop_at(op) for op in tape.operations):
            tapes[i] = tape.expand(stop_at=stop_at, depth=max_expansion)

    return tapes


def _get_ml_boundary_execute(interface: str, grad_on_execution: bool) -> Callable:
    """Imports and returns the function that binds derivatives of the required ml framework.

    Args:
        interface (str): The designated ml framework.

        grad_on_execution (bool): whether or not the device derivatives are taken upon execution
    Returns:
        Callable

    Raises:
        pennylane.QuantumFunctionError if the required package is not installed.

    """
    mapped_interface = INTERFACE_MAP[interface]
    try:
        if mapped_interface == "autograd":
            from .autograd import execute as ml_boundary

        elif mapped_interface == "tf":
            import tensorflow as tf

            if not tf.executing_eagerly() or "autograph" in interface:
                from .tensorflow_autograph import execute as ml_boundary

                ml_boundary = partial(ml_boundary, grad_on_execution=grad_on_execution)

            else:
                from .tensorflow import execute as ml_boundary

        elif mapped_interface == "torch":
            from .torch import execute as ml_boundary

        elif interface == "jax-jit":
            from .jax_jit import execute as ml_boundary
        else:  # interface in {"jax", "jax-python", "JAX"}:
            from .jax import execute as ml_boundary

    except ImportError as e:  # pragma: no-cover
        raise qml.QuantumFunctionError(
            f"{mapped_interface} not found. Please install the latest "
            f"version of {mapped_interface} to enable the '{mapped_interface}' interface."
        ) from e
    return ml_boundary


def _batch_transform(
    tapes: Sequence[QuantumTape],
    device: device_type,
    config: "qml.devices.ExecutionConfig",
    override_shots: Union[bool, int, Sequence[int]] = False,
    device_batch_transform: bool = True,
) -> Tuple[Sequence[QuantumTape], Callable, "qml.devices.ExecutionConfig"]:
    """Apply the device batch transform unless requested not to.

    Args:
        tapes (Tuple[.QuantumTape]): batch of tapes to preprocess
        device (Device, devices.Device): the device that defines the required batch transformation
        config (qml.devices.ExecutionConfig): the config that characterizes the requested computation
        override_shots (int): The number of shots to use for the execution. If ``False``, then the
            number of shots on the device is used.
        device_batch_transform (bool): Whether to apply any batch transforms defined by the device
            (within :meth:`Device.batch_transform`) to each tape to be executed. The default behaviour
            of the device batch transform is to expand out Hamiltonian measurements into
            constituent terms if not supported on the device.

    Returns:
        Sequence[QuantumTape], Callable: The new batch of quantum scripts and the post processing

    """
    # TODO: Remove once old device are removed
    if device_batch_transform:
        dev_batch_transform = set_shots(device, override_shots)(device.batch_transform)
        return *qml.transforms.map_batch_transform(dev_batch_transform, tapes), config

    def null_post_processing_fn(results):
        """A null post processing function used because the user requested not to use the device batch transform."""
        return results

    return tapes, null_post_processing_fn, config


def _preprocess_expand_fn(
    expand_fn: Union[str, Callable], device: device_type, max_expansion: int
) -> Callable:
    """Preprocess the ``expand_fn`` configuration property.

    Args:
        expand_fn (str, Callable): If string, then it must be "device".  Otherwise, it should be a map
            from one tape to a new tape. The final tape must be natively executable by the device.
        device (Device, devices.Device): The device that we will be executing on.
        max_expansion (int): The number of times the internal circuit should be expanded when
            executed on a device. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations in the decomposition
            remain unsupported by the device, another expansion occurs.

    Returns:
        Callable: a map from one quantum tape to a new one. The output should be compatible with the device.

    """
    if expand_fn != "device":
        return expand_fn
    if isinstance(device, qml.devices.Device):

        def blank_expansion_function(tape):  # pylint: disable=function-redefined
            """A blank expansion function since the new device handles expansion in preprocessing."""
            return tape

        return blank_expansion_function

    def device_expansion_function(tape):  # pylint: disable=function-redefined
        """A wrapper around the device ``expand_fn``."""
        return device.expand_fn(tape, max_expansion=max_expansion)

    return device_expansion_function


def _make_inner_execute(
    device, override_shots, cache, expand_fn=None, execution_config=None, numpy_only=True
) -> Callable:
    """Construct the function that will execute the tapes inside the ml framework registration
    for the 1st order derivatives.

    Steps in between the ml framework execution and the device are:
    - caching
    - conversion to numpy
    - device expansion (old device)

    For higher order derivatives, the "inner execute" will be another ml framework execute.
    """

    if isinstance(device, qml.Device):
        device_execution = set_shots(device, override_shots)(device.batch_execute)

    else:
        device_execution = partial(device.execute, execution_config=execution_config)

    # use qml.interfaces so that mocker can spy on it during testing
    cached_device_execution = qml.interfaces.cache_execute(
        device_execution, cache, return_tuple=False
    )

    def inner_execute(tapes: Sequence[QuantumTape], **_) -> ResultBatch:
        """Execution that occurs within a machine learning framework boundary.

        Closure Variables:
            expand_fn (Callable[[QuantumTape], QuantumTape]): A device preprocessing step
            numpy_only (bool): whether or not to convert the data to numpy or leave as is
            cached_device_execution (Callable[[Sequence[QuantumTape]], ResultBatch])

        """
        if expand_fn:
            tapes = tuple(expand_fn(t) for t in tapes)
        if numpy_only:
            tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        return cached_device_execution(tapes)

    return inner_execute


def cache_execute(fn: Callable, cache, pass_kwargs=False, return_tuple=True, expand_fn=None):
    """Decorator that adds caching to a function that executes
    multiple tapes on a device.

    This decorator makes use of :attr:`.QuantumTape.hash` to identify
    unique tapes.

    - If a tape does not match a hash in the cache, then the tape
      has not been previously executed. It is executed, and the result
      added to the cache.

    - If a tape matches a hash in the cache, then the tape has been previously
      executed. The corresponding cached result is
      extracted, and the tape is not passed to the execution function.

    - Finally, there might be the case where one or more tapes in the current
      set of tapes to be executed are identical and thus share a hash. If this is the case,
      duplicates are removed, to avoid redundant evaluations.

    Args:
        fn (callable): The execution function to add caching to.
            This function should have the signature ``fn(tapes, **kwargs)``,
            and it should return ``list[tensor_like]``, with the
            same length as the input ``tapes``.
        cache (None or dict or Cache or bool): The cache to use. If ``None``,
            caching will not occur.
        pass_kwargs (bool): If ``True``, keyword arguments passed to the
            wrapped function will be passed directly to ``fn``. If ``False``,
            they will be ignored.
        return_tuple (bool): If ``True``, the output of ``fn`` is returned
            as a tuple ``(fn_ouput, [])``, to match the output of execution functions
            that also return gradients.

    Returns:
        function: a wrapped version of the execution function ``fn`` with caching
        support
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Entry with args=(fn=%s, cache=%s, pass_kwargs=%s, return_tuple=%s, expand_fn=%s) called by=%s",
            fn
            if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(fn))
            else "\n" + inspect.getsource(fn),
            cache,
            pass_kwargs,
            return_tuple,
            expand_fn
            if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(expand_fn))
            else "\n" + inspect.getsource(expand_fn) + "\n",
            "::L".join(str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]),
        )

    if expand_fn is not None:
        original_fn = fn

        def fn(tapes: Sequence[QuantumTape], **kwargs):  # pylint: disable=function-redefined
            tapes = [expand_fn(tape) for tape in tapes]
            return original_fn(tapes, **kwargs)

    @wraps(fn)
    def wrapper(tapes: Sequence[QuantumTape], **kwargs):
        if not pass_kwargs:
            kwargs = {}

        if cache is None or (isinstance(cache, bool) and not cache):
            # No caching. Simply execute the execution function
            # and return the results.

            # must convert to list as new device interface returns tuples
            res = list(fn(tapes, **kwargs))
            return (res, []) if return_tuple else res

        execution_tapes = {}
        cached_results = {}
        hashes = {}
        repeated = {}

        for i, tape in enumerate(tapes):
            h = tape.hash

            if h in hashes.values():
                # Tape already exists within ``tapes``. Determine the
                # index of the first occurrence of the tape, store this,
                # and continue to the next iteration.
                idx = list(hashes.keys())[list(hashes.values()).index(h)]
                repeated[i] = idx
                continue

            hashes[i] = h

            if hashes[i] in cache:
                # Tape exists within the cache, store the cached result
                cached_results[i] = cache[hashes[i]]
                if tape.shots and getattr(cache, "_persistent_cache", True):
                    warnings.warn(
                        "Cached execution with finite shots detected!\n"
                        "Note that samples as well as all noisy quantities computed via sampling "
                        "will be identical across executions. This situation arises where tapes "
                        "are executed with identical operations, measurements, and parameters.\n"
                        "To avoid this behavior, provide 'cache=False' to the QNode or execution "
                        "function.",
                        UserWarning,
                    )
            else:
                # Tape does not exist within the cache, store the tape
                # for execution via the execution function.
                execution_tapes[i] = tape

        # if there are no execution tapes, simply return!
        if not execution_tapes:
            if not repeated:
                res = list(cached_results.values())
                return (res, []) if return_tuple else res

        else:
            # execute all unique tapes that do not exist in the cache
            # convert to list as new device interface returns a tuple
            res = list(fn(tuple(execution_tapes.values()), **kwargs))

        final_res = []

        for i, tape in enumerate(tapes):
            if i in cached_results:
                # insert cached results into the results vector
                final_res.append(cached_results[i])

            elif i in repeated:
                # insert repeated results into the results vector
                final_res.append(final_res[repeated[i]])

            else:
                # insert evaluated results into the results vector
                r = res.pop(0)
                final_res.append(r)
                cache[hashes[i]] = r

        return (final_res, []) if return_tuple else final_res

    wrapper.fn = fn
    return wrapper


def execute(
    tapes: Sequence[QuantumTape],
    device: device_type,
    gradient_fn: Optional[Union[Callable, str]] = None,
    interface="auto",
    transform_program=None,
    config=None,
    grad_on_execution="best",
    gradient_kwargs=None,
    cache: Union[bool, dict, Cache] = True,
    cachesize=10000,
    max_diff=1,
    override_shots: int = False,
    expand_fn="device",  # type: ignore
    max_expansion=10,
    device_batch_transform=True,
) -> ResultBatch:
    """New function to execute a batch of tapes on a device in an autodifferentiable-compatible manner. More cases will be added,
    during the project. The current version is supporting forward execution for Numpy and does not support shot vectors.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (pennylane.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        gradient_fn (None or callable): The gradient transform function to use
            for backward passes. If "device", the device will be queried directly
            for the gradient (if supported).
        interface (str): The interface that will be used for classical autodifferentiation.
            This affects the types of parameters that can exist on the input tapes.
            Available options include ``autograd``, ``torch``, ``tf``, ``jax`` and ``auto``.
        transform_program(.TransformProgram): A transform program to be applied to the initial tape.
        config (qml.devices.ExecutionConfig): A datastructure describing the parameters needed to fully describe the execution.
        grad_on_execution (bool, str): Whether the gradients should be computed on the execution or not. Only applies
            if the device is queried for the gradient; gradient transform
            functions available in ``qml.gradients`` are only supported on the backward
            pass. The 'best' option chooses automatically between the two options and is default.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes
        cache (bool, dict, Cache): Whether to cache evaluations. This can result in
            a significant reduction in quantum evaluations during gradient computations.
        cachesize (int): the size of the cache
        max_diff (int): If ``gradient_fn`` is a gradient transform, this option specifies
            the maximum number of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.
        override_shots (int): The number of shots to use for the execution. If ``False``, then the
            number of shots on the device is used.
        expand_fn (str, function): Tape expansion function to be called prior to device execution.
            Must have signature of the form ``expand_fn(tape, max_expansion)``, and return a
            single :class:`~.QuantumTape`. If not provided, by default :meth:`Device.expand_fn`
            is called.
        max_expansion (int): The number of times the internal circuit should be expanded when
            executed on a device. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations in the decomposition
            remain unsupported by the device, another expansion occurs.
        device_batch_transform (bool): Whether to apply any batch transforms defined by the device
            (within :meth:`Device.batch_transform`) to each tape to be executed. The default behaviour
            of the device batch transform is to expand out Hamiltonian measurements into
            constituent terms if not supported on the device.

    Returns:
        list[tensor_like[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.

    **Example**

    Consider the following cost function:

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        def cost_fn(params, x):
            ops1 = [qml.RX(params[0], wires=0), qml.RY(params[1], wires=0)]
            measurements1 = [qml.expval(qml.PauliZ(0))]
            tape1 = qml.tape.QuantumTape(ops1, measurements1)

            ops2 = [
                qml.RX(params[2], wires=0),
                qml.RY(x[0], wires=1),
                qml.CNOT(wires=(0,1))
            ]
            measurements2 = [qml.probs(wires=0)]
            tape2 = qml.tape.QuantumTape(ops2, measurements2)

            tapes = [tape1, tape2]

            # execute both tapes in a batch on the given device
            res = qml.execute(tapes, dev, gradient_fn=qml.gradients.param_shift, max_diff=2)

            return res[0] + res[1][0] - res[1][1]

    In this cost function, two **independent** quantum tapes are being
    constructed; one returning an expectation value, the other probabilities.
    We then batch execute the two tapes, and reduce the results to obtain
    a scalar.

    Let's execute this cost function while tracking the gradient:

    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> x = np.array([0.5], requires_grad=True)
    >>> cost_fn(params, x)
    1.93050682

    Since the ``execute`` function is differentiable, we can
    also compute the gradient:

    >>> qml.grad(cost_fn)(params, x)
    (array([-0.0978434 , -0.19767681, -0.29552021]), array([5.37764278e-17]))

    Finally, we can also compute any nth-order derivative. Let's compute the Jacobian
    of the gradient (that is, the Hessian):

    >>> x.requires_grad = False
    >>> qml.jacobian(qml.grad(cost_fn))(params, x)
    array([[-0.97517033,  0.01983384,  0.        ],
           [ 0.01983384, -0.97517033,  0.        ],
           [ 0.        ,  0.        , -0.95533649]])
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            """Entry with args=(tapes=%s, device=%s, gradient_fn=%s, interface=%s, grad_on_execution=%s, gradient_kwargs=%s, cache=%s, cachesize=%s, max_diff=%s, override_shots=%s, expand_fn=%s, max_expansion=%s, device_batch_transform=%s) called by=%s""",
            tapes,
            repr(device),
            gradient_fn
            if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(gradient_fn))
            else "\n" + inspect.getsource(gradient_fn) + "\n",
            interface,
            grad_on_execution,
            gradient_kwargs,
            cache,
            cachesize,
            max_diff,
            override_shots,
            expand_fn
            if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(expand_fn))
            else "\n" + inspect.getsource(expand_fn) + "\n",
            max_expansion,
            device_batch_transform,
            "::L".join(str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]),
        )

    ### Specifying and preprocessing variables ####
    transform_program = transform_program or qml.transforms.core.TransformProgram()

    if interface == "auto":
        params = []
        for tape in tapes:
            params.extend(tape.get_parameters(trainable_only=False))
        interface = qml.math.get_interface(*params)
    if interface == "jax":
        try:  # pragma: no-cover
            from .jax import get_jax_interface_name
        except ImportError as e:  # pragma: no-cover
            raise qml.QuantumFunctionError(  # pragma: no-cover
                "jax not found. Please install the latest "  # pragma: no-cover
                "version of jax to enable the 'jax' interface."  # pragma: no-cover
            ) from e  # pragma: no-cover

        interface = get_jax_interface_name(tapes)

    gradient_kwargs = gradient_kwargs or {}
    config = config or _get_execution_config(gradient_fn, grad_on_execution, interface, device)

    if isinstance(cache, bool) and cache:
        # cache=True: create a LRUCache object
        cache = LRUCache(maxsize=cachesize)
        setattr(cache, "_persistent_cache", False)

    expand_fn = _preprocess_expand_fn(expand_fn, device, max_expansion)

    # changing this set of conditions causes a bunch of tests to break.
    no_interface_boundary_required = interface is None or gradient_fn in {None, "backprop"}
    device_supports_interface_data = no_interface_boundary_required and (
        interface is None
        or gradient_fn == "backprop"
        or getattr(device, "short_name", "") == "default.mixed"
        or "passthru_interface" in getattr(device, "capabilities", lambda: {})()
    )

    inner_execute = _make_inner_execute(
        device,
        override_shots,
        cache,
        expand_fn,
        config,
        numpy_only=not device_supports_interface_data,
    )

    # moved to its own explicit step so it will be easier to remove
    def inner_execute_with_empty_jac(tapes, **_):
        return (inner_execute(tapes), [])

    execute_fn = inner_execute_with_empty_jac

    #### Executing the configured setup #####

    if isinstance(device, qml.devices.Device):
        if not device_batch_transform:
            warnings.warn(
                "device batch transforms cannot be turned off with the new device interface.",
                UserWarning,
            )
        tapes, post_processing = transform_program(tapes)
    else:
        # TODO: Remove once old device are removed
        tapes, program_post_processing = transform_program(tapes)
        tapes, program_pre_processing, config = _batch_transform(
            tapes, device, config, override_shots, device_batch_transform
        )

        def post_processing(results):
            return program_post_processing(program_pre_processing(results))

    if transform_program.is_informative:
        return post_processing(tapes)

    # Exiting early if we do not need to deal with an interface boundary
    if no_interface_boundary_required:
        results = inner_execute(tapes)
        return post_processing(results)

    _grad_on_execution = False

    if config.use_device_gradient:
        # must be new device if this is specified as true
        _grad_on_execution = config.grad_on_execution

        if config.grad_on_execution:

            def execute_fn(internal_tapes):
                """A partial function that wraps the execute_and_compute_derivatives method of the device.

                Closure Variables:
                    device: The device to execute on
                    config: the ExecutionConfig that specifies how to perform the simulations.
                """
                numpy_tapes = tuple(
                    qml.transforms.convert_to_numpy_parameters(t) for t in internal_tapes
                )
                return device.execute_and_compute_derivatives(numpy_tapes, config)

            gradient_fn = None

        else:

            def execute_fn(internal_tapes) -> Tuple[ResultBatch, Tuple]:
                """A wrapper around device.execute that adds an empty tuple instead of derivatives.

                Closure Variables:
                    device: the device to execute on
                    config: the ExecutionConfig that specifies how to perform the simulations.
                """
                numpy_tapes = tuple(
                    qml.transforms.convert_to_numpy_parameters(t) for t in internal_tapes
                )
                return (device.execute(numpy_tapes, config), tuple())

            def gradient_fn(internal_tapes):
                """A partial function that wraps compute_derivatives method of the device.

                Closure Variables:
                    device: the device to execute on
                    config: the ExecutionConfig that specifies how to take the derivative.
                """
                numpy_tapes = tuple(
                    qml.transforms.convert_to_numpy_parameters(t) for t in internal_tapes
                )
                return device.compute_derivatives(numpy_tapes, config)

    elif gradient_fn == "device":
        # gradient function is a device method

        # Expand all tapes as per the device's expand function here.
        # We must do this now, prior to the interface, to ensure that
        # decompositions with parameter processing is tracked by the
        # autodiff frameworks.
        tapes = [expand_fn(t) for t in tapes]

        if gradient_kwargs.get("method", "") == "adjoint_jacobian":
            tapes = _adjoint_jacobian_expansion(tapes, grad_on_execution, interface, max_expansion)

        # grad on execution or best was chosen
        if grad_on_execution is True or grad_on_execution == "best":
            # replace the forward execution function to return
            # both results and gradients
            def device_execute_and_gradients(internal_tapes, **gradient_kwargs):
                numpy_tapes = tuple(
                    qml.transforms.convert_to_numpy_parameters(t) for t in internal_tapes
                )
                return set_shots(device, override_shots)(device.execute_and_gradients)(
                    numpy_tapes, **gradient_kwargs
                )

            execute_fn = device_execute_and_gradients
            gradient_fn = None
            _grad_on_execution = True

        else:
            # need to override to have no cache
            inner_execute = _make_inner_execute(device, override_shots, cache=None)

            def inner_execute_with_empty_jac(tapes, **_):
                return (inner_execute(tapes), [])

            execute_fn = inner_execute_with_empty_jac

            # replace the backward gradient computation
            # use qml.interfaces so that mocker can spy on it during testing
            gradient_fn_with_shots = set_shots(device, override_shots)(device.gradients)
            cached_gradient_fn = qml.interfaces.cache_execute(
                gradient_fn_with_shots,
                cache,
                pass_kwargs=True,
                return_tuple=False,
            )

            def device_gradient_fn(inner_tapes, **gradient_kwargs):
                numpy_tapes = tuple(
                    qml.transforms.convert_to_numpy_parameters(t) for t in inner_tapes
                )
                return cached_gradient_fn(numpy_tapes, **gradient_kwargs)

            gradient_fn = device_gradient_fn

            # Adjoint Jacobian with backward pass and jitting needs the original circuit output state which
            # can not be reused from the device if `grad_on_execution is False`.
            if interface == "jax-jit":
                use_device_state = gradient_kwargs.get("use_device_state", None)
                if use_device_state:
                    gradient_kwargs["use_device_state"] = False

    elif grad_on_execution is True:
        # In "forward" mode, gradients are automatically handled
        # within execute_and_gradients, so providing a gradient_fn
        # in this case would have ambiguous behaviour.
        raise ValueError("Gradient transforms cannot be used with grad_on_execution=True")

    ml_boundary_execute = _get_ml_boundary_execute(interface, _grad_on_execution)
    results = ml_boundary_execute(
        tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=max_diff
    )

    return post_processing(results)


def _get_execution_config(gradient_fn, grad_on_execution, interface, device):
    """Helper function to get the execution config."""
    if gradient_fn is None:
        _gradient_method = None
    elif isinstance(gradient_fn, str):
        _gradient_method = gradient_fn
    else:
        _gradient_method = "gradient-transform"
    config = qml.devices.ExecutionConfig(
        interface=interface,
        gradient_method=_gradient_method,
        grad_on_execution=None if grad_on_execution == "best" else grad_on_execution,
    )
    if isinstance(device, qml.devices.Device):
        _, config = device.preprocess(config)
    return config

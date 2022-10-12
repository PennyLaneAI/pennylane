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
# pylint: disable=too-many-statements, invalid-unary-operand-type

import inspect
import warnings
from contextlib import _GeneratorContextManager
from functools import wraps
from typing import Callable, Sequence

from cachetools import LRUCache

import pennylane as qml
from pennylane import Device
from pennylane.tape import QuantumTape

from .set_shots import set_shots

INTERFACE_MAP = {
    None: "Numpy",
    "autograd": "autograd",
    "numpy": "autograd",
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
    tapes: Sequence[QuantumTape], mode: str, interface: str, max_expansion: int
):
    """Performs adjoint jacobian specific expansion.  Expands so that every
    trainable operation has a generator.

    TODO: Let the device specify any gradient-specific expansion logic.  This
    function will be removed once the device-support pipeline is improved.
    """
    if mode == "forward" and INTERFACE_MAP[interface] == "jax":
        # qml.math.is_trainable doesn't work with jax on the forward pass
        non_trainable = qml.operation.has_nopar
    else:
        non_trainable = ~qml.operation.is_trainable

    stop_at = ~qml.operation.is_measurement & (
        non_trainable  # pylint: disable=unsupported-binary-operation
        | qml.operation.has_unitary_gen
    )
    for i, tape in enumerate(tapes):
        if any(not stop_at(op) for op in tape.operations):
            tapes[i] = tape.expand(stop_at=stop_at, depth=max_expansion)

    return tapes


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
            res = fn(tapes, **kwargs)
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

                # Introspect the set_shots decorator of the input function:
                #   warn the user in case of finite shots with cached results
                finite_shots = False

                closure = inspect.getclosurevars(fn).nonlocals
                if "original_fn" in closure:  # deal with expand_fn wrapper above
                    closure = inspect.getclosurevars(closure["original_fn"]).nonlocals

                # retrieve the captured context manager instance (for set_shots)
                if "self" in closure and isinstance(closure["self"], _GeneratorContextManager):
                    # retrieve the shots from the arguments or device instance
                    if closure["self"].func.__name__ == "set_shots":
                        dev, shots = closure["self"].args
                        shots = dev.shots if shots is False else shots
                        finite_shots = isinstance(shots, int)

                if finite_shots and getattr(cache, "_persistent_cache", True):
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
            res = fn(execution_tapes.values(), **kwargs)

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


def _execute_new(
    tapes: Sequence[QuantumTape],
    device: Device,
    gradient_fn: Callable = None,
    interface="autograd",
    mode="best",
    gradient_kwargs=None,
    cache=True,
    cachesize=10000,
    max_diff=1,
    override_shots=False,
    expand_fn="device",
    max_expansion=10,
    device_batch_transform=True,
):
    """New function to execute a batch of tapes on a device in an autodifferentiable-compatible manner. More cases will be added,
    during the project. The current version is supporting forward execution for Numpy and does not support shot vectors.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        gradient_fn (None or callable): The gradient transform function to use
            for backward passes. If "device", the device will be queried directly
            for the gradient (if supported).
        interface (str): The interface that will be used for classical autodifferentiation.
            This affects the types of parameters that can exist on the input tapes.
            Available options include ``autograd``, ``torch``, ``tf``, and ``jax``.
        mode (str): Whether the gradients should be computed on the forward
            pass (``forward``) or the backward pass (``backward``). Only applies
            if the device is queried for the gradient; gradient transform
            functions available in ``qml.gradients`` are only supported on the backward
            pass.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes
        cache (bool): Whether to cache evaluations. This can result in
            a significant reduction in quantum evaluations during gradient computations.
        cachesize (int): the size of the cache
        max_diff (int): If ``gradient_fn`` is a gradient transform, this option specifies
            the maximum number of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.
        expand_fn (function): Tape expansion function to be called prior to device execution.
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
            with qml.tape.QuantumTape() as tape1:
                qml.RX(params[0], wires=0)
                qml.RY(params[1], wires=0)
                qml.expval(qml.PauliZ(0))

            with qml.tape.QuantumTape() as tape2:
                qml.RX(params[2], wires=0)
                qml.RY(x[0], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.probs(wires=0)

            tapes = [tape1, tape2]

            # execute both tapes in a batch on the given device
            res = qml.execute(tapes, dev, qml.gradients.param_shift, max_diff=2)

            return res[0][0] + res[1][0, 0] - res[1][0, 1]

    In this cost function, two **independent** quantum tapes are being
    constructed; one returning an expectation value, the other probabilities.
    We then batch execute the two tapes, and reduce the results to obtain
    a scalar.

    Let's execute this cost function while tracking the gradient:

    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> x = np.array([0.5], requires_grad=True)
    >>> cost_fn(params, x)
    tensor(1.93050682, requires_grad=True)

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
    gradient_kwargs = gradient_kwargs or {}

    if device_batch_transform:
        tapes, batch_fn = qml.transforms.map_batch_transform(device.batch_transform, tapes)
    else:
        batch_fn = lambda res: res  # pragma: no cover

    if isinstance(cache, bool) and cache:
        # cache=True: create a LRUCache object
        cache = LRUCache(maxsize=cachesize)
        setattr(cache, "_persistent_cache", False)

    batch_execute = set_shots(device, override_shots)(device.batch_execute)

    if expand_fn == "device":
        expand_fn = lambda tape: device.expand_fn(tape, max_expansion=max_expansion)

    if gradient_fn is None:
        # don't unwrap if it's an interface device
        if "passthru_interface" in device.capabilities() or device.short_name == "default.mixed":
            return batch_fn(
                qml.interfaces.cache_execute(
                    batch_execute, cache, return_tuple=False, expand_fn=expand_fn
                )(tapes)
            )
        with qml.tape.Unwrap(*tapes):
            res = qml.interfaces.cache_execute(
                batch_execute, cache, return_tuple=False, expand_fn=expand_fn
            )(tapes)

        return batch_fn(res)

    if gradient_fn == "backprop" or interface is None:
        return batch_fn(
            qml.interfaces.cache_execute(
                batch_execute, cache, return_tuple=False, expand_fn=expand_fn
            )(tapes)
        )

    # the default execution function is batch_execute
    execute_fn = qml.interfaces.cache_execute(batch_execute, cache, expand_fn=expand_fn)
    _mode = "backward"

    if gradient_fn == "device":
        # gradient function is a device method

        # Expand all tapes as per the device's expand function here.
        # We must do this now, prior to the interface, to ensure that
        # decompositions with parameter processing is tracked by the
        # autodiff frameworks.
        for i, tape in enumerate(tapes):
            tapes[i] = expand_fn(tape)

        if gradient_kwargs.get("method", "") == "adjoint_jacobian":
            tapes = _adjoint_jacobian_expansion(tapes, mode, interface, max_expansion)

        if mode in ("forward", "best"):
            # replace the forward execution function to return
            # both results and gradients
            execute_fn = set_shots(device, override_shots)(device.execute_and_gradients)
            gradient_fn = None
            _mode = "forward"

        elif mode == "backward":
            # disable caching on the forward pass
            execute_fn = qml.interfaces.cache_execute(batch_execute, cache=None)

            # replace the backward gradient computation
            gradient_fn = qml.interfaces.cache_execute(
                set_shots(device, override_shots)(device.gradients),
                cache,
                pass_kwargs=True,
                return_tuple=False,
            )

    elif mode == "forward":
        # In "forward" mode, gradients are automatically handled
        # within execute_and_gradients, so providing a gradient_fn
        # in this case would have ambiguous behaviour.
        raise ValueError("Gradient transforms cannot be used with mode='forward'")

    try:
        mapped_interface = INTERFACE_MAP[interface]
    except KeyError as e:
        raise ValueError(
            f"Unknown interface {interface}. Supported " f"interfaces are {SUPPORTED_INTERFACES}"
        ) from e
    try:
        if mapped_interface == "autograd":
            from .autograd import execute as _execute
        elif mapped_interface == "tf":
            import tensorflow as tf  # pragma: no cover

            if not tf.executing_eagerly() or "autograph" in interface:  # pragma: no cover
                from .tensorflow_autograph import execute as _execute  # pragma: no cover
            else:
                from .tensorflow import execute as _execute  # pragma: no cover
        elif mapped_interface == "torch":
            from .torch import execute as _execute  # pragma: no cover
        else:  # is jax
            _execute = _get_jax_execute_fn(interface, tapes)  # pragma: no cover
    except ImportError as e:
        raise qml.QuantumFunctionError(
            f"{mapped_interface} not found. Please install the latest "
            f"version of {mapped_interface} to enable the '{mapped_interface}' interface."
        ) from e

    res = _execute(
        tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=max_diff, mode=_mode
    )

    return batch_fn(res)


def execute(
    tapes: Sequence[QuantumTape],
    device: Device,
    gradient_fn: Callable = None,
    interface="autograd",
    mode="best",
    gradient_kwargs=None,
    cache=True,
    cachesize=10000,
    max_diff=1,
    override_shots=False,
    expand_fn="device",
    max_expansion=10,
    device_batch_transform=True,
):
    """Execute a batch of tapes on a device in an autodifferentiable-compatible manner.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        gradient_fn (None or callable): The gradient transform function to use
            for backward passes. If "device", the device will be queried directly
            for the gradient (if supported).
        interface (str): The interface that will be used for classical autodifferentiation.
            This affects the types of parameters that can exist on the input tapes.
            Available options include ``autograd``, ``torch``, ``tf``, and ``jax``.
        mode (str): Whether the gradients should be computed on the forward
            pass (``forward``) or the backward pass (``backward``). Only applies
            if the device is queried for the gradient; gradient transform
            functions available in ``qml.gradients`` are only supported on the backward
            pass.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes
        cache (bool): Whether to cache evaluations. This can result in
            a significant reduction in quantum evaluations during gradient computations.
        cachesize (int): the size of the cache
        max_diff (int): If ``gradient_fn`` is a gradient transform, this option specifies
            the maximum number of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.
        expand_fn (function): Tape expansion function to be called prior to device execution.
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
            with qml.tape.QuantumTape() as tape1:
                qml.RX(params[0], wires=0)
                qml.RY(params[1], wires=0)
                qml.expval(qml.PauliZ(0))

            with qml.tape.QuantumTape() as tape2:
                qml.RX(params[2], wires=0)
                qml.RY(x[0], wires=1)
                qml.CNOT(wires=[0, 1])
                qml.probs(wires=0)

            tapes = [tape1, tape2]

            # execute both tapes in a batch on the given device
            res = qml.execute(tapes, dev, qml.gradients.param_shift, max_diff=2)

            return res[0][0] + res[1][0, 0] - res[1][0, 1]

    In this cost function, two **independent** quantum tapes are being
    constructed; one returning an expectation value, the other probabilities.
    We then batch execute the two tapes, and reduce the results to obtain
    a scalar.

    Let's execute this cost function while tracking the gradient:

    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> x = np.array([0.5], requires_grad=True)
    >>> cost_fn(params, x)
    tensor(1.93050682, requires_grad=True)

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
    if qml.active_return():
        return _execute_new(
            tapes,
            device,
            gradient_fn,
            interface=interface,
            mode=mode,
            gradient_kwargs=gradient_kwargs,
            cache=cache,
            cachesize=cachesize,
            max_diff=max_diff,
            override_shots=override_shots,
            expand_fn=expand_fn,
            max_expansion=max_expansion,
            device_batch_transform=device_batch_transform,
        )

    gradient_kwargs = gradient_kwargs or {}

    if device_batch_transform:
        tapes, batch_fn = qml.transforms.map_batch_transform(device.batch_transform, tapes)
    else:
        batch_fn = lambda res: res

    if isinstance(cache, bool) and cache:
        # cache=True: create a LRUCache object

        cache = LRUCache(maxsize=cachesize, getsizeof=lambda x: qml.math.shape(x)[0])
        setattr(cache, "_persistent_cache", False)

    batch_execute = set_shots(device, override_shots)(device.batch_execute)

    if expand_fn == "device":
        expand_fn = lambda tape: device.expand_fn(tape, max_expansion=max_expansion)

    if gradient_fn is None:
        # don't unwrap if it's an interface device
        if "passthru_interface" in device.capabilities():
            return batch_fn(
                qml.interfaces.cache_execute(
                    batch_execute, cache, return_tuple=False, expand_fn=expand_fn
                )(tapes)
            )
        with qml.tape.Unwrap(*tapes):
            res = qml.interfaces.cache_execute(
                batch_execute, cache, return_tuple=False, expand_fn=expand_fn
            )(tapes)

        return batch_fn(res)

    if gradient_fn == "backprop" or interface is None:
        return batch_fn(
            qml.interfaces.cache_execute(
                batch_execute, cache, return_tuple=False, expand_fn=expand_fn
            )(tapes)
        )

    # the default execution function is batch_execute
    execute_fn = qml.interfaces.cache_execute(batch_execute, cache, expand_fn=expand_fn)
    _mode = "backward"

    if gradient_fn == "device":
        # gradient function is a device method

        # Expand all tapes as per the device's expand function here.
        # We must do this now, prior to the interface, to ensure that
        # decompositions with parameter processing is tracked by the
        # autodiff frameworks.
        for i, tape in enumerate(tapes):
            tapes[i] = expand_fn(tape)

        if gradient_kwargs.get("method", "") == "adjoint_jacobian":
            tapes = _adjoint_jacobian_expansion(tapes, mode, interface, max_expansion)

        if mode in ("forward", "best"):
            # replace the forward execution function to return
            # both results and gradients
            execute_fn = set_shots(device, override_shots)(device.execute_and_gradients)
            gradient_fn = None
            _mode = "forward"

        elif mode == "backward":
            # disable caching on the forward pass
            execute_fn = qml.interfaces.cache_execute(batch_execute, cache=None)

            # replace the backward gradient computation
            gradient_fn = qml.interfaces.cache_execute(
                set_shots(device, override_shots)(device.gradients),
                cache,
                pass_kwargs=True,
                return_tuple=False,
            )

    elif mode == "forward":
        # In "forward" mode, gradients are automatically handled
        # within execute_and_gradients, so providing a gradient_fn
        # in this case would have ambiguous behaviour.
        raise ValueError("Gradient transforms cannot be used with mode='forward'")

    try:
        mapped_interface = INTERFACE_MAP[interface]
    except KeyError as e:
        raise ValueError(
            f"Unknown interface {interface}. Supported " f"interfaces are {SUPPORTED_INTERFACES}"
        ) from e
    try:
        if mapped_interface == "autograd":
            from .autograd import execute as _execute
        elif mapped_interface == "tf":
            import tensorflow as tf

            if not tf.executing_eagerly() or "autograph" in interface:
                from .tensorflow_autograph import execute as _execute
            else:
                from .tensorflow import execute as _execute
        elif mapped_interface == "torch":
            from .torch import execute as _execute
        else:  # is jax
            _execute = _get_jax_execute_fn(interface, tapes)
    except ImportError as e:
        raise qml.QuantumFunctionError(
            f"{mapped_interface} not found. Please install the latest "
            f"version of {mapped_interface} to enable the '{mapped_interface}' interface."
        ) from e

    res = _execute(
        tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=max_diff, mode=_mode
    )

    return batch_fn(res)


def _get_jax_execute_fn(interface: str, tapes: Sequence[QuantumTape]):
    """Auxiliary function to determine the execute function to use with the JAX
    interface."""

    # The most general JAX interface was specified, automatically determine if
    # support for jitting is needed by swapping to "jax-jit" or "jax-python"
    if interface == "jax":
        from .jax import get_jax_interface_name

        interface = get_jax_interface_name(tapes)

    if interface == "jax-jit":
        from .jax_jit import execute as _execute
    else:
        from .jax import execute as _execute
    return _execute

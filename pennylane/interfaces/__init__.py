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
This subpackage defines functions for interfacing devices' batch execution
capabilities with different machine learning libraries.
"""
# pylint: disable=import-outside-toplevel,too-many-arguments,too-many-branches,protected-access
import contextlib
from functools import wraps
import itertools

from cachetools import LRUCache
import numpy as np

import pennylane as qml


INTERFACE_NAMES = {
    "NumPy": (None,),
    "Autograd": ("autograd", "numpy"),  # for backwards compatibility
    "JAX": ("jax", "jax-jit", "jax-python", "JAX"),
    "PyTorch": ("torch", "pytorch"),
    "TensorFlow": ("tf", "tensorflow", "tensorflow-autograph", "tf-autograph"),
}
"""dict[str, str]: maps allowed interface strings to the name of the interface"""

SUPPORTED_INTERFACES = list(itertools.chain(*INTERFACE_NAMES.values()))


class InterfaceUnsupportedError(NotImplementedError):
    """Exception raised when features not supported by an interface are
    attempted to be used."""


@contextlib.contextmanager
def set_shots(device, shots):
    """Context manager to temporarily change the shots
    of a device.

    This context manager can be used in two ways.

    As a standard context manager:

    >>> dev = qml.device("default.qubit", wires=2, shots=None)
    >>> with set_shots(dev, shots=100):
    ...     print(dev.shots)
    100
    >>> print(dev.shots)
    None

    Or as a decorator that acts on a function that uses the device:

    >>> set_shots(dev, shots=100)(lambda: dev.shots)()
    100
    """
    if shots == device.shots:
        yield
        return

    original_shots = device.shots
    original_shot_vector = device._shot_vector

    try:
        if shots is not False and device.shots != shots:
            device.shots = shots
        yield
    finally:
        device.shots = original_shots
        device._shot_vector = original_shot_vector


def cache_execute(fn, cache, pass_kwargs=False, return_tuple=True, expand_fn=None):
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

        def fn(tapes, **kwargs):  # pylint: disable=function-redefined
            tapes = [expand_fn(tape) for tape in tapes]
            return original_fn(tapes, **kwargs)

    @wraps(fn)
    def wrapper(tapes, **kwargs):

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


def execute(
    tapes,
    device,
    gradient_fn,
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
        list[list[float]]: A nested list of tape results. Each element in
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
        batch_fn = lambda res: res

    if isinstance(cache, bool) and cache:
        # cache=True: create a LRUCache object
        cache = LRUCache(maxsize=cachesize, getsizeof=lambda x: qml.math.shape(x)[0])

    batch_execute = set_shots(device, override_shots)(device.batch_execute)

    if expand_fn == "device":
        expand_fn = lambda tape: device.expand_fn(tape, max_expansion=max_expansion)

    if gradient_fn is None:
        # don't unwrap if it's an interface device
        if "passthru_interface" in device.capabilities():
            return batch_fn(
                cache_execute(batch_execute, cache, return_tuple=False, expand_fn=expand_fn)(tapes)
            )
        with qml.tape.Unwrap(*tapes):
            res = cache_execute(batch_execute, cache, return_tuple=False, expand_fn=expand_fn)(
                tapes
            )

        return batch_fn(res)

    if gradient_fn == "backprop" or interface is None:
        return batch_fn(
            cache_execute(batch_execute, cache, return_tuple=False, expand_fn=expand_fn)(tapes)
        )

    # the default execution function is batch_execute
    execute_fn = cache_execute(batch_execute, cache, expand_fn=expand_fn)
    _mode = "backward"

    if gradient_fn == "device":
        # gradient function is a device method

        # Expand all tapes as per the device's expand function here.
        # We must do this now, prior to the interface, to ensure that
        # decompositions with parameter processing is tracked by the
        # autodiff frameworks.
        for i, tape in enumerate(tapes):
            tapes[i] = expand_fn(tape)

        if mode in ("forward", "best"):
            # replace the forward execution function to return
            # both results and gradients
            execute_fn = set_shots(device, override_shots)(device.execute_and_gradients)
            gradient_fn = None
            _mode = "forward"

        elif mode == "backward":
            # disable caching on the forward pass
            execute_fn = cache_execute(batch_execute, cache=None)

            # replace the backward gradient computation
            gradient_fn = cache_execute(
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
        if interface in INTERFACE_NAMES["Autograd"]:
            from .autograd import execute as _execute
        elif interface in INTERFACE_NAMES["TensorFlow"]:
            import tensorflow as tf

            if not tf.executing_eagerly() or "autograph" in interface:
                from .tensorflow_autograph import execute as _execute
            else:
                from .tensorflow import execute as _execute

        elif interface in INTERFACE_NAMES["PyTorch"]:
            from .torch import execute as _execute
        elif interface in INTERFACE_NAMES["JAX"]:
            _execute = _get_jax_execute_fn(interface, tapes)
        else:
            raise ValueError(
                f"Unknown interface {interface}. Supported "
                f"interfaces are {SUPPORTED_INTERFACES}"
            )
    except ImportError as e:
        interface_name = [k for k, v in INTERFACE_NAMES.items() if interface in v][0]

        raise qml.QuantumFunctionError(
            f"{interface_name} not found. Please install the latest "
            f"version of {interface_name} to enable the '{interface}' interface."
        ) from e

    res = _execute(
        tapes, device, execute_fn, gradient_fn, gradient_kwargs, _n=1, max_diff=max_diff, mode=_mode
    )

    return batch_fn(res)


def _get_jax_execute_fn(interface, tapes):
    """Auxiliary function to determine the execute function to use with the JAX
    interface."""

    # The most general JAX interface was sepcified, automatically determine if
    # support for jitting is needed by swapping to "jax-jit" or "jax-python"
    if interface == "jax":
        from .jax import get_jax_interface_name

        interface = get_jax_interface_name(tapes)

    if interface == "jax-jit":
        from .jax_jit import execute as _execute
    else:
        from .jax import execute as _execute
    return _execute

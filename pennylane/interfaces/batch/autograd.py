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
This module contains functions for adding the Autograd interface
to a PennyLane Device class.
"""
# pylint: disable=protected-access,too-many-arguments
import contextlib

import autograd.extend
import autograd.builtins

from autograd.numpy.numpy_boxes import ArrayBox

import pennylane as qml
from pennylane import numpy as np


from .unwrap import UnwrapTape, batch_vjp


def get_trainable_params(tape):
    """Gets the trainable Autograd parameters of a tape.

    Trainable Autograd parameters are any tensor that have the ``requires_grad``
    attribute. If not provided, parameters are assumed to be non-trainable
    by default.

    Args:
        tape (.QuantumTape): a quantum tape

    Returns:
        tuple[set[int], list[Any]: a tuple returning both a set containing
        integers corresponding to tape parameters that are differentiable Autograd tensors,
        as well as the full list of tape parameters.

    **Example**

    >>> with qml.tape.QuantumTape() as tape:
    ...     qml.RX(np.array(0.1, requires_grad=True), wires=0)
    ...     qml.RY(0.2, wires=0)
    ...     qml.RZ(np.array(0.3, requires_grad=True), wires=0)
    >>> trainable_params, params = get_trainable_params(tape)
    >>> trainable_params
    {0, 2}
    >>> params
    [tensor(0.1, requires_grad=True), 0.2, tensor(0.3, requires_grad=True)])
    """
    params = []
    iterator = tape._par_info

    for p_idx in iterator:
        op = tape._par_info[p_idx]["op"]
        op_idx = tape._par_info[p_idx]["p_idx"]
        params.append(op.data[op_idx])

    trainable_params = set()

    for idx, p in enumerate(params):
        if getattr(p, "requires_grad", False) or isinstance(p, ArrayBox):
            trainable_params.add(idx)

    return trainable_params, params


def _unwrap_arraybox(arraybox, max_depth=None, _n=0):
    if max_depth is not None and _n == max_depth:
        return arraybox

    val = arraybox._value

    if isinstance(val, ArrayBox):
        return _unwrap_arraybox(val, max_depth=max_depth, _n=_n + 1)

    return val


def convert_to_numpy(tensors):
    """Converts any ArrayBox in a sequence to NumPy arrays."""
    unwrapped_tensors = []

    for t in tensors:
        if isinstance(t, np.tensor):
            unwrapped_tensors.append(t.numpy())
        elif isinstance(t, ArrayBox):
            unwrapped_tensors.append(_unwrap_arraybox(t))
        else:
            unwrapped_tensors.append(t)

    return unwrapped_tensors


def batch_execute(
    tapes, device, gradient_fn=None, cache=[], _n=1
):  # pylint: disable=dangerous-default-value
    """Execute a batch of tapes with NumPy parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        gradient_fn (None or callable): The gradient transform function to use
            for backward passes. The provided gradient transform should have
            the signature

            .. code-block:: python

                gradient_fn(tape, idx)

            where ``tape`` is the quantum function to differentiate, and
            ``idx`` is the trainable parameter to return the partial
            derivative of. The function should return a tuple
            ``(gradient_tape, fn)`` containing the list of generated tapes, in
            addition to a post-processing function to be applied to the
            evaluated tapes.

            If not provided, the 'best' gradient function will be determined.

        cache (list[dict[str, float]]): cache of tape parameter-shifts

    Returns:
        list[list[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.

    **Example**

    Consider the following cost function:

    .. code-block:: python

        def cost_fn(params, x, dev):
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
            res = batch_execute(tapes, dev)

            return res[0][0] + res[1][0, 0] - res[1][0, 1]

    In this cost function, two **independent** quantum tapes are being
    constructed; one returning an expectation value, the other probabilities.
    We then batch execute the two tapes, and reduce the results to obtain
    a scalar.

    Let's execute this cost function while tracking the gradient:

    >>> dev = qml.device("lightning.qubit", wires=2)
    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> x = np.array([0.5], requires_grad=True)
    >>> cost_fn(params, x)
    1.9305068163274222

    Since the ``batch_execute`` function is differentiable, we can
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
    if gradient_fn is None:
        gradient_fn = qml.transforms.gradients.qubit_parameter_shift.grad

    parameters = autograd.builtins.tuple(
        [autograd.builtins.list(t.get_parameters()) for t in tapes]
    )


    unsupported_op = lambda op: op.grad_recipe is None
    supported_op = lambda op: op.grad_recipe is not None
    trainable_op = lambda op: any(qml.math.requires_grad(p) for p in op.parameters)

    g_tapes = []

    for t in tapes:

        if any(unsupported_op(op) and trainable_op(op) for op in t.operations):

            g_tapes.append(t.expand(
                depth=10,
                stop_at=lambda obj: not isinstance(obj, qml.measure.MeasurementProcess)
                and ((supported_op(obj) and trainable_op(obj)) or not trainable_op(obj))
            ))
        else:
            g_tapes.append(t)

    return _batch_execute(
        parameters, tapes=tapes, gradient_tapes=g_tapes, device=device, gradient_fn=gradient_fn, cache=cache, _n=_n
    )


@autograd.extend.primitive
def _batch_execute(
    parameters, tapes=None, gradient_tapes=g_tapes, device=None, gradient_fn=None, cache=[], _n=1
):  # pylint: disable=dangerous-default-value,unused-argument
    """Autodifferentiable wrapper around ``Device.batch_execute``.

    The signature of this function is designed to workaround Autograd restrictions.
    Note that the ``parameters`` argument is dependent on the ``tapes`` argument;
    this function should always be called as follows:

    >>> parameters = [autograd.builtins.list(t.get_parameters()) for t in tapes])
    >>> parameters = autograd.builtins.tuple(parameters)
    >>> _batch_execute(parameters, tapes=tapes, device=device)

    In particular:

    - ``parameters`` is dependent on the provided tapes: always extract them as above
    - ``tapes`` is a *required* argument
    - ``device`` is a *required* argument

    The private argument ``_n`` is used to track nesting of derivatives, for example
    if the nth-order derivative is requested. Do not set this argument unless you
    understand the consequences!
    """
    with contextlib.ExitStack() as stack:
        unwrapped_tapes = [
            stack.enter_context(UnwrapTape(t, convert_to_numpy, get_trainable_params))
            for t in tapes
        ]
        res = device.batch_execute(unwrapped_tapes)

    return [np.tensor(r) for r in res]


def vjp(
    ans, parameters, tapes=None, device=None, gradient_fn=None, cache=[], _n=1
):  # pylint: disable=dangerous-default-value,unused-argument
    """Returns the vector-Jacobian product operator for a batch of quantum tapes.

    Args:
        ans (array): the result of the batch tape execution
        parameters (list[list[Any]]): Nested list of the quantum tape parameters.
            This argument should be generated from the provided list of tapes.
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        gradient_fn (None or callable): The gradient transform function to use
            for backward passes. The provided gradient transform should have
            the signature

            .. code-block:: python

                gradient_fn(tape, idx)

            where ``tape`` is the quantum function to differentiate, and
            ``idx`` is the trainable parameter to return the partial
            derivative of. The function should return a tuple
            ``(gradient_tape, fn)`` containing the list of generated tapes, in
            addition to a post-processing function to be applied to the
            evaluated tapes.

            If not provided, the 'best' gradient function will be determined.

        cache (list[dict[str, float]]): cache of tape parameter-shifts
        _n (int): a positive integer used to track nesting of derivatives, for example
            if the nth-order derivative is requested.

    Returns:
        function: this function accepts the backpropagation
        gradient output vector, and computes the vector-Jacobian product
    """

    def grad_fn(dy):
        """Returns the vector-Jacobian product with given
        parameter values p and output gradient dy"""
        vjps = batch_vjp(
            dy,
            tapes,
            batch_execute,
            gradient_fn,
            reduction="append",
            device=device,
            cache=cache,
            _n=_n + 1,
        )
        return [np.asarray(_unwrap_arraybox(v, max_depth=_n)) for v in vjps]

    return grad_fn


autograd.extend.defvjp(_batch_execute, vjp, argnums=[0])

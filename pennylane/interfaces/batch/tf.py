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
This module contains functions for adding the TensorFlow interface
to a PennyLane Device class.
"""
# pylint: disable=protected-access
import contextlib

import tensorflow as tf
from tensorflow.python.eager.tape import should_record_backprop

import pennylane as qml


from .unwrap import UnwrapTape, batch_vjp


def get_trainable_params(tape):
    """Gets the trainable TensorFlow parameters of a tape.

    Trainable TensorFlow parameters are any tensor that is being watched by a
    TensorFlow ``GradientTape``. As a result, if this function is called
    outside of a ``GradientTape`` context, **no parameters will be
    marked as trainable**.

    Args:
        tape (.QuantumTape): a quantum tape

    Returns:
        tuple[set[int], list[Any]: a tuple returning both a set containing
        integers corresponding to tape parameters that are differentiable TensorFlow tensors,
        as well as the full list of tape parameters.

    **Example**

    >>> with tf.GradientTape():
    ...     with qml.tape.QuantumTape() as tape:
    ...         qml.RX(tf.Variable(0.1), wires=0)
    ...         qml.RY(tf.constant(0.2), wires=0)
    ...         qml.RZ(tf.Variable(0.3), wires=0)
    ...     trainable_params = get_trainable_params(tape)
    >>> trainable_params
    {0, 2}
    """
    params = tape.get_parameters(trainable_only=False)

    trainable_params = set()

    for idx, p in enumerate(params):
        if isinstance(p, (tf.Variable, tf.Tensor)) and should_record_backprop(
            [tf.convert_to_tensor(p)]
        ):
            trainable_params.add(idx)

    return trainable_params, params


def convert_to_numpy(tensors):
    """Converts any TensorFlow tensors in a sequence to NumPy arrays."""
    return [i.numpy() if isinstance(i, (tf.Variable, tf.Tensor)) else i for i in tensors]


@tf.custom_gradient
def _batch_execute(*parameters, **kwargs):  # pylint: disable=unused-argument
    """Implements the forward pass batch tape evaluation.

    The signature of this function` is designed to
    workaround ``@tf.custom_gradient`` restrictions.

    In particular:

    - Positional arguments **must** be TensorFlow tensors, so
      we extract the parameters from all tapes. To pass the other
      data, we pass keyword arguments. These keyword arguments
      are **required**, and must contain the following:

      * ``"tapes"``: the quantum tapes to batch evaluate
      * ``"device"``: the device to use to evaluate the tapes
      * ``"gradient_fn"``: The gradient transform function to use
        for backward passes.
      * ``"cache"``: the cache list

    Further, note that the ``parameters`` argument is dependent on the
    ``tapes``; this Function should always be called
    with the parameters extracted directly from the tapes as follows:

    >>> parameters = []
    >>> [parameters.extend(t.get_parameters()) for t in tapes])
    >>> _batch_execute(*parameters, tapes=tapes, device=device, ...)

    The private argument ``_n`` is used to track nesting of derivatives, for example
    if the nth-order derivative is requested. Do not set this argument unless you
    understand the consequences!
    """

    tapes = kwargs["tapes"]
    device = kwargs["device"]
    gradient_fn = kwargs["gradient_fn"]
    cache = kwargs.get("cache", [])
    _n = kwargs.get("_n", 1)

    with contextlib.ExitStack() as stack:
        unwrapped_tapes = [
            stack.enter_context(UnwrapTape(t, convert_to_numpy, get_trainable_params))
            for t in tapes
        ]
        device._cache = 1000000000
        res = device.batch_execute(unwrapped_tapes)
        # device._cache = False

    res = [tf.convert_to_tensor(r) for r in res]

    def grad_fn(*dy, **tfkwargs):
        vjps = batch_vjp(
            dy,
            tapes,
            batch_execute,
            gradient_fn,
            reduction="extend",
            device=device,
            cache=cache,
            _n=_n + 1,
        )
        variables = tfkwargs.get("variables", None)
        return (vjps, variables) if variables is not None else vjps

    return res, grad_fn


def batch_execute(
    tapes, device, gradient_fn=None, cache=[], _n=1
):  # pylint: disable=dangerous-default-value
    """Execute a batch of tapes with TensorFlow parameters on a device.

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
    >>> params = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float64)
    >>> x = tf.Variable([0.5], dtype=tf.float64)
    >>> with tf.GradientTape(persistent=True) as t1:
    ...     with tf.GradientTape(persistent=True) as t2:
    ...         res = cost_fn(params, x, dev)

    Printing the results:
    >>> res
    tf.Tensor(1.9305068163274222, shape=(), dtype=float64)

    Since the ``batch_execute`` function is differentiable, we can
    also compute the gradient:

    >>> grad = t2.gradient(res, [params, x])
    >>> grad
    [<tf.Tensor: shape=(3,), dtype=float64, numpy=array([-0.0978434 , -0.19767681, -0.29552021])>,
     <tf.Tensor: shape=(1,), dtype=float64, numpy=array([5.37764278e-17])>]

    Finally, we can also compute any nth-order derivative. Let's compute the Jacobian
    of the gradient (that is, the Hessian):

    >>> t1.jacobian(grad[0], params, experimental_use_pfor=False)
    tf.Tensor(
    [[-0.97517033  0.01983384  0.        ]
     [ 0.01983384 -0.97517033  0.        ]
     [ 0.          0.         -0.95533649]], shape=(3, 3), dtype=float64)
    """
    if gradient_fn is None:
        gradient_fn = qml.transforms.gradients.qubit_parameter_shift.grad

    parameters = []
    for t in tapes:
        parameters.extend(t.get_parameters())

    kwargs = dict(tapes=tapes, device=device, gradient_fn=gradient_fn, cache=cache, _n=_n)
    return _batch_execute(*parameters, **kwargs)

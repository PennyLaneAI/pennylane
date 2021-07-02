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
import copy
import contextlib

import tensorflow as tf
from tensorflow.python.eager.tape import should_record_backprop

import pennylane as qml


def conversion_func(value, dtype=None, name=None, as_ref=False):  # pylint: disable=unused-argument
    """To convert a tape to a tf.Tensor, simply stack the parameters
    of the tape."""
    return tf.stack(value.get_parameters())


# Register the tf.Tensor conversion function for quantum tapes
tf.register_tensor_conversion_function(qml.tape.QuantumTape, conversion_func, priority=100)


# Register the tf.Tensor conversion function for devices.
# Here, we simply treat quantum devices as a constant.
tf.register_tensor_conversion_function(
    qml.Device, lambda *args, **kwargs: tf.constant(0.1), priority=100
)


def get_trainable_params(tape):
    """Gets the trainable TensorFlow parameters of a tape.

    Trainable TensorFlow parameters are any tensor that is being watched by a
    TensorFlow ``GradientTape``. As a result, if this function is called
    outside of a ``GradientTape`` context, **no parameters will be
    marked as trainable**.

    Args:
        tape (.QuantumTape): a quantum tape

    Returns:
        set[int]: a set containing integers corresponding to tape
        parameters that are differentiable TensorFlow tensors

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

    return trainable_params


def convert_to_numpy(tensors):
    """Converts any TensorFlow tensors in a sequence to NumPy arrays."""
    return [i.numpy() if isinstance(i, (tf.Variable, tf.Tensor)) else i for i in tensors]


class UnwrapTape:
    """A context manager that unwraps a tape with TensorFlow parameters
    to NumPy arrays.

    Args:
        tape (.QuantumTape): the quantum tape to unwrap

    Returns:
        .QuantumTape: the unwrapped quantum tape

    **Example**

    >>> with tf.GradientTape():
    ...     with qml.tape.QuantumTape() as tape:
    ...         qml.RX(tf.Variable(0.1), wires=0)
    ...         qml.RY(tf.constant(0.2), wires=0)
    ...         qml.RZ(tf.Variable(0.3), wires=0)
    ...     with UnwrapTapeTF(tape) as unwrapped_tape:
    ...         print("Trainable params:", unwrapped_tape.trainable_params)
    ...         print("Unwrapped params:", unwrapped_tape.get_parameters())
    Trainable params: {0, 2}
    Unwrapped params: [0.1, 0.3]
    >>> print("Original parameters:", tape.get_parameters())
    Original parameters: [<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.1>,
      <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.3>]
    """

    def __init__(self, tape):
        self.tape = tape
        self._original_params = None
        self._unwrapped_params = None

    def __enter__(self):
        self.tape.trainable_params = get_trainable_params(self.tape)

        self._original_params = self.tape.get_parameters(trainable_only=False)
        self._unwrapped_params = convert_to_numpy(self._original_params)

        self.tape.set_parameters(self._unwrapped_params, trainable_only=False)
        return self.tape

    def __exit__(self, exception_type, exception_value, traceback):
        self.tape.set_parameters(self._original_params, trainable_only=False)


@tf.custom_gradient
def batch_execute(tapes, device, gradient_fn=None, cache=[]):
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
        gradient_fn = qml.transforms.gradients.qubit_parameter_shift.expval_grad

    with contextlib.ExitStack() as stack:
        unwrapped_tapes = [stack.enter_context(UnwrapTape(t)) for t in tapes]
        res = device.batch_execute(unwrapped_tapes)

    res = [tf.convert_to_tensor(r) for r in res]

    def grad_fn(*dy, **tfkwargs):
        variables = tfkwargs.get("variables", None)

        reshape_info = []
        gradient_tapes = []
        processing_fns = []

        for t in tapes:
            processing_fns.append([])

            for idx, _ in enumerate(t.trainable_params):
                g_tapes, fn = gradient_fn(t, idx)

                reshape_info.append(len(g_tapes))
                gradient_tapes.extend(g_tapes)
                processing_fns[-1].append(fn)

        results = batch_execute(gradient_tapes, device, gradient_fn=None, cache=cache)
        vjp = []
        start = 0

        for t, d in zip(range(len(tapes)), dy):
            num_params = len(tapes[t].trainable_params)
            jac = []

            if num_params == 0:
                vjp.append(None)
                continue

            for fn, res_len in zip(processing_fns[t], reshape_info):
                # extract the correct results from the flat list
                res = results[start : start + res_len]
                start += res_len

                # postprocess results to compute the gradient
                jac.append(fn(res))

            dy_row = tf.reshape(d, [-1])
            jac = tf.transpose(tf.stack(jac))
            jac = tf.reshape(jac, [-1, num_params])
            jac = tf.cast(jac, tf.float64)

            vjp.append(tf.tensordot(dy_row, jac, axes=[[0], [0]]))

        vjp.append(None)
        return (vjp, variables) if variables is not None else vjp

    return res, grad_fn

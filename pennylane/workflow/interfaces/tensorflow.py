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

**How to bind a custom derivative with TensorFlow.**

To bind a custom derivative with tensorflow, you:

1. Decorate the function with ``tf.custom_gradient``
2. Alter the return to include a function that computes the VJP.

.. code-block:: python

    @tf.custom_gradient
    def f(x):
        print("forward pass")
        y = x**2

        def vjp(*dy):
            print("In the VJP function with: ", dy)
            print("eager? ", tf.executing_eagerly())
            return dy[0] * 2 * x
        return y, vjp

>>> x = tf.Variable(0.1)
>>> with tf.GradientTape(persistent=True) as tape:
...         y = f(x)
forward pass
>>> tape.gradient(y, x)
In the VJP function with:  (<tf.Tensor: shape=(), dtype=float32, numpy=1.0>,)
eager?  True
<tf.Tensor: shape=(), dtype=float32, numpy=0.2>
>>> tape.jacobian(y, x)
In the VJP function with:  (<tf.Tensor 'gradient_tape/Reshape:0' shape=() dtype=float32>,)
eager?  False
<tf.Tensor: shape=(), dtype=float32, numpy=0.2>
>>> tape.jacobian(y, x, experimental_use_pfor=False)
In the VJP function with:  (<tf.Tensor: shape=(), dtype=float32, numpy=1.0>,)
eager?  True
<tf.Tensor: shape=(), dtype=float32, numpy=0.2>

You will note in this example that the we printed out whether or not tensorflow was
in eager mode execution inside the VJP function or not. Whether or not eager mode
is enabled will effect what we can and cannot do inside the VJP function. Non-eager mode
(tracing mode) is enabled when we are taking a jacobian and not explicitly setting
``experimental_use_pfor=False``.

For example, when eager mode is disabled, we cannot cast the relevant parameters to numpy.
To circumvent this, we convert the parameters to numpy outside the VJP function, and then
use those numbers instead.

Due to the fact that the ``dy`` must be converted to numpy
for it to be used with a device-provided VJP, we are restricting the use of device VJP's to
when the VJP calculation is strictly eager. If someone wishes to calculate a full Jacobian
with ``device_vjp=True``, they must set ``experimental_use_pfor=False``.

Alternatively, we could have calculated the VJP inside a ``tf.py_function`` or ``tf.numpy_function``.
Unfortunately, we then get an extra call to the vjp function.

.. code-block:: python

    @tf.custom_gradient
    def f(x):
        y = x**2

        @tf.py_function(Tout=x.dtype)
        def vjp(*dy):
            print("In the VJP function with: ", dy)
            print("eager? ", tf.executing_eagerly())
            return dy[0] * 2 * x

        return y, vjp

>>> x = tf.Variable(0.1)
>>> with tf.GradientTape(persistent=True) as tape:
...         y = f(x)
>>> tape.jacobian(y, x)
In the VJP function with:  (<tf.Tensor: shape=(), dtype=float32, numpy=1.0>,)
eager?  True
In the VJP function with:  (<tf.Tensor: shape=(), dtype=float32, numpy=1.0>,)
eager?  True
<tf.Tensor: shape=(), dtype=float32, numpy=0.2>

As you can see, we got 2 calls to ``vjp`` instead of 1, and the calls have identical ``dy``. We do not want
to have to perform this extra call.

"""
# pylint: disable=unused-argument
import inspect
import logging
import warnings

import tensorflow as tf
from tensorflow.python.eager import context

import pennylane as qml
from pennylane.measurements import Shots

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def set_parameters_on_copy(tapes, params):
    """Copy a set of tapes with operations and set parameters"""
    return tuple(
        t.bind_new_parameters(a, list(range(len(a)))) for t, a in zip(tapes, params, strict=True)
    )


def _get_parameters_dtype(parameters):
    for p in parameters:
        if qml.math.get_interface(p) == "tensorflow":
            return p.dtype
    return None


_complex_dtype_map = {
    tf.float32: tf.complex64,
    tf.float64: tf.complex128,
}


def _to_tensors(x, dtype=None, complex_safe=False):
    """
    Convert a nested tuple structure of arrays into a nested tuple
    structure of TF tensors
    """
    if x is None or isinstance(x, dict):
        # qml.counts returns a dict (list of dicts when broadcasted), can't form a valid tensor
        return x

    if isinstance(x, (tuple, list)):
        return tuple(_to_tensors(x_, dtype=dtype, complex_safe=complex_safe) for x_ in x)

    if complex_safe and "complex" in qml.math.get_dtype_name(x):
        return tf.convert_to_tensor(x, dtype=_complex_dtype_map.get(dtype, dtype))
    return tf.convert_to_tensor(x, dtype=dtype)


def _recursive_conj(dy):
    if isinstance(dy, (tf.Variable, tf.Tensor)):
        return tf.math.conj(dy)
    return tuple(_recursive_conj(d) for d in dy)


def _res_restructured(res, tapes):
    """
    Reconstruct the nested tuple structure of the output of a list of tapes
    """
    start = 0
    res_nested = []
    for tape in tapes:
        tape_shots = tape.shots or Shots(1)
        shot_res_nested = []
        num_meas = len(tape.measurements)

        for _ in range(tape_shots.num_copies):
            shot_res = tuple(res[start : start + num_meas])
            shot_res_nested.append(shot_res[0] if num_meas == 1 else shot_res)
            start += num_meas

        res_nested.append(
            tuple(shot_res_nested) if tape_shots.has_partitioned_shots else shot_res_nested[0]
        )

    return tuple(res_nested)


def tf_execute(tapes, execute_fn, jpc, device=None, differentiable=False):
    """Execute a batch of tapes with TensorFlow parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        execute_fn (Callable[[Sequence[.QuantumTape]], ResultBatch]): a function that turns a batch of circuits into results
        jpc (JacobianProductCalculator): a class that can compute the vector Jacobian product (VJP)
            for the input tapes.

    Keyword Args:
        device=None: not used for tensorflow
        differentiable=False: whether or not the custom gradient vjp needs to be
            differentiable. Note that this keyword argument is unique to tensorflow.

    Returns:
        TensorLike: A nested tuple of tape results. Each element in
        the returned tuple corresponds in order to the provided tapes.
    """

    if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
        logger.debug(
            "Entry with (tapes=%s, execute_fn=%s, jpc=%s, differentiable=%s) called by %s",
            tapes,
            execute_fn,
            jpc,
            differentiable,
            "::L".join(str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]),
        )

    parameters = []
    numpy_params = []

    for tape in tapes:
        # store the trainable parameters
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

        parameters += [p for i, p in enumerate(params) if i in tape.trainable_params]

        # store all unwrapped parameters
        numpy_params.append(
            [i.numpy() if isinstance(i, (tf.Variable, tf.Tensor)) else i for i in params]
        )

    numpy_tapes, _ = qml.transforms.convert_to_numpy_parameters(tapes)

    tapes = tuple(tapes)

    # need to use same tapes for forward pass execution that we will use for the vjp
    # if we are using device derivatives (`not differentiable`) so we can find them in the cache
    params_dtype = _get_parameters_dtype(parameters)
    dtype = params_dtype if params_dtype in {tf.float64, tf.complex128} else None
    # make sure is float64 if data is float64.  May cause errors otherwise if device returns float32 precision
    res = _to_tensors(execute_fn(numpy_tapes), dtype=dtype, complex_safe=True)

    @tf.custom_gradient
    def custom_gradient_execute(*parameters):  # pylint:disable=unused-argument
        """An execution of tapes with VJP's registered with tensorflow.

        Args:
            *parameters (TensorLike): the trainable parameters for the tapes.

        Closure:
            tapes (tuple(QuantumTape)): the tapes to execute. Contains tensorflow parameters.
            numpy_tapes (tuple(QuantumTape)): tapes but with numpy parameters
            numpy_params (list(numpy.ndarray)): numpy versions of ``parameters``.
            jpc (JacobianProductCalculator): a class that can calculate the VJP.

        Returns:
            ResultBatch, Callable: the result of executing the tapes and a function capable of calculating the VJP.

        """

        def vjp_fn(*dy, **tfkwargs):
            # TF obeys the dL/dz_conj convention instead of the
            # dL/dz convention of PennyLane, autograd and jax. This converts between the formats
            dy = _recursive_conj(dy)

            if not differentiable:
                inner_tapes = numpy_tapes
            elif not context.executing_eagerly():
                warnings.warn(
                    "PennyLane does not provide the higher order derivatives of tensorflow jacobians."
                )
                # Using numpy_tapes instead seems to cause failures
                inner_tapes = set_parameters_on_copy(tapes, numpy_params)
            else:
                inner_tapes = tapes

            dy_dtype = dy[0].dtype

            # reconstruct the nested structure of dy
            nested_dy = _res_restructured(dy, tapes)

            try:
                vjps = jpc.compute_vjp(inner_tapes, nested_dy)
            except AttributeError as e:
                message = (
                    "device VJPs cannot be vectorized with tensorflow. "
                    "To use device_vjp=True, \n set experimental_use_pfor=False"
                    " as a keyword argument to GradientTape.jacobian\n and set persistent=True to GradientTape."
                )
                raise ValueError(message) from e

            vjps = _to_tensors(vjps, dtype=dy_dtype)
            if isinstance(vjps, tuple):
                extended_vjps = []
                for vjp in vjps:
                    if vjp is not None and 0 not in qml.math.shape(vjp):
                        extended_vjps.extend(qml.math.unstack(vjp))
                vjps = tuple(extended_vjps)

            variables = tfkwargs.get("variables")

            return (vjps, variables) if variables is not None else vjps

        return res, vjp_fn

    return custom_gradient_execute(*parameters)

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
"""This module contains a TensorFlow implementation of the :class:`~.DefaultQubit`
reference plugin.
"""
import numpy as np
import semantic_version

try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        raise ImportError("default.qubit.tf device requires TensorFlow>=2.0")

    from tensorflow.python.framework.errors_impl import InvalidArgumentError

    SUPPORTS_APPLY_OPS = semantic_version.match(">=2.3.0", tf.__version__)

except ImportError as e:
    raise ImportError("default.qubit.tf device requires TensorFlow>=2.0") from e


# With TF 2.1+, the legacy tf.einsum was renamed to _einsum_v1, while
# the replacement tf.einsum introduced the bug. This try-except block
# will dynamically patch TensorFlow versions where _einsum_v1 exists, to make it the
# default einsum implementation.
#
# For more details, see https://github.com/tensorflow/tensorflow/issues/37307
try:
    from tensorflow.python.ops.special_math_ops import _einsum_v1

    tf.einsum = _einsum_v1
except ImportError:
    pass

from . import DefaultQubit


class DefaultQubitTF(DefaultQubit):
    """Simulator plugin based on ``"default.qubit"``, written using TensorFlow.

    **Short name:** ``default.qubit.tf``

    This device provides a pure-state qubit simulator written using TensorFlow.
    As a result, it supports classical backpropagation as a means to compute the Jacobian. This can
    be faster than the parameter-shift rule for analytic quantum gradients
    when the number of parameters to be optimized is large.

    To use this device, you will need to install TensorFlow:

    .. code-block:: console

        pip install tensorflow>=2.0

    **Example**

    The ``default.qubit.tf`` is designed to be used with end-to-end classical backpropagation
    (``diff_method="backprop"``) with the TensorFlow interface. This is the default method
    of differentiation when creating a QNode with this device.

    Using this method, the created QNode is a 'white-box', and is
    tightly integrated with your TensorFlow computation:

    >>> dev = qml.device("default.qubit.tf", wires=1)
    >>> @qml.qnode(dev, interface="tf", diff_method="backprop")
    ... def circuit(x):
    ...     qml.RX(x[1], wires=0)
    ...     qml.Rot(x[0], x[1], x[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> weights = tf.Variable([0.2, 0.5, 0.1])
    >>> with tf.GradientTape() as tape:
    ...     res = circuit(weights)
    >>> print(tape.gradient(res, weights))
    tf.Tensor([-2.2526717e-01 -1.0086454e+00  1.3877788e-17], shape=(3,), dtype=float32)

    Autograph mode will also work when using classical backpropagation:

    >>> @tf.function
    ... def cost(weights):
    ...     return tf.reduce_sum(circuit(weights)**3) - 1
    >>> with tf.GradientTape() as tape:
    ...     res = cost(weights)
    >>> print(tape.gradient(res, weights))
    tf.Tensor([-3.5471588e-01 -1.5882589e+00  3.4694470e-17], shape=(3,), dtype=float32)

    There are a couple of things to keep in mind when using the ``"backprop"``
    differentiation method for QNodes:

    * You must use the ``"tf"`` interface for classical backpropagation, as TensorFlow is
      used as the device backend.

    * Only exact expectation values, variances, and probabilities are
      differentiable. Creation of a backpropagation QNode with finite shots
      raises an error. If you do try and take a derivative with finite shots on
      this device, the gradient will be ``None``.


    If you wish to use a different machine-learning interface, or prefer to calculate quantum
    gradients using the ``parameter-shift`` or ``finite-diff`` differentiation methods,
    consider using the ``default.qubit`` device instead.


    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means
            that the device returns analytical results.
            If ``shots > 0`` is used, the ``diff_method="backprop"``
            QNode differentiation method is not supported and it is recommended to consider
            switching device to ``default.qubit`` and using ``diff_method="parameter-shift"``.
    """

    name = "Default qubit (TensorFlow) PennyLane plugin"
    short_name = "default.qubit.tf"

    C_DTYPE = tf.complex128
    R_DTYPE = tf.float64
    _asarray = staticmethod(tf.convert_to_tensor)
    _dot = staticmethod(lambda x, y: tf.tensordot(x, y, axes=1))
    _abs = staticmethod(tf.abs)
    _reduce_sum = staticmethod(tf.reduce_sum)
    _reshape = staticmethod(tf.reshape)
    _flatten = staticmethod(lambda tensor: tf.reshape(tensor, [-1]))
    _gather = staticmethod(tf.gather)
    _einsum = staticmethod(tf.einsum)
    _cast = staticmethod(tf.cast)
    _transpose = staticmethod(tf.transpose)
    _tensordot = staticmethod(tf.tensordot)
    _conj = staticmethod(tf.math.conj)
    _real = staticmethod(tf.math.real)
    _imag = staticmethod(tf.math.imag)
    _roll = staticmethod(tf.roll)
    _stack = staticmethod(tf.stack)

    @staticmethod
    def _asarray(array, dtype=None):
        try:
            res = tf.convert_to_tensor(array, dtype=dtype)
        except InvalidArgumentError:
            res = tf.concat([tf.reshape(i, [-1]) for i in array], axis=0)

            if dtype is not None:
                res = tf.cast(res, dtype=dtype)

        return res

    def __init__(self, wires, *, shots=None, analytic=None):
        super().__init__(wires, shots=shots, cache=0, analytic=analytic)

        # prevent using special apply method for this gate due to slowdown in TF implementation
        del self._apply_ops["CZ"]

        # Versions of TF before 2.3.0 do not support using the special apply methods as they
        # raise an error when calculating the gradient. For versions of TF after 2.3.0,
        # special apply methods are also not supported when using more than 8 wires due to
        # limitations with TF slicing.
        if not SUPPORTS_APPLY_OPS or self.num_wires > 8:
            self._apply_ops = {}

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            passthru_interface="tf",
            supports_reversible_diff=False,
        )
        return capabilities

    @staticmethod
    def _scatter(indices, array, new_dimensions):
        indices = np.expand_dims(indices, 1)
        return tf.scatter_nd(indices, array, new_dimensions)

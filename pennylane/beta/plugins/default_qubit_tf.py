# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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

from pennylane import QubitStateVector, BasisState
from pennylane.operation import DiagonalOperation
from pennylane.plugins import DefaultQubit

try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        raise ImportError("default.tensor.tf device requires TensorFlow>=2.0")

except ImportError as e:
    raise ImportError("default.tensor.tf device requires TensorFlow>=2.0")


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

from . import tf_ops


class DefaultQubitTF(DefaultQubit):
    """Experimental simulator plugin based on ``"default.qubit"``, written
    using TensorFlow.

    **Short name:** ``default.qubit.tf``

    This experimental device provides a pure-state qubit simulator written using TensorFlow.
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

    There are a couple of things to keep in mind when using ``"backprop"`` mode:

    * You must use the ``"tf"`` interface for classical backpropagation, as TensorFlow is
      used as the device backend.

    * Only exact expectation values, variances, and probabilities are differentiable.
      When instantiating the device with ``analytic=False``, differentiating QNode
      outputs will result in ``None``.


    If you wish to use a different machine-learning interface, or prefer to calculate quantum
    gradients using the ``parameter-shift`` or ``finite-diff`` differentiation methods,

    consider using the ``default.qubit`` device instead.


    Args:
        wires (int): the number of wires to initialize the device with

        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to 1000 if not specified.
            If ``analytic == True``, then the number of shots is ignored
            in the calculation of expectation values and variances, and only controls the number
            of samples returned by ``sample``.
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically
    """

    name = "Default qubit (TensorFlow) PennyLane plugin"
    short_name = "default.qubit.tf"

    _capabilities = {
        "model": "qubit",
        "provides_jacobian": False,
        "passthru_interface": "tf",
    }

    parametric_ops = {
        "PhaseShift": tf_ops.PhaseShift,
        "RX": tf_ops.RX,
        "RY": tf_ops.RY,
        "RZ": tf_ops.RZ,
        "Rot": tf_ops.Rot,
        "CRX": tf_ops.CRX,
        "CRY": tf_ops.CRY,
        "CRZ": tf_ops.CRZ,
    }

    C_DTYPE = tf.complex128
    R_DTYPE = tf.float64
    _asarray = staticmethod(tf.convert_to_tensor)
    _dot = staticmethod(lambda x, y: tf.tensordot(x, y, axes=1))
    _abs = staticmethod(tf.abs)
    _reduce_sum = staticmethod(tf.reduce_sum)
    _reshape = staticmethod(tf.reshape)
    _flatten = staticmethod(lambda tensor: tf.reshape(tensor, [-1]))
    _gather = staticmethod(tf.gather)
    _scatter = staticmethod(tf.scatter_nd)
    _einsum = staticmethod(tf.einsum)
    _cast = staticmethod(tf.cast)

    @staticmethod
    def _scatter(indices, array, new_dimensions):
        indices = np.expand_dims(indices, 1)
        return tf.scatter_nd(indices, array, new_dimensions)

    def _apply_operation(self, operation):
        """Applies operations to the internal device state.

        Args:
            operation (~.Operation): operation to apply to the device
        """
        if isinstance(operation, QubitStateVector):
            self._apply_state_vector(operation.parameters[0], operation.wires)
            return

        if isinstance(operation, BasisState):
            self._apply_basis_state(operation.parameters[0], operation.wires)
            return

        matrix = None
        if operation.name in self.parametric_ops:
            matrix = self.parametric_ops[operation.name](*operation.parameters)

        if isinstance(operation, DiagonalOperation):
            matrix = operation.eigvals if matrix is None else matrix
            self._state = self._vec_vec_product(matrix, self._state, operation.wires)
        else:
            matrix = operation.matrix if matrix is None else matrix
            self._state = self._mat_vec_product_einsum(matrix, self._state, operation.wires)

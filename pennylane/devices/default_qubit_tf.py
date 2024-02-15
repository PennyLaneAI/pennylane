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
"""This module contains a TensorFlow implementation of the :class:`~.DefaultQubitLegacy`
reference plugin.
"""
import itertools
import numpy as np
import semantic_version

import pennylane as qml

try:
    import tensorflow as tf

    if tf.__version__[0] == "1":  # pragma: no cover
        raise ImportError("default.qubit.tf device requires TensorFlow>=2.0")

    from tensorflow.python.framework.errors_impl import InvalidArgumentError

    SUPPORTS_APPLY_OPS = semantic_version.match(">=2.3.0", tf.__version__)

except ImportError as e:  # pragma: no cover
    raise ImportError("default.qubit.tf device requires TensorFlow>=2.0") from e


from pennylane.math.single_dispatch import _ndim_tf
from . import DefaultQubitLegacy
from .default_qubit_legacy import tolerance


class DefaultQubitTF(DefaultQubitLegacy):
    """Simulator plugin based on ``"default.qubit.legacy"``, written using TensorFlow.

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
    ...     return qml.expval(qml.Z(0))
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
    _size = staticmethod(tf.size)
    _ndim = staticmethod(_ndim_tf)

    @staticmethod
    def _const_mul(constant, array):
        return constant * array

    @staticmethod
    def _asarray(array, dtype=None):
        if isinstance(array, tf.Tensor):
            if dtype is None or dtype == array.dtype:
                return array
            return tf.cast(array, dtype)

        try:
            res = tf.convert_to_tensor(array, dtype)
        except InvalidArgumentError:
            axis = 0
            res = tf.concat([tf.reshape(i, [-1]) for i in array], axis)

            if dtype is not None:
                res = tf.cast(res, dtype)

        return res

    def __init__(self, wires, *, shots=None, analytic=None):
        r_dtype = tf.float64
        c_dtype = tf.complex128

        super().__init__(wires, shots=shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)

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
        capabilities.update(passthru_interface="tf")
        return capabilities

    @staticmethod
    def _scatter(indices, array, new_dimensions):
        indices = np.expand_dims(indices, 1)
        return tf.scatter_nd(indices, array, new_dimensions)

    def _get_batch_size(self, tensor, expected_shape, expected_size):
        """Determine whether a tensor has an additional batch dimension for broadcasting,
        compared to an expected_shape. Differs from QubitDevice implementation by the
        exception made for abstract tensors."""
        try:
            size = self._size(tensor)
            ndim = qml.math.ndim(tensor)
            if ndim > len(expected_shape) or size > expected_size:
                return size // expected_size

        except (ValueError, tf.errors.OperatorNotAllowedInGraphError) as err:
            # This except clause covers the usage of tf.function, which is not compatible
            # with `DefaultQubit._get_batch_size`
            if not qml.math.is_abstract(tensor):
                raise err

        return None

    def _apply_state_vector(self, state, device_wires):
        """Initialize the internal state vector in a specified state.

        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state

        This implementation only adds a check for parameter broadcasting when initializing
        a quantum state on subsystems of the device.
        """

        # translate to wire labels used by device
        device_wires = self.map_wires(device_wires)
        dim = 2 ** len(device_wires)

        state = self._asarray(state, dtype=self.C_DTYPE)
        batch_size = self._get_batch_size(state, (dim,), dim)
        output_shape = [2] * self.num_wires
        if batch_size:
            output_shape.insert(0, batch_size)

        if not (state.shape in [(dim,), (batch_size, dim)]):
            raise ValueError("State vector must have shape (2**wires,) or (batch_size, 2**wires).")

        if not qml.math.is_abstract(state):
            norm = qml.math.linalg.norm(state, axis=-1, ord=2)
            if not qml.math.allclose(norm, 1.0, atol=tolerance):
                raise ValueError("Sum of amplitudes-squared does not equal one.")

        if len(device_wires) == self.num_wires and sorted(device_wires) == device_wires:
            # Initialize the entire device state with the input state
            self._state = self._reshape(state, output_shape)
            return

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(itertools.product([0, 1], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

        if batch_size:
            # This is the only logical branch that differs from DefaultQubitLegacy
            raise NotImplementedError(
                "Parameter broadcasting is not supported together with initializing the state "
                "vector of a subsystem of the device when using DefaultQubitTF."
            )
        # The following line is unchanged in the "else"-clause in DefaultQubitLegacy's implementation
        state = self._scatter(ravelled_indices, state, [2**self.num_wires])
        state = self._reshape(state, output_shape)
        self._state = self._asarray(state, dtype=self.C_DTYPE)

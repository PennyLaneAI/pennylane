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
import itertools
import functools
from string import ascii_letters as ABC

import numpy as np

from pennylane import QubitDevice, DeviceError, QubitStateVector, BasisState
from pennylane.operation import DiagonalOperation

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

from .tf_ops import C_DTYPE, R_DTYPE
from . import tf_ops

ABC_ARRAY = np.array(list(ABC))

# tolerance for numerical errors
tolerance = 1e-10


class DefaultQubitTF(QubitDevice):
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

    Autograd mode will also work when using classical backpropagation:

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
      output will result in ``None``.

    If you wish to use a different machine-learning interface, or prefer to calculate quantum
    gradients using the ``parameter-shift`` or ``finite-difff`` differentiation methods,
    it is recommended to use the ``default.qubit`` device instead.

    Args:
        wires (int): the number of modes to initialize the device in
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
    pennylane_requires = "0.10"
    version = "0.10.0"
    author = "Xanadu Inc."
    _capabilities = {
        "model": "qubit",
        "provides_jacobian": False,
        "passthru_interface": "tf",
    }

    operations = {
        "BasisState",
        "QubitStateVector",
        "QubitUnitary",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "Hadamard",
        "S",
        "T",
        "CNOT",
        "SWAP",
        "CSWAP",
        "Toffoli",
        "CZ",
        "PhaseShift",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
    }

    observables = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian", "Identity"}

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

    _asarray = staticmethod(tf.convert_to_tensor)

    def __init__(self, wires, *, shots=1000, analytic=True):
        # create the initial state
        state = np.zeros(2 ** wires, dtype=np.complex128)
        state[0] = 1
        state = tf.convert_to_tensor(state, dtype=C_DTYPE)

        # Internally, we store the state as a tensor of dimension [2]*wires
        self._state = tf.reshape(state, [2] * wires)
        self._pre_rotated_state = self._state

        # call QubitDevice init
        super().__init__(wires, shots, analytic)

    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []

        # apply the circuit operations
        for i, operation in enumerate(operations):

            if i > 0 and isinstance(operation, (QubitStateVector, BasisState)):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(operation.name, self.short_name)
                )

            self._apply_operation(operation)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            self._apply_operation(operation)

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

        if isinstance(operation, DiagonalOperation):
            self._state = self._apply_diagonal_unitary(operation.eigvals, operation.wires)
            return

        if operation.name in self.parametric_ops:
            matrix = self.parametric_ops[operation.name](*operation.parameters)
        else:
            matrix = operation.matrix

        self._state = self._apply_unitary(matrix, operation.wires)

    def _apply_state_vector(self, input_state, wires):
        """Initialize the internal state vector in a specified state.

        Args:
            input_state (array[complex]): normalized input state of length
                ``2**len(wires)``
            wires (list[int]): list of wires where the provided state should
                be initialized
        """
        if not np.isclose(tf.linalg.norm(input_state, 2), 1.0, atol=tolerance):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        n_state_vector = input_state.shape[0]

        if input_state.ndim != 1 or n_state_vector != 2 ** len(wires):
            raise ValueError("State vector must be of length 2**wires.")

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(itertools.product([0, 1], repeat=len(wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(wires), self.num_wires), dtype=int)
        unravelled_indices[:, wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)
        state = np.zeros_like(self._state)
        state[ravelled_indices] = input_state
        self._state = tf.convert_to_tensor(state, dtype=C_DTYPE)

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (list[int]): list of wires where the provided computational state should
                be initialized
        """
        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self.num_wires - 1 - np.array(wires))
        num = int(np.dot(state, basis_states))

        state = np.zeros_like(self._state)
        state[num] = 1.0
        self._state = tf.convert_to_tensor(state, dtype=C_DTYPE)

    def _apply_unitary(self, mat, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        This function uses einsum instead of tensordot. This approach is only
        faster for single- and two-qubit gates.

        Args:
            mat (tf.Tensor): matrix to multiply
            wires (Sequence[int]): target subsystems

        Returns:
            tf.Tensor: output vector after applying ``mat`` to input ``vec`` on specified subsystems
        """
        mat = tf.cast(tf.reshape(mat, [2] * len(wires) * 2), dtype=C_DTYPE)

        # Tensor indices of the quantum state
        state_indices = ABC[: self.num_wires]

        # Indices of the quantum state affected by this operation
        affected_indices = "".join(ABC_ARRAY[wires].tolist())

        # All affected indices will be summed over, so we need the same number of new indices
        new_indices = ABC[self.num_wires : self.num_wires + len(wires)]

        # The new indices of the state are given by the old ones with the affected indices
        # replaced by the new_indices
        new_state_indices = functools.reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
            zip(affected_indices, new_indices),
            state_indices,
        )

        # We now put together the indices in the notation numpy's einsum requires
        einsum_indices = "{new_indices}{affected_indices},{state_indices}->{new_state_indices}".format(
            affected_indices=affected_indices,
            state_indices=state_indices,
            new_indices=new_indices,
            new_state_indices=new_state_indices,
        )

        return tf.einsum(einsum_indices, mat, self._state)

    def _apply_diagonal_unitary(self, phases, wires):
        r"""Apply multiplication of a phase vector to subsystems of the quantum state.

        This represents the multiplication with diagonal gates in a more efficient manner.

        Args:
            phases (tf.Tensor): vector to multiply
            wires (Sequence[int]): target subsystems

        Returns:
            tf.Tensor: output vector after applying ``phases`` to input ``vec`` on specified subsystems
        """
        # reshape vectors
        phases = tf.reshape(phases, [2] * len(wires))

        state_indices = ABC[: self.num_wires]
        affected_indices = "".join(ABC_ARRAY[wires].tolist())

        einsum_indices = "{affected_indices},{state_indices}->{state_indices}".format(
            affected_indices=affected_indices, state_indices=state_indices
        )

        return tf.einsum(einsum_indices, phases, self._state)

    @property
    def state(self):
        return tf.reshape(self._pre_rotated_state, [-1])

    def reset(self):
        """Reset the device"""
        # init the state vector to |00..0>
        super().reset()

        state = np.zeros([np.prod(self._state.shape)], dtype=np.complex128)
        state[0] = 1
        state = state.reshape(self._state.shape)
        self._state = tf.convert_to_tensor(state, dtype=C_DTYPE)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None

        wires = wires or range(self.num_wires)

        prob = self.marginal_prob(tf.abs(self._state) ** 2, wires)
        return prob

    def marginal_prob(self, prob, wires=None):
        if wires is None:
            # no need to marginalize
            return prob

        wires = np.hstack(wires)

        # determine which wires are to be summed over
        inactive_wires = list(set(range(self.num_wires)) - set(wires))

        # sum over all inactive wires
        prob = tf.reshape(tf.reduce_sum(prob, axis=inactive_wires), [-1])

        # The wires provided might not be in consecutive order (i.e., wires might be [2, 0]).
        # If this is the case, we must permute the marginalized probability so that
        # it corresponds to the orders of the wires passed.
        basis_states = np.array(list(itertools.product([0, 1], repeat=len(wires))))
        perm = np.ravel_multi_index(
            basis_states[:, np.argsort(np.argsort(wires))].T, [2] * len(wires)
        )

        return tf.gather(prob, perm.tolist())

    def expval(self, observable):
        wires = observable.wires

        if self.analytic:
            # exact expectation value
            eigvals = tf.convert_to_tensor(observable.eigvals, dtype=R_DTYPE)
            prob = self.probability(wires=wires)
            return tf.tensordot(eigvals, prob, axes=1)

        return super().expval(observable)

    def var(self, observable):
        wires = observable.wires

        if self.analytic:
            # exact variance value
            eigvals = tf.convert_to_tensor(observable.eigvals, dtype=R_DTYPE)
            prob = self.probability(wires=wires)
            return (
                tf.tensordot(eigvals ** 2, prob, axes=1) - tf.tensordot(eigvals, prob, axes=1) ** 2
            )

        # estimate the variance
        return super().var(observable)

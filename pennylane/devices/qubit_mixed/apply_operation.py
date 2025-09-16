# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to apply operations to a qubit mixed state."""
# pylint: disable=unused-argument

from functools import singledispatch
from string import ascii_letters as alphabet

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.devices.qubit.apply_operation import _apply_grover_without_matrix
from pennylane.operation import Channel, Operator
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane.typing import TensorLike

from .einsum_manpulation import get_einsum_mapping

alphabet_array = math.array(list(alphabet))

TENSORDOT_STATE_NDIM_PERF_THRESHOLD = 9


def _get_slice(index, axis, num_axes):
    """Allows slicing along an arbitrary axis of an array or tensor.

    Args:
        index (int): the index to access
        axis (int): the axis to slice into
        num_axes (int): total number of axes

    Returns:
        tuple[slice or int]: a tuple that can be used to slice into an array or tensor

    **Example:**

    Accessing the 2 index along axis 1 of a 3-axis array:

    >>> sl = _get_slice(2, 1, 3)
    >>> sl
    (slice(None, None, None), 2, slice(None, None, None))
    >>> a = np.arange(27).reshape((3, 3, 3))
    >>> a[sl]
    array([[ 6,  7,  8],
           [15, 16, 17],
           [24, 25, 26]])
    """
    idx = [slice(None)] * num_axes
    idx[axis] = index
    return tuple(idx)


def _phase_shift(state, axis, phase_factor=-1, debugger=None, **_):
    """
    Applies a phase shift operation to a density matrix along a specified axis.

    This function implements a phase shift operation on a mixed quantum state (density matrix).
    For a given axis, it applies the phase shift by conjugating the density matrix with the
    phase shift operator: ρ -> U ρ U†, where U is the phase shift operator. This implementation
    is specific to single-qubit operations without broadcasting.

    Args:
        state (array-like): The density matrix to transform, with shape (2^n, 2^n) where n is
            the number of qubits.
        axis (int): The target qubit axis (0-based indexing) where the phase shift is applied.
        phase_factor (complex, optional): The complex phase to apply. Common values include:
            * -1 for Pauli-Z gate
            * 1j for S gate (π/2 phase)
            * exp(1j * π/4) for T gate (π/4 phase)
        debugger (callable, optional): A debug function for operation verification.
            Defaults to None.
        **_: Additional unused keyword arguments.

    Returns:
        array-like: The transformed density matrix with the same shape as the input.

    Raises:
        ValueError: If axis is invalid for the given density matrix dimension.
        ValueError: If the input state is not a valid density matrix (not square or
            incorrect dimensions).

    Example:
        >>> import numpy as np
        >>> # Single-qubit case: density matrix for |+⟩⟨+|
        >>> plus_state = np.array([[0.5, 0.5],
        ...                       [0.5, 0.5]])
        >>> # Apply Pauli-Z (phase_factor=-1)
        >>> z_applied = _phase_shift(plus_state, axis=0)
        >>> print(z_applied)
        [[0.5, -0.5],
         [-0.5, 0.5]]

        >>> # Two-qubit case: density matrix for |0⟩⟨0| ⊗ |+⟩⟨+|
        >>> two_qubit_state = np.array([
        ...     [0.5, 0.5, 0, 0],
        ...     [0.5, 0.5, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 0, 0, 0]
        ... ]).reshape(2,2,2,2)
        >>> # Apply phase shift on second qubit (axis=1)
        >>> z_on_second = _phase_shift(two_qubit_state, axis=1)
        >>> print(z_on_second)
        ... [[[[ 0.5  0.5]
        ...    [ 0.   0. ]]

        ...   [[-0.5 -0.5]
        ...    [-0.  -0. ]]]


        ...  [[[ 0.   0. ]
        ...    [ 0.   0. ]]

        ...   [[-0.  -0. ]
        ...    [-0.  -0. ]]]]

        >>> # Apply phase shift on first qubit (axis=1)
        >>> z_on_first = _phase_shift(two_qubit_state, axis=0)
        >>> print(z_on_first)
        ... [[[[ 0.5  0.5]
        ...    [ 0.   0. ]]

        ...   [[ 0.5  0.5]
        ...    [ 0.   0. ]]]


        ...  [[[-0.  -0. ]
        ...    [-0.  -0. ]]

        ...   [[-0.  -0. ]
        ...    [-0.  -0. ]]]]

    Notes:
        - The operation is performed in-place for computational efficiency
        - The function assumes the density matrix is in the computational basis
        - For an n-qubit system, the axis should be in range [0, n-1]
        - The phase shift operator U for single-qubit case is:
          U = [[1, 0],
               [0, phase_factor]]

    """
    n_dim = math.ndim(state)
    sl_0 = _get_slice(0, axis, n_dim)
    sl_1 = _get_slice(1, axis, n_dim)
    state_1 = math.multiply(state[sl_1], phase_factor)
    return math.stack([state[sl_0], state_1], axis=axis)


def _get_dagger_symmetric_real_op(op, num_wires):
    """Get the conjugate transpose of an operation by shifting num_wires. Should only be used for real, symmetric operations."""
    return qml.map_wires(op, {w: w + num_wires for w in op.wires})


def _get_num_wires(state, is_state_batched):
    """
    For density matrix, we need to infer the number of wires from the state.
    """

    shape = qml.math.shape(state)
    batch_size = shape[0] if is_state_batched else 1
    total_dim = math.prod(shape) // batch_size

    # total_dim should be 2^(2*num_wires)
    # Solve for num_wires: 2*num_wires = log2(total_dim) -> num_wires = log2(total_dim)/2
    num_wires = int(math.log2(total_dim) / 2)

    return num_wires


def _conjugate_state_with(k, state, axes_left, axes_right):
    """Perform the double tensor product k @ state @ k.conj(), with given, single matrix k.
    The `axes_left` and `axes_right` arguments are taken from the ambient variable space
    and `axes_right` is assumed to incorporate the tensor product and the transposition
    of k.conj() simultaneously."""
    return math.tensordot(
        math.tensordot(k, state, axes_left),
        math.conj(k),
        axes_right,
    )


def apply_operation_einsum(
    op: qml.operation.Operator,
    state,
    is_state_batched: bool = False,
    debugger=None,
    **_,
):
    r"""Apply a quantum channel specified by a list of Kraus operators to subsystems of the
    quantum state. For a unitary gate, there is a single Kraus operator.

    Args:
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        array[complex]: output_state
    """

    num_ch_wires = len(op.wires)

    if isinstance(op, Channel):
        kraus = op.kraus_matrices()
    else:
        kraus = [math.cast_like(op.matrix(), state)]

    # Shape kraus operators
    kraus_shape = [len(kraus)] + [2] * num_ch_wires * 2
    if not isinstance(op, Channel):
        mat = op.matrix() + 0j
        dim = 2**num_ch_wires
        batch_size = math.get_batch_size(mat, (dim, dim), dim**2)
        if batch_size is not None:
            # Add broadcasting dimension to shape
            kraus_shape = [batch_size] + kraus_shape

    kraus = math.stack(kraus)
    kraus_transpose = math.stack(math.moveaxis(kraus, source=-1, destination=-2))
    # Torch throws error if math.conj is used before stack
    kraus_dagger = math.conj(kraus_transpose)

    kraus = math.cast(math.reshape(kraus, kraus_shape), complex)
    kraus_dagger = math.reshape(kraus_dagger, kraus_shape)

    #! Check the def of helper func for details
    einsum_indices = get_einsum_mapping(op, state, is_state_batched)

    # Cast back to the same as state
    return math.einsum(einsum_indices, kraus, state, kraus_dagger)


def apply_operation_tensordot(
    op: qml.operation.Operator, state, is_state_batched: bool = False, debugger=None, **_
):
    """Apply ``Operator`` to ``state`` using ``math.tensordot``. This is more efficent at higher qubit
    numbers.

    Args:
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        array[complex]: output_state
    """
    channel_wires = op.wires
    num_ch_wires = len(channel_wires)

    num_wires = _get_num_wires(state, is_state_batched)
    #! Note that here we do not take into consideration the len of kraus list
    kraus_shape = [2] * num_ch_wires * 2
    # This could be pulled into separate function if tensordot is added
    if isinstance(op, Channel):
        kraus = [math.cast_like(math.reshape(k, kraus_shape), state) for k in op.kraus_matrices()]
    else:
        # !Note: we don't treat the batched ops inside tensordot calling
        # here's for the unified treatment of the ops in the tensordot calling
        # i.e. treating the op as a kraus list len 1
        mat = op.matrix() + 0j
        kraus = [mat]
    kraus = [math.reshape(k, kraus_shape) for k in kraus]
    kraus = math.array(kraus)  # Necessary for Jax
    # Small trick: _apply_channel_tensordot, here for the contraction on the right side we
    # also directly contract the column indices of the channel instead of rows
    # for simplicity. This can also save a step when transposing the Kraus operators.
    row_wires_list = [w + is_state_batched for w in channel_wires.tolist()]
    col_wires_list = [w + num_wires for w in row_wires_list]
    channel_col_ids = list(range(-num_ch_wires, 0))
    new_channel_col_ids = [-num_wires + w for w in channel_wires]
    axes_left = [channel_col_ids, row_wires_list]
    axes_right = [[0] + new_channel_col_ids, [0] + channel_col_ids]

    _state = _conjugate_state_with(kraus, state, axes_left, axes_right)
    source_left = list(range(num_ch_wires))
    dest_left = row_wires_list
    source_right = list(range(-num_ch_wires, 0))
    dest_right = col_wires_list

    result = math.moveaxis(_state, source_left + source_right, dest_left + dest_right)

    return result


@singledispatch
def apply_operation(
    op: qml.operation.Operator,
    state,
    is_state_batched: bool = False,
    debugger=None,
    **_,
):
    """Apply an operation to a given state.

    Args:
        op (Operator): The operation to apply to ``state``
        state (TensorLike): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not.
        debugger (_Debugger): The debugger to use.

    Keyword Arguments:
        rng (Optional[numpy.random._generator.Generator]): A NumPy random number generator.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``.
            This is the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, a ``numpy.random.default_rng`` will be used for sampling.
        tape_shots (Shots): The shots object of the tape.

    Returns:
        ndarray: The output state.

    .. warning::

        ``apply_operation`` is an internal function, and thus subject to change without a deprecation cycle.

    .. warning::

        ``apply_operation`` applies no validation to its inputs.

        This function assumes that the wires of the operator correspond to indices
        of the state. See :func:`~.map_wires` to convert operations to integer wire labels.

        The shape of the state should be ``[2] * (num_wires * 2)`` (the original tensor form) or
        ``[2**num_wires, 2**num_wires]`` (the expanded matrix form), where ``2`` is
        the dimension of the system.

    This is a ``functools.singledispatch`` function, so additional specialized kernels
    for specific operations can be registered like:

    .. code-block:: python

        @apply_operation.register
        def _(op: type_op, state, is_state_batched=False, **kwargs):
            # custom op application method here

    **Example:**

    >>> state = np.zeros((2, 2, 2, 2))
    >>> state[0][0][0][0] = 1
    >>> state
    array([[[[1., 0.],
             [0., 0.]],
            [[0., 0.],
             [0., 0.]]],
           [[[0., 0.],
             [0., 0.]],
            [[0., 0.],
             [0., 0.]]]])
    >>> apply_operation(qml.PauliX(0), state)
    array([[[[0., 0.],
             [0., 0.]],
            [[0., 0.],
             [0., 0.]]],
           [[[0., 0.],
             [1., 0.]],
            [[0., 0.],
             [0., 0.]]]])

    """

    return _apply_operation_default(op, state, is_state_batched, debugger, **_)


def _apply_operation_default(op, state, is_state_batched, debugger, **_):
    """The default behaviour of apply_operation, accessed through the standard dispatch
    of apply_operation, as well as conditionally in other dispatches.
    """
    if op in diagonal_in_z_basis:
        return apply_diagonal_unitary(op, state, is_state_batched, **_)
    num_op_wires = len(op.wires)
    interface = math.get_interface(state)

    # Add another layer of condition to rule out batched op (not channel) for tensordot calling
    if (op.batch_size is None) and (
        (num_op_wires > 2 and interface in {"autograd", "numpy"}) or num_op_wires > 7
    ):
        return apply_operation_tensordot(op, state, is_state_batched, debugger, **_)
    return apply_operation_einsum(op, state, is_state_batched, debugger, **_)


@apply_operation.register
def apply_identity(op: qml.Identity, state, is_state_batched: bool = False, debugger=None, **_):
    """Applies a :class:`~.Identity` operation by just returning the input state."""
    return state


@apply_operation.register
def apply_global_phase(
    op: qml.GlobalPhase, state, is_state_batched: bool = False, debugger=None, **_
):
    """Applies a :class:`~.GlobalPhase` operation by multiplying the state by ``exp(1j * op.data[0])``"""
    # Note: the global phase is a scalar, so we can just multiply the
    # state by it. For density matrix we suppose that the global phase
    # means a phase factor acting on the basis state vectors, which
    # implies that in the final density matrix there will be no effect.
    return state


@apply_operation.register
def apply_paulix(op: qml.X, state, is_state_batched: bool = False, debugger=None, **_):
    """Applies a :class:`~.PauliX` operation by multiplying the state by the Pauli-X matrix."""
    # PauliX is basically a bit flip, so we can just apply the X gate to the state
    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)
    axis_left = op.wires[0] + is_state_batched
    axis_right = axis_left + num_wires

    return math.roll(math.roll(state, 1, axis_left), 1, axis_right)


@apply_operation.register
def apply_pauliz(op: qml.Z, state, is_state_batched: bool = False, debugger=None, **_):
    """Applies a :class:`~.PauliZ` operation by multiplying the state by the Pauli-Z matrix."""
    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)
    n_dim = math.ndim(state)

    if (
        n_dim >= TENSORDOT_STATE_NDIM_PERF_THRESHOLD and math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)

    # First, flip the left side
    axis = op.wires[0] + is_state_batched
    state = _phase_shift(state, axis)

    # Second, flip the right side
    axis = op.wires[0] + is_state_batched + num_wires
    state = _phase_shift(state, axis)

    return state


@apply_operation.register
def apply_T(op: qml.T, state, is_state_batched: bool = False, debugger=None, **_):
    """Applies a :class:`~.T` operation by multiplying the state by the T matrix."""
    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)
    n_dim = math.ndim(state)

    if (
        n_dim >= TENSORDOT_STATE_NDIM_PERF_THRESHOLD and math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)

    # First, flip the left side
    axis = op.wires[0] + is_state_batched
    state = _phase_shift(state, axis, phase_factor=math.exp(0.25j * np.pi))

    # Second, flip the right side
    axis = op.wires[0] + is_state_batched + num_wires
    state = _phase_shift(state, axis, phase_factor=math.exp(-0.25j * np.pi))

    return state


@apply_operation.register
def apply_S(op: qml.S, state, is_state_batched: bool = False, debugger=None, **_):
    """Applies a :class:`~.S` operation by multiplying the state by the S matrix."""
    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)
    n_dim = math.ndim(state)

    if (
        n_dim >= TENSORDOT_STATE_NDIM_PERF_THRESHOLD and math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)

    # First, flip the left side
    axis = op.wires[0] + is_state_batched
    state = _phase_shift(state, axis, phase_factor=1.0j)

    # Second, flip the right side
    axis = op.wires[0] + is_state_batched + num_wires
    state = _phase_shift(state, axis, phase_factor=-1.0j)

    return state


@apply_operation.register
def apply_phaseshift(op: qml.PhaseShift, state, is_state_batched: bool = False, debugger=None, **_):
    """Applies a :class:`~.Phaseshift` operation by multiplying the state by the Phaseshift matrix."""
    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)
    n_dim = math.ndim(state)

    if (
        n_dim >= TENSORDOT_STATE_NDIM_PERF_THRESHOLD and math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)

    # Common constants always needed
    n_dim = math.ndim(state)
    num_wires = _get_num_wires(state, is_state_batched)

    # Start applying from the left side
    axis = op.wires[0] + is_state_batched

    # Slice indices of the affected axis
    sl_0 = _get_slice(0, axis, n_dim)
    sl_1 = _get_slice(1, axis, n_dim)

    # Get the phase shift parameter
    params = math.cast(op.parameters[0], dtype=complex)
    state0 = state[sl_0]
    state1 = state[sl_1]
    if op.batch_size is not None and len(params) > 1:
        interface = math.get_interface(state)
        if interface == "torch":
            params = math.array(params, like=interface)
        if is_state_batched:
            # If both op and state are batched, they have to have the same batch size
            params = math.reshape(params, (-1,) + (1,) * (n_dim - 2))
        else:
            # Op is batched, state is not, so we need to expand the state to batched
            params = math.reshape(params, (-1,) + (1,) * (n_dim - 1))
            state0 = math.expand_dims(state0, 0) + math.zeros_like(params)
            state1 = math.expand_dims(state1, 0)
            # Update status
            is_state_batched = True
            axis = axis + 1
            n_dim = n_dim + 1
    state1 = math.multiply(math.cast(state1, dtype=complex), math.exp(1.0j * params))
    state = math.stack([state0, state1], axis=axis)
    # Left side finished

    # Now start right side
    axis += num_wires  # Move to the right side (conjugate side)
    # Slice indices of the affected axis
    sl_0 = _get_slice(0, axis, n_dim)
    sl_1 = _get_slice(1, axis, n_dim)
    # Get the phase shift parameter, conjugated
    state0 = state[sl_0]
    state1 = state[sl_1]
    # No need for expanding, since on the left side we already did

    state1 = math.multiply(math.cast(state1, dtype=complex), math.exp(-1.0j * params))
    state = math.stack([state0, state1], axis=axis)
    return state


# !TODO: in the future investigate if there's other missing operations
# satisfying this condition.
SYMMETRIC_REAL_OPS = (
    qml.CNOT,
    qml.MultiControlledX,
    qml.Toffoli,
    qml.SWAP,
    qml.CSWAP,
    qml.CZ,
    qml.CH,
)


def apply_symmetric_real_op(
    op: qml.CNOT | qml.MultiControlledX | qml.Toffoli | qml.SWAP | qml.CSWAP | qml.CZ | qml.CH,
    state,
    is_state_batched: bool = False,
    debugger=None,
    **_,
):
    r"""Apply real, symmetric operator (e.g. X, CX and related controlled-X variants) to a density matrix state.

    This function handles CZ, CH, CNOT, CSWAP, SWAP, Toffoli, and general MultiControlledX operations using the same underlying
    implementation, as they share the properties of being real and symmetric. For operations with 8 or fewer wires,
    it uses the default einsum contraction. For larger operations, it leverages a custom kernel that
    exploits the fact that for real, symmetric operators, the adjoint operation can be implemented
    by shifting wires by `num_wires`.

    Args:
        op (.Operation): CZ, CH, CNOT, CSWAP, SWAP, Toffoli, and general MultiControlledX operation
        state (tensor_like): The density matrix state to apply the operation to
        is_state_batched (bool): Whether the state has a batch dimension. Rather than checking
            matrix dimensions, we use op.batch_size for efficiency
        debugger (optional): A debugger instance for operation validation

    Returns:
        tensor_like: The transformed density matrix state

    Note:
        This is not a final version. Two possible improvements are:
        1. More existing real, symmetric ops to include in this dispatch
        2. A more general approach to handle other types of ops but following
        similar logic as in this function.
    """

    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)
    if len(op.wires) < TENSORDOT_STATE_NDIM_PERF_THRESHOLD:
        return _apply_operation_default(op, state, is_state_batched, debugger)

    state = qml.devices.qubit.apply_operation(op, state, is_state_batched, debugger)

    op_dagger = _get_dagger_symmetric_real_op(op, num_wires)
    state = qml.devices.qubit.apply_operation(op_dagger, state, is_state_batched, debugger)
    return state


# NOTE: this loop is for a nice doc rendering for `apply_operation`. With a direct multiple single
# registers over different op class there will be severe rendering issue as discussed in
# https://github.com/PennyLaneAI/pennylane/pull/6684#pullrequestreview-2565634328
for op_class in SYMMETRIC_REAL_OPS:
    apply_operation.register(op_class)(apply_symmetric_real_op)


@apply_operation.register
def apply_grover(
    op: qml.GroverOperator,
    state,
    is_state_batched: bool = False,
    debugger=None,
    **_,
):
    """Apply GroverOperator either via a custom matrix-free method (more than 8 operation
    wires) or via standard matrix based methods (else)."""
    if len(op.wires) < TENSORDOT_STATE_NDIM_PERF_THRESHOLD:
        return _apply_operation_default(op, state, is_state_batched, debugger)
    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)

    state = _apply_grover_without_matrix(state, op.wires, is_state_batched)
    state = _apply_grover_without_matrix(state, [w + num_wires for w in op.wires], is_state_batched)
    return state


def apply_diagonal_unitary(op: Operator, state: TensorLike, is_state_batched: bool = False, **_):
    """Apply a diagonal unitary operation to the density matrix state using its eigenvalues.

    Args:
        op (qml.Operation): The diagonal unitary operation to apply.
        state (TensorLike): The density matrix state to apply the operation to.
        is_state_batched (bool, optional): Whether the state has a batch dimension. Defaults to False.

    Returns:
        TensorLike: The transformed density matrix state.
    """
    channel_wires = op.wires
    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)

    eigvals = math.stack(op.eigvals())
    expand_shape = [-1, 2 * len(channel_wires)] if bool(op.batch_size) else [2] * len(channel_wires)
    eigvals = math.reshape(eigvals, expand_shape)
    eigvals = math.cast_like(eigvals, state)

    state_indices = alphabet[: 2 * num_wires + is_state_batched]

    row_wires_list = [w + is_state_batched for w in channel_wires.tolist()]
    col_wires_list = [w + num_wires for w in row_wires_list]

    row_indices = "".join(alphabet_array[row_wires_list].tolist())
    col_indices = "".join(alphabet_array[col_wires_list].tolist())

    # Use ellipsis to represent the batch dimensions as eigvals.shape = (batch, 2 * len(channel_wires))
    ellipsis = "..." if bool(op.batch_size) else ""
    einsum_indices = f"{ellipsis}{row_indices},{ellipsis}{state_indices},{ellipsis}{col_indices}->{ellipsis}{state_indices}"

    return math.einsum(einsum_indices, eigvals, state, math.conj(eigvals))


@apply_operation.register
def apply_snapshot(
    op: qml.Snapshot, state, is_state_batched: bool = False, debugger=None, **execution_kwargs
):
    """Take a snapshot of the mixed state

    Args:
        op (qml.Snapshot): the snapshot operation
        state (array): current quantum state
        is_state_batched (bool): whether the state is batched
        debugger: the debugger instance for storing snapshots
    Returns:
        array: the unchanged quantum state
    """
    if debugger and debugger.active:
        measurement = op.hyperparameters.get(
            "measurement", None
        )  # default: None, meaning no measurement, simply copy the state
        if op.hyperparameters["shots"] == "workflow":
            shots = execution_kwargs.get("tape_shots")
        else:
            shots = op.hyperparameters["shots"]

        if isinstance(measurement, qml.measurements.StateMP) or not shots:
            snapshot = qml.devices.qubit_mixed.measure(measurement, state, is_state_batched)
        else:
            snapshot = qml.devices.qubit_mixed.measure_with_samples(
                [measurement],
                state,
                shots,
                is_state_batched,
                execution_kwargs.get("rng"),
                execution_kwargs.get("prng_key"),
            )[0]

        # Store snapshot with optional tag
        if op.tag:
            debugger.snapshots[op.tag] = snapshot
        else:
            debugger.snapshots[len(debugger.snapshots)] = snapshot

    return state


@apply_operation.register
def apply_density_matrix(
    op: qml.QubitDensityMatrix,
    state,
    is_state_batched: bool = False,
    debugger=None,
    **execution_kwargs,
):
    """
    Applies a QubitDensityMatrix operation by initializing or replacing
    the quantum state with the provided density matrix.

    - If the QubitDensityMatrix covers all wires, we directly return the provided density matrix as the new state.
    - If only a subset of the wires is covered, we:
      1. Partial trace out those wires from the current state to get the density matrix of the complement wires.
      2. Take the tensor product of the complement density matrix and the provided density_matrix.
      3. Reshape to the correct final shape and return.

    Args:
        op (qml.QubitDensityMatrix): The QubitDensityMatrix operation.
        state (array-like): The current quantum state.
        is_state_batched (bool): Whether the state is batched.
        debugger: A debugger instance for diagnostics.
        **execution_kwargs: Additional kwargs.

    Returns:
        array-like: The updated quantum state.

    Raises:
        ValueError: If the density matrix is invalid.
    """
    density_matrix = op.parameters[0]
    num_wires = len(op.wires)
    expected_dim = 2**num_wires

    # Cast density_matrix to the same type and device as state
    density_matrix = math.cast_like(density_matrix, state)

    # Extract total wires
    num_state_wires = _get_num_wires(state, is_state_batched)
    all_wires = list(range(num_state_wires))
    op_wires = op.wires
    complement_wires = [w for w in all_wires if w not in op_wires]

    # If the operation covers the full system, just return it
    if len(op_wires) == num_state_wires:
        # If batched, broadcast
        if is_state_batched:
            batch_size = math.shape(state)[0]
            density_matrix = math.broadcast_to(
                density_matrix, (batch_size,) + math.shape(density_matrix)
            )

        # Reshape to match final shape of state
        return math.reshape(density_matrix, math.shape(state))

    # Partial system update:
    # 1. Partial trace out op_wires from state
    # partial_trace reduces the dimension to only the complement wires
    sigma = qml.math.partial_trace(state, indices=op_wires)
    # sigma now has shape:
    # (batch_size, 2^(n - num_wires), 2^(n - num_wires)) where n = total wires

    # 2. Take kron(sigma, density_matrix)
    sigma_dim = 2 ** len(complement_wires)  # dimension of complement subsystem
    dm_dim = expected_dim  # dimension of the replaced subsystem
    if is_state_batched:
        batch_size = math.shape(sigma)[0]
        sigma_2d = math.reshape(sigma, (batch_size, sigma_dim, sigma_dim))
        dm_2d = math.reshape(density_matrix, (dm_dim, dm_dim))

        # Initialize new_dm and fill via a loop or vectorized kron if available
        new_dm = []
        for b in range(batch_size):
            new_dm.append(math.kron(sigma_2d[b], dm_2d))
        rho = math.stack(new_dm, axis=0)
    else:
        sigma_2d = math.reshape(sigma, (sigma_dim, sigma_dim))
        dm_2d = math.reshape(density_matrix, (dm_dim, dm_dim))
        rho = math.kron(sigma_2d, dm_2d)

    # rho now has shape (batch_size?, 2^n, 2^n)

    # 3. Reshape rho into the full tensor form [2]*(2*n) or [batch_size, 2]*(2*n)
    final_shape = ([batch_size] if is_state_batched else []) + [2] * (2 * num_state_wires)
    rho = math.reshape(rho, final_shape)

    # Return the updated state
    return reorder_after_kron(rho, complement_wires, op_wires, is_state_batched)


def reorder_after_kron(rho, complement_wires, op_wires, is_state_batched):
    """
    Reorder the wires of `rho` from [complement_wires + op_wires] back to [0,1,...,N-1].

    Args:
        rho (tensor): The density matrix after kron(sigma, density_matrix).
        complement_wires (list[int]): The wires not affected by the QubitDensityMatrix update.
        op_wires (Wires): The wires affected by the QubitDensityMatrix.
        is_state_batched (bool): Whether the state is batched.

    Returns:
        tensor: The density matrix with wires in the original order.
    """
    # Final order after kron is complement_wires + op_wires (for both left and right sides).
    all_wires = complement_wires + list(op_wires)
    num_wires = len(all_wires)

    batch_offset = 1 if is_state_batched else 0

    # The current axis mapping is:
    # Left side wires: offset to offset+num_wires-1
    # Right side wires: offset+num_wires to offset+2*num_wires-1
    #
    # We want to reorder these so that the left side wires are [0,...,num_wires-1] and
    # the right side wires are [num_wires,...,2*num_wires-1].

    # Create a lookup from wire label to its position in the current order.
    wire_to_pos = {w: i for i, w in enumerate(all_wires)}

    # We'll construct a permutation of axes. `rho` has dimensions:
    # [batch?] + [2]*num_wires (left side) + [2]*num_wires (right side)
    #
    # After transpose, dimension i in the new tensor should correspond to dimension new_axes[i] in the old tensor.

    old_ndim = rho.ndim
    new_axes = [None] * old_ndim

    # If batched, batch dimension remains at axis 0
    if is_state_batched:
        new_axes[0] = 0

    # For the left wires:
    # Desired final order: 0,1,...,num_wires-1
    # Currently: all_wires in some order
    # old axis = batch_offset + wire_to_pos[w]
    # new axis = batch_offset + w
    for w in range(num_wires):
        old_axis = batch_offset + wire_to_pos[w]
        new_axes[batch_offset + w] = old_axis

    # For the right wires:
    # Desired final order: num_wires,...,2*num_wires-1
    # Currently: batch_offset+num_wires+wire_to_pos[w]
    # new axis: batch_offset+num_wires+w
    for w in range(num_wires):
        old_axis = batch_offset + num_wires + wire_to_pos[w]
        new_axes[batch_offset + num_wires + w] = old_axis

    # Apply the transpose
    rho = math.transpose(rho, axes=tuple(new_axes))
    return rho

import functools
from functools import singledispatch
import pennylane as qml
from pennylane import math
from string import ascii_letters as alphabet
from pennylane import numpy as np
from pennylane.operation import Channel

alphabet_array = np.array(list(alphabet))
qutrit_diagonal_in_z_basis = qml.Attribute(["TClock", "TRZ"])

EINSUM_OP_WIRECOUNT_PERF_THRESHOLD = 3
EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = 13

qudit_dim = 3  # specifies qudit dimension

def apply_operation_einsum(op: qml.operation.Operator, state, is_state_batched: bool = False):
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
    num_wires = int((qml.math.shape(state) - is_state_batched)/2)
    rho_dim = 2 * num_wires

    # Computes K^\dagger, needed for the transformation K \rho K^\dagger
    kraus_dagger = [math.conj(math.transpose(k)) for k in kraus]

    kraus = math.stack(kraus)
    kraus_dagger = math.stack(kraus_dagger)

    # Shape kraus operators
    kraus_shape = [len(kraus)] + [qudit_dim] * num_ch_wires * 2
    kraus = math.cast(math.reshape(kraus, kraus_shape), complex)
    kraus_dagger = math.cast(math.reshape(kraus_dagger, kraus_shape), complex)

    # Tensor indices of the state. For each qubit, need an index for rows *and* columns
    state_indices = alphabet[:rho_dim]

    # row indices of the quantum state affected by this operation
    row_wires_list = op.wires.tolist()
    row_indices = "".join(alphabet_array[row_wires_list].tolist())

    # column indices are shifted by the number of wires
    col_wires_list = [w + num_wires for w in row_wires_list]
    col_indices = "".join(alphabet_array[col_wires_list].tolist())

    # indices in einsum must be replaced with new ones
    new_row_indices = alphabet[rho_dim: rho_dim + num_ch_wires]
    new_col_indices = alphabet[rho_dim + num_ch_wires: rho_dim + 2 * num_ch_wires]

    # index for summation over Kraus operators
    kraus_index = alphabet[rho_dim + 2 * num_ch_wires: rho_dim + 2 * num_ch_wires + 1]

    # new state indices replace row and column indices with new ones
    new_state_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(col_indices + row_indices, new_col_indices + new_row_indices),
        state_indices,
    )

    # index mapping for einsum, e.g., 'iga,abcdef,idh->gbchef'
    einsum_indices = (
        f"{kraus_index}{new_row_indices}{row_indices}, {state_indices},"
        f"{kraus_index}{col_indices}{new_col_indices}->{new_state_indices}"
    )

    return math.einsum(einsum_indices, kraus, state, kraus_dagger)


def apply_operation_tensordot(op: qml.operation.Operator, state, is_state_batched: bool = False):
    r"""Apply a quantum channel specified by a list of Kraus operators to subsystems of the
    quantum state. For a unitary gate, there is a single Kraus operator.

    Args:
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        is_state_batched (bool): Boolean representing whether the state is batched or not # TODO

    Returns:
        array[complex]: output_state
    """
    num_ch_wires = len(op.wires)
    num_wires = (qml.math.shape(state) - is_state_batched)/2

    # Shape kraus operators and cast them to complex data type
    kraus_shape = [qudit_dim] * (num_ch_wires * 2)

    #TODO deal with batching
    dim = 2
    if is_mat_batched := batch_size is not None:
        # Add broadcasting dimension to shape
        new_mat_shape = [batch_size] + new_mat_shape
        if op.batch_size is None:
            op._batch_size = batch_size  # pylint:disable=protected-access

    kraus = [math.cast(math.reshape(k, kraus_shape), complex) for k in kraus]

    # row indices of the quantum state affected by this operation
    row_wires_list = op.wires.tolist()
    # column indices are shifted by the number of wires
    col_wires_list = [w + num_wires for w in row_wires_list]

    channel_col_ids = list(range(num_ch_wires, 2 * num_ch_wires))
    axes_left = [channel_col_ids, row_wires_list]
    # Use column indices instead or rows to incorporate transposition of K^\dagger
    axes_right = [col_wires_list, channel_col_ids]

    # Apply the Kraus operators, and sum over all Kraus operators afterwards
    def _conjugate_state_with(k):
        """Perform the double tensor product k @ self._state @ k.conj().
        The `axes_left` and `axes_right` arguments are taken from the ambient variable space
        and `axes_right` is assumed to incorporate the tensor product and the transposition
        of k.conj() simultaneously."""
        return math.tensordot(math.tensordot(k, state, axes_left), math.conj(k), axes_right)

    if len(kraus) == 1:
        _state = _conjugate_state_with(kraus[0])
    else:
        _state = math.sum(math.stack([_conjugate_state_with(k) for k in kraus]), axis=0)

    # Permute the affected axes to their destination places.
    # The row indices of the kraus operators are moved from the beginning to the original
    # target row locations, the column indices from the end to the target column locations
    source_left = list(range(num_ch_wires))
    dest_left = row_wires_list
    source_right = list(range(-num_ch_wires, 0))
    dest_right = col_wires_list
    return math.moveaxis(_state, source_left + source_right, dest_left + dest_right)


@singledispatch
def apply_operation(
        op: qml.operation.Operator, state, is_state_batched: bool = False, debugger=None
):
    """Apply and operator to a given state.

    Args:
        op (Operator): The operation to apply to ``state``
        state (TensorLike): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not
        debugger (_Debugger): The debugger to use

    Returns:
        ndarray: output state

    .. warning::

        ``apply_operation`` is an internal function, and thus subject to change without a deprecation cycle.

    .. warning::
        ``apply_operation`` applies no validation to its inputs.

        This function assumes that the wires of the operator correspond to indices
        of the state. See :func:`~.map_wires` to convert operations to integer wire labels.

        The shape of state should be ``[qudit_dim]*(num_wires * 2)``.

    This is a ``functools.singledispatch`` function, so additional specialized kernels
    for specific operations can be registered like:

    .. code-block:: python

        @apply_operation.register
        def _(op: type_op, state):
            # custom op application method here

    **Example:**

    >>> state = np.zeros((3,3))
    >>> state[0][0] = 1
    >>> state
    tensor([[1., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]], requires_grad=True)
    >>> apply_operation(qml.TShift(0), state)
    tensor([[0., 0., 0.],
        [0., 1., 0],
        [0., 0., 0.],], requires_grad=True)

    """
    return _apply_operation_default(op, state, is_state_batched, debugger)


def _apply_operation_default(op, state, is_state_batched, debugger):
    """The default behaviour of apply_operation, accessed through the standard dispatch
    of apply_operation, as well as conditionally in other dispatches.
    """
    if op in qutrit_diagonal_in_z_basis:
        return _apply_diagonal_unitary(op, state, is_state_batched)
    if (
        len(op.wires) < EINSUM_OP_WIRECOUNT_PERF_THRESHOLD
        and math.ndim(state) < EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD
    ) or (op.batch_size and is_state_batched):
        return apply_operation_einsum(op, state, is_state_batched=is_state_batched)
    return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)


def _apply_diagonal_unitary(op: qml.operation.Operator, state, is_state_batched):
    r"""Apply a diagonal unitary gate specified by a list of eigenvalues. This method uses
    the fact that the unitary is diagonal for a more efficient implementation.

    Args:
        op (Operator): The operation to apply to ``state``
        state (TensorLike): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        ndarray: output state
    """

    num_wires = (qml.math.shape(state) - is_state_batched)/2

    eigvals = math.stack(eigvals)

    # reshape vectors
    eigvals = math.cast(math.reshape(eigvals, [qudit_dim] * len(wires)), complex)

    # Tensor indices of the state. For each qubit, need an index for rows *and* columns
    state_indices = alphabet[: 2 * num_wires]

    # row indices of the quantum state affected by this operation
    row_indices = "".join(alphabet_array[op.wires].tolist())

    # column indices are shifted by the number of wires
    col_wires_list = [w + num_wires for w in op.wires]
    col_indices = "".join(alphabet_array[col_wires_list].tolist())

    einsum_indices = f"{row_indices},{state_indices},{col_indices}->{state_indices}"

    return math.einsum(einsum_indices, eigvals, state, math.conj(eigvals))


@apply_operation.register
def apply_identity(op: qml.Identity, state, is_state_batched: bool = False, debugger=None):
    """Applies a :class:`~.Identity` operation by just returning the input state."""
    return state


@apply_operation.register
def apply_snapshot(op: qml.Snapshot, state, is_state_batched: bool = False, debugger=None):
    if debugger and debugger.active:
        dim = int(np.sqrt(state.size))
        density_matrix = math.reshape(state, (dim, dim))
        if op.tag:
            debugger.snapshots[op.tag] = density_matrix
        else:
            debugger.snapshots[len(debugger.snapshots)] = density_matrix
    return state


@apply_operation.register
def apply_(op: qml.TAdd, state, is_state_batched: bool = False, debugger=None):
    """Apply TAdd gate to state."""
    pass


def _get_kraus(operation):  # pylint: disable=no-self-use
    """Return the Kraus operators representing the operation.

    Args:
        operation (.Operation): a PennyLane operation

    Returns:
        list[array[complex]]: Returns a list of 2D matrices representing the Kraus operators. If
        the operation is unitary, returns a single Kraus operator. In the case of a diagonal
        unitary, returns a 1D array representing the matrix diagonal.
    """
    if operation in qutrit_diagonal_in_z_basis:
        return operation.eigvals()

    if isinstance(operation, Channel):
        return operation.kraus_matrices()
    return [operation.matrix()]



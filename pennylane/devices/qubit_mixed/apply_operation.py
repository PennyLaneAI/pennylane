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

import warnings
from functools import singledispatch
from string import ascii_letters as alphabet

import pennylane as qml
from pennylane import math
from pennylane import numpy as np
from pennylane.operation import Channel
from pennylane.ops.qubit.attributes import diagonal_in_z_basis

from .constants import QUDIT_DIM
from .utils import get_einsum_mapping, get_new_state_einsum_indices

alphabet_array = np.array(list(alphabet))


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


def _phase_shift(state, axis, phase_factor=-1):
    """
    Applies a phase shift to a quantum state along a specified axis.

    This function takes a quantum state and applies a phase shift along a given axis.
    The phase shift operation multiplies one part of the state by a complex phase factor.
    This can represent various quantum gates including Pauli-Z, S, T, and other phase gates.

    Args:
        state (array-like): The quantum state to which the phase shift will be applied. Can be a vector or a multi-dimensional array representing a quantum state.
        axis (int): The axis along which to perform the phase shift operation.
        phase_factor (complex, optional): The complex factor to multiply the affected part of the state by. Defaults to -1 (which represents a Pauli-Z operation).

    Returns:
        array-like: The phase-shifted quantum state, with the same shape as the input state.

    Raises:
        ValueError: If the axis is out of bounds for the given state.

    Note:
        This function assumes the use of a math library (like numpy or jax.numpy)
        for array operations. The specific library should be imported as 'math'
        before using this function.

    Example:
        >>> import numpy as np
        >>> state = np.array([1, 1]) / np.sqrt(2)  # |+⟩ state
        >>> z_applied_state = _phase_shift(state, axis=0)  # Applying Pauli-Z
        >>> print(z_applied_state)
        [0.70710678, -0.70710678]  # Approximately [1/√2, -1/√2]

        # For S gate (π/2 phase shift)
        >>> s_applied_state = _phase_shift(state, axis=0, phase_factor=1j)
        >>> print(s_applied_state)
        [0.70710678+0.j, 0.+0.70710678j]  # Approximately [1/√2, i/√2]

        # For T gate (π/4 phase shift)
        >>> t_applied_state = _phase_shift(state, axis=0, phase_factor=np.exp(1j * np.pi/4))
        >>> print(t_applied_state)
        [0.70710678+0.j, 0.5+0.5j]  # Approximately [1/√2, (1+i)/2]
    """
    n_dim = math.ndim(state)
    sl_0 = _get_slice(0, axis, n_dim)
    sl_1 = _get_slice(1, axis, n_dim)
    state_1 = math.multiply(state[sl_1], phase_factor)
    return math.stack([state[sl_0], state_1], axis=axis)


def _map_indices_apply_channel(**kwargs):
    """Map indices to einsum string
    Args:
        **kwargs (dict): Stores indices calculated in `get_einsum_mapping`

    Returns:
        String of einsum indices to complete einsum calculations
    """
    op_1_indices = f"{kwargs['kraus_index']}{kwargs['new_row_indices']}{kwargs['row_indices']}"
    op_2_indices = f"{kwargs['kraus_index']}{kwargs['col_indices']}{kwargs['new_col_indices']}"

    new_state_indices = get_new_state_einsum_indices(
        old_indices=kwargs["col_indices"] + kwargs["row_indices"],
        new_indices=kwargs["new_col_indices"] + kwargs["new_row_indices"],
        state_indices=kwargs["state_indices"],
    )
    # index mapping for einsum, e.g., '...iga,...abcdef,...idh->...gbchef'
    return (
        f"...{op_1_indices},...{kwargs['state_indices']},...{op_2_indices}->...{new_state_indices}"
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

    #! Note that there the state should be a density matrix
    einsum_indices = get_einsum_mapping(op, state, _map_indices_apply_channel, is_state_batched)

    num_ch_wires = len(op.wires)

    # This could be pulled into separate function if tensordot is added
    if isinstance(op, Channel):
        kraus = op.kraus_matrices()
    else:
        kraus = [math.cast_like(op.matrix(), state)]

    # Shape kraus operators
    kraus_shape = [len(kraus)] + [QUDIT_DIM] * num_ch_wires * 2
    if not isinstance(op, Channel):
        mat = op.matrix() + 0j
        dim = QUDIT_DIM**num_ch_wires
        batch_size = math.get_batch_size(mat, (dim, dim), dim**2)
        if batch_size is not None:
            # Add broadcasting dimension to shape
            kraus_shape = [batch_size] + kraus_shape
            if op.batch_size is None:
                op._batch_size = batch_size  # pylint:disable=protected-access

    kraus = math.stack(kraus)
    kraus_transpose = math.stack(math.moveaxis(kraus, source=-1, destination=-2))
    # Torch throws error if math.conj is used before stack
    kraus_dagger = math.conj(kraus_transpose)

    kraus = math.cast(math.reshape(kraus, kraus_shape), complex)
    kraus_dagger = math.cast(math.reshape(kraus_dagger, kraus_shape), complex)

    res = math.einsum(einsum_indices, kraus, state, kraus_dagger)
    # Cast back to the same as state
    return math.cast_like(res, state)


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
    # We use this implicit casting strategy as autograd raises ComplexWarnings
    # when backpropagating if casting explicitly. Some type of casting is needed
    # to prevent ComplexWarnings with backpropagation with other interfaces

    channel_wires = op.wires
    num_ch_wires = len(channel_wires)

    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)

    #! Note that here we do not take into consideration the len of kraus list
    kraus_shape = [QUDIT_DIM] * num_ch_wires * 2
    # This could be pulled into separate function if tensordot is added
    if isinstance(op, Channel):

        kraus = [math.cast_like(math.reshape(k, kraus_shape), state) for k in op.kraus_matrices()]
    else:
        kraus = [math.cast_like(op.matrix(), state)]

    # Small trick: following the same logic as in the legacy DefaultMixed._apply_channel_tensordot, here for the contraction on the right side we also directly contract the col ids of channel instead of rows for simplicity. This can also save a step of transposing the kraus operators.
    row_wires_list = channel_wires.tolist()  # Example: H0 => [0]
    col_wires_list = [w + num_wires for w in row_wires_list]  # Example: H0 => [3]
    channel_col_ids = list(range(num_ch_wires, 2 * num_ch_wires))
    axes_left = [channel_col_ids, row_wires_list]
    axes_right = [col_wires_list, channel_col_ids]

    # Apply the Kraus operators, and sum over all Kraus operators afterwards
    def _conjugate_state_with(k):
        """Perform the double tensor product k @ self._state @ k.conj().
        The `axes_left` and `axes_right` arguments are taken from the ambient variable space
        and `axes_right` is assumed to incorporate the tensor product and the transposition
        of k.conj() simultaneously."""
        return math.tensordot(
            math.tensordot(k, state, axes_left),
            math.conj(k),
            axes_right,
        )

    if len(kraus) == 1:
        _state = _conjugate_state_with(kraus[0])

    else:
        _state = math.sum(math.stack([_conjugate_state_with(k) for k in kraus]), axis=0)

    source_left = list(range(num_ch_wires))
    dest_left = row_wires_list
    source_right = list(range(-num_ch_wires, 0))
    dest_right = col_wires_list
    result = math.moveaxis(_state, source_left + source_right, dest_left + dest_right)

    return math.cast_like(result, state)


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
        is_state_batched (bool): Boolean representing whether the state is batched or not
        debugger (_Debugger): The debugger to use

    Keyword Arguments:
        rng (Optional[numpy.random._generator.Generator]): A NumPy random number generator.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, a ``numpy.random.default_rng`` will be used for sampling.
        tape_shots (Shots): the shots object of the tape

    Returns:
        ndarray: output state

    .. warning::

        ``apply_operation`` is an internal function, and thus subject to change without a deprecation cycle.

    .. warning::
        ``apply_operation`` applies no validation to its inputs.

        This function assumes that the wires of the operator correspond to indices
        of the state. See :func:`~.map_wires` to convert operations to integer wire labels.

        The shape of state should be ``[QUDIT_DIM]*(num_wires * 2)``, where ``QUDIT_DIM`` is
        the dimension of the system.

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
    return _apply_operation_default(op, state, is_state_batched, debugger, **_)


def _apply_operation_default(op, state, is_state_batched, debugger, **_):
    """The default behaviour of apply_operation, accessed through the standard dispatch
    of apply_operation, as well as conditionally in other dispatches.
    """
    if op in diagonal_in_z_basis:
        return apply_diagonal_unitary(op, state, is_state_batched, debugger, **_)
    num_op_wires = len(op.wires)
    interface = math.get_interface(state)
    if (num_op_wires > 2 and interface in {"autograd", "numpy"}) or num_op_wires > 7:
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
    # Note: the global phase is a scalar, so we can just multiply the state by it. For density matrix we suppose that the global phase means a phase factor acting on the basis statevectors, which implies that in the final density matrix there will be no effect. Therefore, we would like to warn users that even though an identity operation is applied, the global phase operation will not have any effect on the density matrix.
    warnings.warn(
        "The GlobalPhase operation does not have any effect on the density matrix. "
        "This operation is only meaningful for state vectors.",
        UserWarning,
    )
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

    if n_dim >= 9 and math.get_interface(state) == "tensorflow":
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

    if n_dim >= 9 and math.get_interface(state) == "tensorflow":
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

    if n_dim >= 9 and math.get_interface(state) == "tensorflow":
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

    if n_dim >= 9 and math.get_interface(state) == "tensorflow":
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)
    params = math.cast(op.parameters[0], dtype=complex)
    # First, flip the left side
    axis = op.wires[0] + is_state_batched
    state = _phase_shift(state, axis, phase_factor=math.exp(1j * params))

    # Second, flip the right side
    axis = op.wires[0] + is_state_batched + num_wires
    state = _phase_shift(state, axis, phase_factor=math.exp(-1j * params))

    return state


# pylint: disable=no-cover
@apply_operation.register
def apply_snapshot(
    op: qml.Snapshot, state, is_state_batched: bool = False, debugger=None, **execution_kwargs
):
    """Take a snapshot of the state"""
    if debugger is not None and debugger.active:
        raise NotImplementedError("Snapshot is not implemented yet for mixed states.")
        # measurement = op.hyperparameters["measurement"]

        # shots = execution_kwargs.get("tape_shots")

        # if isinstance(measurement, qml.measurements.StateMP) or not shots:
        #     snapshot = qml.devices.qubit_mixed.measure(measurement, state, is_state_batched)
        # else:
        #     snapshot = qml.devices.qubit_mixed.measure_with_samples(
        #         [measurement],
        #         state,
        #         shots,
        #         is_state_batched,
        #         execution_kwargs.get("rng"),
        #         execution_kwargs.get("prng_key"),
        #     )[0]

        # if op.tag:
        #     debugger.snapshots[op.tag] = snapshot
        # else:
        #     debugger.snapshots[len(debugger.snapshots)] = snapshot
    return state


def apply_diagonal_unitary(op, state, is_state_batched: bool = False, debugger=None, **_):
    """_summary_

    Args:
        op (_type_): _description_
        state (_type_): _description_
        is_state_batched (bool, optional): _description_. Defaults to False.
        debugger (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    channel_wires = op.wires
    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)

    eigvals = op.eigvals()
    eigvals = math.stack(eigvals)
    eigvals = math.reshape(eigvals, [QUDIT_DIM] * len(channel_wires))
    eigvals = math.cast_like(eigvals, state)

    state_indices = alphabet[: 2 * num_wires]

    row_wires_list = channel_wires.tolist()
    row_indices = "".join(alphabet_array[row_wires_list].tolist())

    col_wires_list = [w + num_wires for w in row_wires_list]
    col_indices = "".join(alphabet_array[col_wires_list].tolist())

    # Basically, we want to do, lambda_a rho_ab lambda_b
    einsum_indices = f"{row_indices},{state_indices},{col_indices}->{state_indices}"

    return math.einsum(einsum_indices, eigvals, state, eigvals.conj())


# TODO add special case speedups

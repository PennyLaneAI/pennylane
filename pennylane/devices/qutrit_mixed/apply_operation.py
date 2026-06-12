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
"""Functions to apply operations to a qutrit mixed state."""

# pylint: disable=unused-argument

from functools import singledispatch
from string import ascii_letters as alphabet

import pennylane as qp
from pennylane import math
from pennylane import numpy as np
from pennylane.core.operator import Channel

from .utils import QUDIT_DIM, get_einsum_mapping, get_new_state_einsum_indices, get_num_wires

alphabet_array = np.array(list(alphabet))


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


def apply_operation_einsum(op: qp.operation.Operator, state, is_state_batched: bool = False):
    r"""Apply a quantum channel specified by a list of Kraus operators to subsystems of the
    quantum state. For a unitary gate, there is a single Kraus operator.

    Args:
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        array[complex]: output_state
    """
    einsum_indices = get_einsum_mapping(op, state, _map_indices_apply_channel, is_state_batched)

    num_ch_wires = len(op.wires)

    # This could be pulled into separate function if tensordot is added
    if isinstance(op, Channel):
        kraus = op.kraus_matrices()
    else:
        kraus = [op.matrix()]

    # Shape kraus operators
    kraus_shape = [len(kraus)] + [QUDIT_DIM] * num_ch_wires * 2
    if not isinstance(op, Channel):
        mat = op.matrix()
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

    return math.einsum(einsum_indices, kraus, state, kraus_dagger)


@singledispatch
def apply_operation(
    op: qp.operation.Operator,
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

    .. code-block:: py

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
    >>> apply_operation(qp.TShift(0), state)
    tensor([[0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j]], requires_grad=True)

    """
    return _apply_operation_default(op, state, is_state_batched, debugger)


def _apply_operation_default(op, state, is_state_batched, debugger):
    """The default behaviour of apply_operation, accessed through the standard dispatch
    of apply_operation, as well as conditionally in other dispatches.
    """

    return apply_operation_einsum(op, state, is_state_batched=is_state_batched)
    # TODO add tensordot and benchmark for performance


# TODO add diagonal for speed up.


@apply_operation.register
def apply_snapshot(
    op: qp.Snapshot, state, is_state_batched: bool = False, debugger=None, **execution_kwargs
):
    """Take a snapshot of the mixed state"""
    if debugger and debugger.active:
        measurement = op.hyperparameters["measurement"]

        if op.hyperparameters["shots"] == "workflow":
            shots = execution_kwargs.get("tape_shots")
        else:
            shots = op.hyperparameters["shots"]

        if isinstance(measurement, qp.measurements.StateMP) or not shots:
            snapshot = qp.devices.qutrit_mixed.measure(measurement, state, is_state_batched)
        else:
            snapshot = qp.devices.qutrit_mixed.measure_with_samples(
                measurement,
                state,
                shots,
                is_state_batched,
                execution_kwargs.get("rng"),
                execution_kwargs.get("prng_key"),
            )

        if op.tag:
            debugger.snapshots[op.tag] = snapshot
        else:
            debugger.snapshots[len(debugger.snapshots)] = snapshot

    return state


@apply_operation.register
def apply_identity(op: qp.Identity, state, is_state_batched: bool = False, debugger=None, **_):
    """Applies a :class:`~.Identity` operation by just returning the input state."""
    return state


@apply_operation.register
def apply_density_matrix(
    op: qp.QutritDensityMatrix,
    state,
    is_state_batched: bool = False,
    debugger=None,
    **execution_kwargs,
):
    """
    Applies a QutritDensityMatrix operation by initializing or replacing
    the quantum state with the provided density matrix.

    - If the QutritDensityMatrix covers all wires, we directly return the provided density matrix as the new state.
    - If only a subset of the wires is covered, we:
      1. Partial trace out those wires from the current state to get the density matrix of the complement wires.
      2. Take the tensor product of the complement density matrix and the provided density_matrix.
      3. Reshape to the correct final shape and return.

    Args:
        op (qp.QutritDensityMatrix): The QutritDensityMatrix operation.
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
    expected_dim = QUDIT_DIM**num_wires

    # Cast density_matrix to the same type and device as state
    density_matrix = math.cast_like(density_matrix, state)

    # Extract total wires
    num_state_wires = get_num_wires(state, is_state_batched)
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
    # Note: partial_trace expects state in 2D (dim, dim) or 3D (batch, dim, dim) format,
    # but the mixed device stores state in tensor format (QUDIT_DIM, QUDIT_DIM, ..., QUDIT_DIM).
    # We need to reshape state to 2D/3D format first.
    state_dim = QUDIT_DIM**num_state_wires
    if is_state_batched:
        batch_size = math.shape(state)[0]
        state_2d = math.reshape(state, (batch_size, state_dim, state_dim))
    else:
        state_2d = math.reshape(state, (state_dim, state_dim))

    sigma = qp.math.partial_trace(state_2d, indices=op_wires, qudit_dim=QUDIT_DIM)
    # sigma now has shape:
    # (batch_size, QUDIT_DIM^(n - num_wires), QUDIT_DIM^(n - num_wires)) where n = total wires

    # 2. Take kron(sigma, density_matrix)
    sigma_dim = QUDIT_DIM ** len(complement_wires)  # dimension of complement subsystem
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

    # rho now has shape (batch_size?, QUDIT_DIM^n, QUDIT_DIM^n)

    # 3. Reshape rho into the full tensor form [QUDIT_DIM]*(2*n) or [batch_size] + [QUDIT_DIM]*(2*n)
    final_shape = ([batch_size] if is_state_batched else []) + [QUDIT_DIM] * (2 * num_state_wires)
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
    # [batch?] + [QUDIT_DIM]*num_wires (left side) + [QUDIT_DIM]*num_wires (right side)
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

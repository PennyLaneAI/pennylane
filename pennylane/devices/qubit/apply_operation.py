# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to apply an operation to a state vector."""
# pylint: disable=unused-argument

from functools import singledispatch
from string import ascii_letters as alphabet

import numpy as np
import scipy as sp

import pennylane as qml
from pennylane import math, ops
from pennylane.measurements import MidMeasureMP
from pennylane.operation import Operator
from pennylane.ops import Conditional

SQRT2INV = 1 / math.sqrt(2)

EINSUM_OP_WIRECOUNT_PERF_THRESHOLD = 3
EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = 13


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


def apply_operation_einsum(op: Operator, state, is_state_batched: bool = False):
    """Apply ``Operator`` to ``state`` using ``einsum``. This is more efficent at lower qubit
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
    if (
        math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        mat = math.cast_like(op.matrix(), state)
    else:
        mat = op.matrix() + 0j

    total_indices = len(state.shape) - is_state_batched
    num_indices = len(op.wires)

    state_indices = alphabet[:total_indices]
    affected_indices = "".join(alphabet[i] for i in op.wires)

    new_indices = alphabet[total_indices : total_indices + num_indices]

    new_state_indices = state_indices
    for old, new in zip(affected_indices, new_indices):
        new_state_indices = new_state_indices.replace(old, new)

    einsum_indices = (
        f"...{new_indices}{affected_indices},...{state_indices}->...{new_state_indices}"
    )

    new_mat_shape = [2] * (num_indices * 2)
    dim = 2**num_indices
    batch_size = math.get_batch_size(mat, (dim, dim), dim**2)
    if batch_size is not None:
        # Add broadcasting dimension to shape
        new_mat_shape = [batch_size] + new_mat_shape
        if op.batch_size is None:
            op._batch_size = batch_size  # pylint:disable=protected-access
    reshaped_mat = math.reshape(mat, new_mat_shape)

    return math.einsum(einsum_indices, reshaped_mat, state)


def apply_operation_tensordot(op: Operator, state, is_state_batched: bool = False):
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
    if (
        math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        mat = math.cast_like(op.matrix(), state)
    else:
        mat = op.matrix() + 0j

    total_indices = len(state.shape) - is_state_batched
    num_indices = len(op.wires)

    new_mat_shape = [2] * (num_indices * 2)
    dim = 2**num_indices
    batch_size = math.get_batch_size(mat, (dim, dim), dim**2)
    if is_mat_batched := batch_size is not None:
        # Add broadcasting dimension to shape
        new_mat_shape = [batch_size] + new_mat_shape
        if op.batch_size is None:
            op._batch_size = batch_size  # pylint:disable=protected-access
    reshaped_mat = math.reshape(mat, new_mat_shape)

    mat_axes = list(range(-num_indices, 0))
    state_axes = [i + is_state_batched for i in op.wires]
    axes = (mat_axes, state_axes)

    tdot = math.tensordot(reshaped_mat, state, axes=axes)

    # tensordot causes the axes given in `wires` to end up in the first positions
    # of the resulting tensor. This corresponds to a (partial) transpose of
    # the correct output state
    # We'll need to invert this permutation to put the indices in the correct place
    unused_idxs = [i for i in range(total_indices) if i not in op.wires]
    perm = list(op.wires) + unused_idxs
    if is_mat_batched:
        perm = [0] + [i + 1 for i in perm]
    if is_state_batched:
        perm.insert(num_indices, -1)

    inv_perm = math.argsort(perm)
    return math.transpose(tdot, inv_perm)


@singledispatch
def apply_operation(
    op: Operator,
    state,
    is_state_batched: bool = False,
    debugger=None,
    **_,
):
    """Apply and operator to a given state.

    Args:
        op (Operator): The operation to apply to ``state``
        state (TensorLike): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not
        debugger (_Debugger): The debugger to use
        **execution_kwargs (Optional[dict]): Optional keyword arguments needed for applying
            some operations described below.

    Keyword Arguments:
        mid_measurements (dict, None): Mid-circuit measurement dictionary mutated to record the sampled value
        interface (str): The machine learning interface of the state
        postselect_mode (str): Configuration for handling shots with mid-circuit measurement
            postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
            keep the same number of shots. ``None`` by default.
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

        The shape of state should be ``[2]*num_wires``.

    This is a ``functools.singledispatch`` function, so additional specialized kernels
    for specific operations can be registered like:

    .. code-block:: python

        @apply_operation.register
        def _(op: type_op, state):
            # custom op application method here

    **Example:**

    >>> state = np.zeros((2,2))
    >>> state[0][0] = 1
    >>> state
    tensor([[1., 0.],
        [0., 0.]], requires_grad=True)
    >>> apply_operation(qml.X(0), state)
    tensor([[0., 0.],
        [1., 0.]], requires_grad=True)

    """
    return _apply_operation_default(op, state, is_state_batched, debugger)


def apply_operation_csr_matrix(op, state, is_state_batched: bool = False):
    """The csr_matrix specialized version apply operation."""
    # State is numpy array, should have been stored in tensor version
    # remember the initial shape and recover in the end
    if sp.sparse.issparse(state):
        raise TypeError("State should not be sparse in default qubit pipeline")
    original_shape = math.shape(state)
    num_wires = len(original_shape) - int(is_state_batched)
    full_state = math.reshape(state, [-1, 2**num_wires])  # expected: [batch_size, 2**num_wires]
    state_opT = full_state @ op.sparse_matrix(wire_order=range(num_wires)).T
    state_reshaped = math.reshape(state_opT, original_shape)
    return state_reshaped


def _apply_operation_default(op, state, is_state_batched, debugger):
    """The default behaviour of apply_operation, accessed through the standard dispatch
    of apply_operation, as well as conditionally in other dispatches."""
    if op.has_sparse_matrix and not op.has_matrix:
        return apply_operation_csr_matrix(op, state, is_state_batched=is_state_batched)
    if (
        len(op.wires) < EINSUM_OP_WIRECOUNT_PERF_THRESHOLD
        and math.ndim(state) < EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD
    ) or (op.batch_size and is_state_batched):
        return apply_operation_einsum(op, state, is_state_batched=is_state_batched)
    return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)


@apply_operation.register
def apply_conditional(
    op: Conditional,
    state,
    is_state_batched: bool = False,
    debugger=None,
    **execution_kwargs,
):
    """Applies a conditional operation.

    Args:
        op (Operator): The operation to apply to ``state``
        state (TensorLike): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not
        debugger (_Debugger): The debugger to use
        mid_measurements (dict, None): Mid-circuit measurement dictionary mutated to record the sampled value
        interface (str): The machine learning interface of the state
        rng (Optional[numpy.random._generator.Generator]): A NumPy random number generator.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, a ``numpy.random.default_rng`` will be used for sampling.

    Returns:
        ndarray: output state
    """
    mid_measurements = execution_kwargs.get("mid_measurements", None)
    rng = execution_kwargs.get("rng", None)
    prng_key = execution_kwargs.get("prng_key", None)
    interface = math.get_deep_interface(state)
    if interface == "jax":
        # pylint: disable=import-outside-toplevel
        from jax.lax import cond

        return cond(
            op.meas_val.concretize(mid_measurements),
            lambda x: apply_operation(
                op.base,
                x,
                is_state_batched=is_state_batched,
                debugger=debugger,
                mid_measurements=mid_measurements,
                rng=rng,
                prng_key=prng_key,
            ),
            lambda x: x,
            state,
        )
    if op.meas_val.concretize(mid_measurements):
        return apply_operation(
            op.base,
            state,
            is_state_batched=is_state_batched,
            debugger=debugger,
            mid_measurements=mid_measurements,
            rng=rng,
            prng_key=prng_key,
        )
    return state


@apply_operation.register
def apply_mid_measure(
    op: MidMeasureMP, state, is_state_batched: bool = False, debugger=None, **execution_kwargs
):
    """Applies a native mid-circuit measurement.

    Args:
        op (Operator): The operation to apply to ``state``
        state (TensorLike): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not
        debugger (_Debugger): The debugger to use
        mid_measurements (dict, None): Mid-circuit measurement dictionary mutated to record the sampled value
        postselect_mode (str): Configuration for handling shots with mid-circuit measurement
            postselection. Use ``"hw-like"`` to discard invalid shots and ``"fill-shots"`` to
            keep the same number of shots. ``None`` by default.
        rng (Optional[numpy.random._generator.Generator]): A NumPy random number generator.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, a ``numpy.random.default_rng`` will be used for sampling.

    Returns:
        ndarray: output state
    """
    mid_measurements = execution_kwargs.get("mid_measurements", None)
    rng = execution_kwargs.get("rng", None)
    prng_key = execution_kwargs.get("prng_key", None)

    if is_state_batched:
        raise ValueError("MidMeasureMP cannot be applied to batched states.")
    wire = op.wires
    interface = math.get_deep_interface(state)

    axis = wire.toarray()[0]
    slices = [slice(None)] * math.ndim(state)
    slices[axis] = 0
    prob0 = math.real(math.norm(state[tuple(slices)])) ** 2

    if prng_key is not None:
        # pylint: disable=import-outside-toplevel
        from jax.random import binomial

        def binomial_fn(n, p):
            return binomial(prng_key, n, p).astype(int)

    else:
        binomial_fn = np.random.binomial if rng is None else rng.binomial

    sample = binomial_fn(1, 1 - prob0)

    assert mid_measurements is not None
    mid_measurements[op] = sample

    # Using apply_operation(qml.QubitUnitary,...) instead of apply_operation(qml.Projector([sample], wire),...)
    # to select the sample branch enables jax.jit and prevents it from using Python callbacks
    matrix = math.array([[(sample + 1) % 2, 0.0], [0.0, (sample) % 2]], like=interface)
    state = apply_operation(
        ops.QubitUnitary(matrix, wire),
        state,
        is_state_batched=is_state_batched,
        debugger=debugger,
    )
    state = state / math.norm(state)

    # Using apply_operation(qml.QubitUnitary,...) instead of apply_operation(qml.X(wire), ...)
    # to reset enables jax.jit and prevents it from using Python callbacks
    element = op.reset and sample == 1
    matrix = math.array(
        [[(element + 1) % 2, (element) % 2], [(element) % 2, (element + 1) % 2]],
        like=interface,
        dtype=float,
    )
    state = apply_operation(
        ops.QubitUnitary(matrix, wire), state, is_state_batched=is_state_batched, debugger=debugger
    )

    return state


@apply_operation.register
def apply_identity(op: ops.Identity, state, is_state_batched: bool = False, debugger=None, **_):
    """Applies a :class:`~.Identity` operation by just returning the input state."""
    return state


@apply_operation.register
def apply_global_phase(
    op: ops.GlobalPhase, state, is_state_batched: bool = False, debugger=None, **_
):
    """Applies a :class:`~.GlobalPhase` operation by multiplying the state by ``exp(1j * op.data[0])``"""
    return math.exp(-1j * math.cast(op.data[0], complex)) * state


@apply_operation.register
def apply_paulix(op: ops.X, state, is_state_batched: bool = False, debugger=None, **_):
    """Apply :class:`pennylane.PauliX` operator to the quantum state"""
    axis = op.wires[0] + is_state_batched
    return math.roll(state, 1, axis)


@apply_operation.register
def apply_pauliz(op: ops.Z, state, is_state_batched: bool = False, debugger=None, **_):
    """Apply pauliz to state."""

    axis = op.wires[0] + is_state_batched
    n_dim = math.ndim(state)

    if (
        n_dim >= 9 and math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)

    sl_0 = _get_slice(0, axis, n_dim)
    sl_1 = _get_slice(1, axis, n_dim)

    # must be first state and then -1 because it breaks otherwise
    state1 = math.multiply(state[sl_1], -1)
    return math.stack([state[sl_0], state1], axis=axis)


@apply_operation.register
def apply_phaseshift(op: ops.PhaseShift, state, is_state_batched: bool = False, debugger=None, **_):
    """Apply PhaseShift to state."""

    n_dim = math.ndim(state)

    if (
        n_dim >= 9 and math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)

    axis = op.wires[0] + is_state_batched

    sl_0 = _get_slice(0, axis, n_dim)
    sl_1 = _get_slice(1, axis, n_dim)

    params = math.cast(op.parameters[0], dtype=complex)
    state0 = state[sl_0]
    state1 = state[sl_1]
    if op.batch_size is not None:
        interface = math.get_interface(state)
        if interface == "torch":
            params = math.array(params, like=interface)
        if is_state_batched:
            params = math.reshape(params, (-1,) + (1,) * (n_dim - 2))
        else:
            axis = axis + 1
            params = math.reshape(params, (-1,) + (1,) * (n_dim - 1))
            state0 = math.expand_dims(state0, 0) + math.zeros_like(params)
            state1 = math.expand_dims(state1, 0)
    state1 = math.multiply(math.cast(state1, dtype=complex), math.exp(1.0j * params))
    state = math.stack([state0, state1], axis=axis)
    return state


@apply_operation.register
def apply_T(op: ops.T, state, is_state_batched: bool = False, debugger=None, **_):
    """Apply T to state."""

    axis = op.wires[0] + is_state_batched
    n_dim = math.ndim(state)

    if (
        n_dim >= 9 and math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)

    sl_0 = _get_slice(0, axis, n_dim)
    sl_1 = _get_slice(1, axis, n_dim)

    state1 = math.multiply(math.cast(state[sl_1], dtype=complex), math.exp(0.25j * np.pi))
    return math.stack([state[sl_0], state1], axis=axis)


@apply_operation.register
def apply_S(op: ops.S, state, is_state_batched: bool = False, debugger=None, **_):
    """Apply S to state."""

    axis = op.wires[0] + is_state_batched
    n_dim = math.ndim(state)

    if (
        n_dim >= 9 and math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)

    sl_0 = _get_slice(0, axis, n_dim)
    sl_1 = _get_slice(1, axis, n_dim)

    state1 = math.multiply(math.cast(state[sl_1], dtype=complex), 1j)
    return math.stack([state[sl_0], state1], axis=axis)


@apply_operation.register
def apply_cnot(op: ops.CNOT, state, is_state_batched: bool = False, debugger=None, **_):
    """Apply cnot gate to state."""
    target_axes = (op.wires[1] - 1 if op.wires[1] > op.wires[0] else op.wires[1]) + is_state_batched
    control_axes = op.wires[0] + is_state_batched
    n_dim = math.ndim(state)

    if (
        n_dim >= 9 and math.get_interface(state) == "tensorflow"
    ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
        return apply_operation_tensordot(op, state, is_state_batched=is_state_batched)

    sl_0 = _get_slice(0, control_axes, n_dim)
    sl_1 = _get_slice(1, control_axes, n_dim)

    state_x = math.roll(state[sl_1], 1, target_axes)
    return math.stack([state[sl_0], state_x], axis=control_axes)


@apply_operation.register
def apply_multicontrolledx(
    op: ops.MultiControlledX,
    state,
    is_state_batched: bool = False,
    debugger=None,
    **_,
):
    r"""Apply MultiControlledX to a state with the default einsum/tensordot choice
    for 8 operation wires or less. Otherwise, apply a custom kernel based on
    composing transpositions, rolling of control axes and the CNOT logic above."""
    if len(op.wires) < 9:
        return _apply_operation_default(op, state, is_state_batched, debugger)
    ctrl_wires = [w + is_state_batched for w in op.control_wires]
    # apply x on all control wires with control value 0
    roll_axes = [w for val, w in zip(op.control_values, ctrl_wires) if val is False]
    for ax in roll_axes:
        state = math.roll(state, 1, ax)

    orig_shape = math.shape(state)
    # Move the axes into the order [(batch), other, target, controls]
    transpose_axes = (
        np.array(
            [
                w - is_state_batched
                for w in range(len(orig_shape))
                if w - is_state_batched not in op.wires
            ]
            + [op.wires[-1]]
            + op.wires[:-1].tolist()
        )
        + is_state_batched
    )
    state = math.transpose(state, transpose_axes)

    # Reshape the state into 3-dimensional array with axes [batch+other, target, controls]
    state = math.reshape(state, (-1, 2, 2 ** (len(op.wires) - 1)))

    # The part of the state to which we want to apply PauliX is now in the last entry along the
    # third axis. Extract it, apply the PauliX along the target axis (1), and append a dummy axis
    state_x = math.roll(state[:, :, -1], 1, 1)[:, :, np.newaxis]

    # Stack the transformed part of the state with the unmodified rest of the state
    state = math.concatenate([state[:, :, :-1], state_x], axis=2)

    # Reshape into original shape and undo the transposition
    state = math.transpose(math.reshape(state, orig_shape), np.argsort(transpose_axes))

    # revert x on all "wrong" controls
    for ax in roll_axes:
        state = math.roll(state, 1, ax)
    return state


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
    if len(op.wires) < 9:
        return _apply_operation_default(op, state, is_state_batched, debugger)
    return _apply_grover_without_matrix(state, op.wires, is_state_batched)


def _apply_grover_without_matrix(state, op_wires, is_state_batched):
    r"""Apply GroverOperator to state. This method uses that this operator
    is :math:`2*P-\mathbb{I}`, where :math:`P` is the projector onto the
    all-plus state. This allows us to compute the new state by replacing summing
    over all axes on which the operation acts, and "filling in" the all-plus state
    in the resulting lower-dimensional state via a Kronecker product.
    """
    num_wires = len(op_wires)
    # 2 * Squared normalization of the all-plus state on the op wires
    # (squared, because we skipped the normalization when summing, 2* because of the def of Grover)
    prefactor = 2 ** (1 - num_wires)

    # The axes to sum over in order to obtain <+|\psi>, where <+| only acts on the op wires.
    sum_axes = [w + is_state_batched for w in op_wires]
    collapsed = math.sum(state, axis=tuple(sum_axes))

    if num_wires == (len(math.shape(state)) - is_state_batched):
        # If the operation acts on all wires, we can skip the tensor product with all-ones state
        new_shape = (-1,) + (1,) * num_wires if is_state_batched else (1,) * num_wires
        return prefactor * math.reshape(collapsed, new_shape) - state
        # [todo]: Once Tensorflow support expand_dims with multiple axes in the second argument,
        # use the following line instead of the two above.
        # return prefactor * math.expand_dims(collapsed, sum_axes) - state

    all_plus = math.cast_like(math.full([2] * num_wires, prefactor), state)
    # After the Kronecker product (realized with tensordot with axes=0), we need to move
    # the new axes to the summed-away axes' positions. Finally, subtract the original state.
    source = list(range(math.ndim(collapsed), math.ndim(state)))
    # Probably it will be better to use math.full or math.tile to create the outer product
    # here computed with math.tensordot. However, Tensorflow and Torch do not have full support
    return math.moveaxis(math.tensordot(collapsed, all_plus, axes=0), source, sum_axes) - state


@apply_operation.register
def apply_snapshot(
    op: ops.Snapshot, state, is_state_batched: bool = False, debugger=None, **execution_kwargs
):
    """Take a snapshot of the state."""
    if debugger is None or not debugger.active:
        return state
    measurement = op.hyperparameters["measurement"]
    if op.hyperparameters["shots"] == "workflow":
        shots = execution_kwargs.get("tape_shots")
    else:
        shots = op.hyperparameters["shots"]

    if shots:
        snapshot = qml.devices.qubit.measure_with_samples(
            [measurement],
            state,
            shots,
            is_state_batched,
            execution_kwargs.get("rng"),
            execution_kwargs.get("prng_key"),
        )[0]
    else:
        snapshot = qml.devices.qubit.measure(measurement, state, is_state_batched)

    if op.tag is None:
        debugger.snapshots[len(debugger.snapshots)] = snapshot
    elif op.tag not in debugger.snapshots:
        debugger.snapshots[op.tag] = snapshot
    elif isinstance(debugger.snapshots[op.tag], list):
        debugger.snapshots[op.tag].append(snapshot)
    else:
        debugger.snapshots[op.tag] = [debugger.snapshots[op.tag], snapshot]

    return state


# pylint:disable=import-outside-toplevel
@apply_operation.register
def apply_parametrized_evolution(
    op: qml.pulse.ParametrizedEvolution,
    state,
    is_state_batched: bool = False,
    debugger=None,
    **_,
):
    """Apply ParametrizedEvolution by evolving the state rather than the operator matrix
    if we are operating on more than half of the subsystem"""

    # shape(state) is static (not a tracer), we can use an if statement
    num_wires = len(math.shape(state)) - is_state_batched
    state = math.cast(state, complex)
    if (
        2 * len(op.wires) <= num_wires
        or op.hyperparameters["complementary"]
        or (is_state_batched and op.hyperparameters["return_intermediate"])
    ):
        # the subsystem operated on is half as big as the total system, or less
        # or we want complementary time evolution
        # or both the state and the operation have a batch dimension
        # --> evolve matrix
        return _apply_operation_default(op, state, is_state_batched, debugger)
    # otherwise --> evolve state
    return _evolve_state_vector_under_parametrized_evolution(op, state, num_wires, is_state_batched)


def _evolve_state_vector_under_parametrized_evolution(
    operation: qml.pulse.ParametrizedEvolution, state, num_wires, is_state_batched
):
    """Uses an odeint solver to compute the evolution of the input ``state`` under the given
    ``ParametrizedEvolution`` operation.

    Args:
        state (array[complex]): input state
        operation (ParametrizedEvolution): operation to apply on the state

    Raises:
        ValueError: If the parameters and time windows of the ``ParametrizedEvolution`` are
            not defined.

    Returns:
        TensorLike[complex]: output state
    """

    try:
        import jax
        from jax.experimental.ode import odeint

        from pennylane.pulse.parametrized_hamiltonian_pytree import ParametrizedHamiltonianPytree

    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Module jax is required for the ``ParametrizedEvolution`` class. "
            "You can install jax via: pip install jax~=0.6.0"
        ) from e

    if operation.data is None or operation.t is None:
        raise ValueError(
            "The parameters and the time window are required to execute a ParametrizedEvolution "
            "You can update these values by calling the ParametrizedEvolution class: EV(params, t)."
        )

    if is_state_batched:
        batch_dim = state.shape[0]
        state = math.moveaxis(state.reshape((batch_dim, 2**num_wires)), 1, 0)
        out_shape = [2] * num_wires + [batch_dim]  # this shape is before moving the batch_dim back
    else:
        state = state.flatten()
        out_shape = [2] * num_wires

    with jax.ensure_compile_time_eval():
        H_jax = ParametrizedHamiltonianPytree.from_hamiltonian(  # pragma: no cover
            operation.H,
            dense=operation.dense,
            wire_order=list(np.arange(num_wires)),
        )

    def fun(y, t):
        """dy/dt = -i H(t) y"""
        return (-1j * H_jax(operation.data, t=t)) @ y

    result = odeint(fun, state, operation.t, **operation.odeint_kwargs)
    if operation.hyperparameters["return_intermediate"]:
        return math.reshape(result, [-1] + out_shape)
    result = math.reshape(result[-1], out_shape)
    if is_state_batched:
        return math.moveaxis(result, -1, 0)
    return result

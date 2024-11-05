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
"""Functions to prepare a state."""

from collections.abc import Iterable
from typing import Union

import pennylane as qml
import pennylane.numpy as np
from pennylane import math
from pennylane.wires import Wires

DIM_MISMATCH_ERROR = "Incorrect state size. Expected size 2 to the power of num_wires."
ZERO_NORM_ERROR = "Input state must not be the zero vector."


def create_initial_state(
    wires: Union[qml.wires.Wires, Iterable],
    prep_operation: qml.operation.StatePrepBase = None,
    like: str = None,
):
    r"""
    Returns an initial state, defaulting to :math:`\ket{0}` if no state-prep operator is provided.

    Args:
        wires (Union[Wires, Iterable]): The wires to be present in the initial state
        prep_operation (Optional[StatePrepBase]): An operation to prepare the initial state
        like (Optional[str]): The machine learning interface used to create the initial state.
            Defaults to None

    Returns:
        array: The initial density matrix (tensor form) of a circuit
    """
    num_wires = len(wires)
    num_axes = (
        2 * num_wires
    )  # we initialize the density matrix as the tensor form to keep compatiblity with the rest of the module
    if not prep_operation:
        state = np.zeros((2,) * num_axes, dtype=complex)
        state[(0,) * num_axes] = 1
        return math.asarray(state, like=like)

    # Here, to avoid the extension of the previous class defined in `StatePrepBase`, we directly call the method `state_vector`. However, this requires some levels of abstract translation between state vectors and density matrices.
    # The concise explanation is that, either state vectors or density matrices, they are always originally just higher-rank tensors. Only diff is that state vectors are originally of ranks num_wires, while density matrices are of ranks 2*num_wires. Therefore, we can always define the density matrices as the same wires, appended by a 'shifted' set of wires by num_wires. Actually, this idea is also used in the wire sewing technique in catalyst package.
    dm_wires = qml.wires.Wires(wires + [w + num_wires for w in wires])
    density_matrix = prep_operation.state_vector(device_wires=list(dm_wires))
    density_matrix = np.reshape(density_matrix, (-1,) + (2,) * num_axes)
    dtype = str(density_matrix.dtype)
    floating_single = "float32" in dtype or "complex64" in dtype
    dtype = "complex64" if floating_single else "complex128"
    dtype = "complex128" if like == "tensorflow" else dtype
    return math.cast(math.asarray(density_matrix, like=like), dtype)


def _create_basis_state(num_wires, index, dtype=complex):
    r"""Return the density matrix representing a computational basis state over all wires.

    This function creates a density matrix for a pure computational basis state in a multi-qubit
    system. The resulting state is a projector onto the basis state specified by the index.

    Args:
        num_wires (int): Number of qubits/wires in the system. Must be positive.
        index (int): Index of the computational basis state to create. Must be in range
            [0, 2^num_wires - 1].
        dtype (numpy.dtype, optional): Data type of the output array. Defaults to np.complex128.

    Returns:
        jax.numpy.ndarray: A reshaped density matrix with dimensions [2, 2, ..., 2] (2*num_wires times),
            representing the pure state |index⟩⟨index|.

    Examples:
        >>> # Create |0⟩⟨0| state for 1 qubit
        >>> rho = _create_basis_state(1, 0)
        >>> print(rho.shape)  # (2, 2)
        >>> # Create |01⟩⟨01| state for 2 qubits
        >>> rho = _create_basis_state(2, 1)
        >>> print(rho.shape)  # (2, 2, 2, 2)

    Notes:
        - The function first creates a 2^n × 2^n matrix and then reshapes it to the
          tensor product structure with 2*num_wires dimensions of size 2.
        - The resulting density matrix has trace 1 and represents a pure state.
    """
    rho = np.zeros((2**num_wires, 2**num_wires), dtype=dtype)
    rho[index, index] = 1
    return np.reshape(rho, [2] * (2 * num_wires))


def _apply_state_vector(device_wires, state, full_wires=None):
    r"""Initialize the internal state in a specified pure state.

    Args:
        full_wires (Wires): all wires of the device
        state (array[complex]): normalized input state of length
            ``2**num_wires``, where ``2`` is the dimension of the system.
        num_wires (int): number of wires that get initialized in the state

    Returns:
        array[complex]: complex array of shape ``[2] * (2 * num_wires)``
        representing the density matrix of this state, where ``2`` is
        the dimension of the system.
    """

    num_wires = len(device_wires)
    # Check the wires are in the correct order
    if math.size(state) != 2**num_wires:
        raise ValueError(DIM_MISMATCH_ERROR)

    # Check normalization
    norm = np.linalg.norm(state)
    if math.allclose(norm, 0):
        raise ValueError(ZERO_NORM_ERROR)

    if not math.allclose(norm, 1):  # If not yet normalized, normalize it
        state = state / norm

    # Check the full wires
    if full_wires is None:
        full_wires = device_wires

    full_num_wires = len(full_wires)

    if num_wires == full_num_wires and sorted(device_wires.labels) == list(device_wires.labels):
        # Initialize the entire set of wires with the state
        rho = math.outer(state, math.conj(state))
        rho = math.reshape(rho, [2] * 2 * num_wires)
        return math.cast_like(rho, state)

    expected_shape = [2] * num_wires
    expected_size = 2**num_wires
    is_batch = False
    if math.get_batch_size(state, expected_shape=expected_shape, expected_size=expected_size):
        is_batch = True
    op_vector = math.reshape(state, (-1,) + (2,) * num_wires if is_batch else (2,) * num_wires)
    # add zeros for each wire that isn't being set
    extra_wires = Wires(set(full_wires) - set(device_wires))
    for _ in extra_wires:
        op_vector = math.stack([op_vector, math.zeros_like(op_vector)], axis=-1)

    # transpose from operator wire order to provided wire order
    current_wires = device_wires + extra_wires
    transpose_axes = [current_wires.index(w) for w in full_wires]
    if is_batch:
        transpose_axes = [0] + [a + 1 for a in transpose_axes]
    full_vector = math.transpose(op_vector, transpose_axes)
    full_vector = math.reshape(full_vector, [-1] + [2**full_num_wires])
    return math.reshape(math.dm_from_state_vector(full_vector), [2] * 2 * full_num_wires)


# def _apply_density_matrix(state, device_wires):
#     r"""Initialize the internal state in a specified mixed state.
#     If not all the wires are specified in the full state :math:`\rho`, remaining subsystem is filled by
#     `\mathrm{tr}_in(\rho)`, which results in the full system state :math:`\mathrm{tr}_{in}(\rho) \otimes \rho_{in}`,
#     where :math:`\rho_{in}` is the argument `state` of this function and :math:`\mathrm{tr}_{in}` is a partial
#     trace over the subsystem to be replaced by this operation.

#         Args:
#             state (array[complex]): density matrix of length
#                 ``(2**len(wires), 2**len(wires))``
#             device_wires (Wires): wires that get initialized in the state
#     """

#     # translate to wire labels used by device
#     device_wires = map_wires(device_wires)

#     state = np.asarray(state, dtype=C_DTYPE)
#     state = np.reshape(state, (-1,))

#     state_dim = 2 ** len(device_wires)
#     dm_dim = state_dim**2
#     if dm_dim != state.shape[0]:
#         raise ValueError("Density matrix must be of length (2**wires, 2**wires)")

#     if not qml.math.is_abstract(state) and not np.allclose(
#         np.trace(np.reshape(state, (state_dim, state_dim))), 1.0, atol=tolerance
#     ):
#         raise ValueError("Trace of density matrix is not equal one.")

#     if len(device_wires) == num_wires and sorted(device_wires.labels) == list(device_wires.labels):
#         # Initialize the entire wires with the state

#         _state = np.reshape(state, [2] * 2 * num_wires)
#         _pre_rotated_state = _state

#     else:
#         # Initialize tr_in(ρ) ⊗ ρ_in with transposed wires where ρ is the density matrix before this operation.

#         complement_wires = list(sorted(list(set(range(num_wires)) - set(device_wires))))
#         sigma = density_matrix(Wires(complement_wires))
#         rho = np.kron(sigma, state.reshape(state_dim, state_dim))
#         rho = rho.reshape([2] * 2 * num_wires)

#         # Construct transposition axis to revert back to the original wire order
#         left_axes = []
#         right_axes = []
#         complement_wires_count = len(complement_wires)
#         for i in range(num_wires):
#             if i in device_wires:
#                 index = device_wires.index(i)
#                 left_axes.append(complement_wires_count + index)
#                 right_axes.append(complement_wires_count + index + num_wires)
#             elif i in complement_wires:
#                 index = complement_wires.index(i)
#                 left_axes.append(index)
#                 right_axes.append(index + num_wires)
#         transpose_axes = left_axes + right_axes
#         rho = np.transpose(rho, axes=transpose_axes)
#         assert qml.math.is_abstract(rho) or np.allclose(
#             np.trace(np.reshape(rho, (2**num_wires, 2**num_wires))),
#             1.0,
#             atol=tolerance,
#         )

#         _state = np.asarray(rho, dtype=C_DTYPE)
#         _pre_rotated_state = _state

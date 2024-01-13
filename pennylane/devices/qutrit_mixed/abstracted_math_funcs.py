
"""Temporary file for putting in abstracted pennylane math functions"""
from string import ascii_letters as ABC
import itertools
import functools
from autoray import numpy as np
from pennylane import math
from pennylane.math.quantum import _check_density_matrix
import pennylane as qml
from pennylane.math import (cast, get_interface, convert_like, einsum)
ABC_ARRAY = np.array(list(ABC))



def reduce_dm(density_matrix, indices, check_state=False, c_dtype="complex128", qudit_dim=3):
    """Compute the density matrix from a state represented with a density matrix.
    #TODO FROM math.quantum

    Args:
        density_matrix (tensor_like): 2D or 3D density matrix tensor. This tensor should be of size ``(2**N, 2**N)`` or
            ``(batch_dim, 2**N, 2**N)``, for some integer number of wires``N``.
        indices (list(int)): List of indices in the considered subsystem.
        check_state (bool): If True, the function will check the state validity (shape and norm).
        c_dtype (str): Complex floating point precision type.

    Returns:
        tensor_like: Density matrix of size ``(2**len(indices), 2**len(indices))`` or ``(batch_dim, 2**len(indices), 2**len(indices))``

    .. seealso:: :func:`pennylane.math.reduce_statevector`, and :func:`pennylane.density_matrix`

    **Example**
    """
    density_matrix = cast(density_matrix, dtype=c_dtype)

    if check_state:
        _check_density_matrix(density_matrix)

    if len(np.shape(density_matrix)) == 2:
        batch_dim, dim = None, density_matrix.shape[0]
    else:
        batch_dim, dim = density_matrix.shape[:2]

    num_indices = int(np.log2(dim)/np.log2(qudit_dim))
    consecutive_indices = list(range(num_indices))

    # Return the full density matrix if all the wires are given, potentially permuted
    if len(indices) == num_indices:
        return _permute_dense_matrix(density_matrix, consecutive_indices, indices, batch_dim, qudit_dim)

    if batch_dim is None:
        density_matrix = qml.math.stack([density_matrix])

    # Compute the partial trace
    traced_wires = [x for x in consecutive_indices if x not in indices]
    density_matrix = _batched_partial_trace(density_matrix, traced_wires, qudit_dim)

    if batch_dim is None:
        density_matrix = density_matrix[0]

    # Permute the remaining indices of the density matrix
    return _permute_dense_matrix(density_matrix, sorted(indices), indices, batch_dim, qudit_dim)


#TODO FROM
def _permute_dense_matrix(matrix, wires, wire_order, batch_dim, qudit_dim=3):
    """Permute the matrix to match the wires given in `wire_order`.

    #TODO FROM math.matrix_manipulation
    Args:
        matrix (np.ndarray): matrix to permute
        wires (list): wires determining the subspace that base matrix acts on; a base matrix of
            dimension :math:`2^n` acts on a subspace of :math:`n` wires
        wire_order (list): global wire order, which has to contain all wire labels in ``wires``,
            but can also contain additional labels
        batch_dim (int or None): Batch dimension. If ``None``, batching is ignored.

    Returns:
        np.ndarray: permuted matrix
    """
    if wires == wire_order:
        return matrix

    # compute the permutations needed to match wire order
    perm = [wires.index(wire) for wire in wire_order]
    num_wires = len(wire_order)

    perm += [p + num_wires for p in perm]
    if batch_dim:
        perm = [0] + [p + 1 for p in perm]

    # reshape matrix to match wire values e.g. mat[0, 0, 0, 0] = <00|mat|00>
    # with this reshape we can easily swap wires
    shape = [batch_dim] + [qudit_dim] * (num_wires * 2) if batch_dim else [qudit_dim] * (num_wires * 2)
    matrix = qml.math.reshape(matrix, shape)
    # transpose matrix
    matrix = qml.math.transpose(matrix, axes=perm)
    # reshape back
    shape = [batch_dim] + [qudit_dim**num_wires] * 2 if batch_dim else [qudit_dim**num_wires] * 2
    return qml.math.reshape(matrix, shape)

def _batched_partial_trace(density_matrix, indices, qudit_dim=3):
    """Compute the reduced density matrix by tracing out the provided indices.

    Args:
        density_matrix (tensor_like): 3D density matrix tensor. This tensor should be of size
            ``(batch_dim, 2**N, 2**N)``, for some integer number of wires``N``.
        indices (list(int)): List of indices to be traced.

    Returns:
        tensor_like: (reduced) Density matrix of size ``(batch_dim, 2**len(wires), 2**len(wires))``
    """
    # Autograd does not support same indices sum in backprop, and tensorflow
    # has a limit of 8 dimensions if same indices are used
    if get_interface(density_matrix) in ["autograd", "tensorflow"]:
        return _batched_partial_trace_nonrep_indices(density_matrix, indices)

    # Dimension and reshape
    batch_dim, dim = density_matrix.shape[:2]
    num_indices = int(np.log2(dim)/np.log2(qudit_dim))
    rho_dim = 2 * num_indices

    density_matrix = np.reshape(density_matrix, [batch_dim] + [qudit_dim] * 2 * num_indices)
    indices = np.sort(indices)

    # For loop over wires
    for i, target_index in enumerate(indices):
        target_index = target_index - i
        state_indices = ABC[1 : rho_dim - 2 * i + 1]
        state_indices = list(state_indices)

        target_letter = state_indices[target_index]
        state_indices[target_index + num_indices - i] = target_letter
        state_indices = "".join(state_indices)

        einsum_indices = f"a{state_indices}"
        density_matrix = einsum(einsum_indices, density_matrix)

    number_wires_sub = num_indices - len(indices)
    reduced_density_matrix = np.reshape(
        density_matrix, (batch_dim, qudit_dim**number_wires_sub, qudit_dim**number_wires_sub)
    )
    return reduced_density_matrix


def _batched_partial_trace_nonrep_indices(density_matrix, indices, qudit_dim=3):
    """Compute the reduced density matrix for autograd interface by tracing out the provided indices with the use
    of projectors as same subscripts indices are not supported in autograd backprop.
    """
    # Dimension and reshape
    batch_dim, dim = density_matrix.shape[:2]
    num_indices = int(np.log2(dim)/np.log2(qudit_dim))
    rho_dim = 2 * num_indices
    density_matrix = np.reshape(density_matrix, [batch_dim] + [qudit_dim] * 2 * num_indices)

    kraus = cast(np.eye(qudit_dim), density_matrix.dtype)

    kraus = np.reshape(kraus, (qudit_dim, 1, qudit_dim))
    kraus_dagger = np.asarray([np.conj(np.transpose(k)) for k in kraus])

    kraus = convert_like(kraus, density_matrix)
    kraus_dagger = convert_like(kraus_dagger, density_matrix)
    # For loop over wires
    for target_wire in indices:
        # Tensor indices of density matrix
        state_indices = ABC[1 : rho_dim + 1]
        # row indices of the quantum state affected by this operation
        row_wires_list = [target_wire + 1]
        row_indices = "".join(ABC_ARRAY[row_wires_list].tolist())
        # column indices are shifted by the number of wires
        col_wires_list = [w + num_indices for w in row_wires_list]
        col_indices = "".join(ABC_ARRAY[col_wires_list].tolist())
        # indices in einsum must be replaced with new ones
        num_partial_trace_wires = 1
        new_row_indices = ABC[rho_dim + 1 : rho_dim + num_partial_trace_wires + 1]
        new_col_indices = ABC[
            rho_dim + num_partial_trace_wires + 1 : rho_dim + 2 * num_partial_trace_wires + 1
        ]
        # index for summation over Kraus operators
        kraus_index = ABC[
            rho_dim + 2 * num_partial_trace_wires + 1 : rho_dim + 2 * num_partial_trace_wires + 2
        ]
        # new state indices replace row and column indices with new ones
        new_state_indices = functools.reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
            zip(col_indices + row_indices, new_col_indices + new_row_indices),
            state_indices,
        )
        # index mapping for einsum, e.g., 'iga,abcdef,idh->gbchef'
        einsum_indices = (
            f"{kraus_index}{new_row_indices}{row_indices}, a{state_indices},"
            f"{kraus_index}{col_indices}{new_col_indices}->a{new_state_indices}"
        )
        density_matrix = einsum(einsum_indices, kraus, density_matrix, kraus_dagger)

    number_wires_sub = num_indices - len(indices)
    reduced_density_matrix = np.reshape(
        density_matrix, (batch_dim, qudit_dim**number_wires_sub, qudit_dim**number_wires_sub)
    )
    return reduced_density_matrix
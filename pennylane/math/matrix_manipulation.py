# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains methods to expand the matrix representation of an operator
to a higher hilbert space with re-ordered wires."""
import copy
import itertools
from functools import reduce
from typing import Generator, Tuple

import numpy as np
from scipy.sparse import csr_matrix, eye, issparse, kron

import pennylane as qml
from pennylane.wires import Wires


def expand_matrix(base_matrix, wires, wire_order=None, sparse_format="csr"):
    # pylint: disable=too-many-branches
    """Re-express a base matrix acting on a subspace defined by a set of wire labels
    according to a global wire order.

    Args:
        base_matrix (tensor_like): base matrix to expand
        wires (Iterable): wires determining the subspace that base matrix acts on; a base matrix of
            dimension :math:`2^n` acts on a subspace of :math:`n` wires
        wire_order (Iterable): global wire order, which has to contain all wire labels in ``wires``, but can also
            contain additional labels
        sparse_format (str): if the base matrix is a scipy sparse matrix then this is the string representing the
            preferred scipy sparse matrix format to cast the expanded matrix too

    Returns:
        tensor_like: expanded matrix

    **Example**

    If the wire order is ``None`` or identical to ``wires``, the original matrix gets returned:

    >>> base_matrix = np.array([[1, 2, 3, 4],
    ...                         [5, 6, 7, 8],
    ...                         [9, 10, 11, 12],
    ...                         [13, 14, 15, 16]])
    >>> print(expand_matrix(base_matrix, wires=[0, 2], wire_order=[0, 2]))
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]
     [13 14 15 16]]
    >>> print(expand_matrix(base_matrix, wires=[0, 2]))
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]
     [13 14 15 16]]

    If the wire order is a permutation of ``wires``, the entries of the base matrix get permuted:

    >>> print(expand_matrix(base_matrix, wires=[0, 2], wire_order=[2, 0]))
    [[ 1  3  2  4]
     [ 9 11 10 12]
     [ 5  7  6  8]
     [13 15 14 16]]

    If the wire order contains wire labels not found in ``wires``, the matrix gets expanded:

    >>> print(expand_matrix(base_matrix, wires=[0, 2], wire_order=[0, 1, 2]))
    [[ 1  2  0  0  3  4  0  0]
     [ 5  6  0  0  7  8  0  0]
     [ 0  0  1  2  0  0  3  4]
     [ 0  0  5  6  0  0  7  8]
     [ 9 10  0  0 11 12  0  0]
     [13 14  0  0 15 16  0  0]
     [ 0  0  9 10  0  0 11 12]
     [ 0  0 13 14  0  0 15 16]]

    The method works with tensors from all autodifferentiation frameworks, for example:

    >>> base_matrix_torch = torch.tensor([[1., 2.],
    ...                                   [3., 4.]], requires_grad=True)
    >>> res = expand_matrix(base_matrix_torch, wires=["b"], wire_order=["a", "b"])
    >>> type(res)
    torch.Tensor
    >>> res.requires_grad
    True

    The method words with scipy sparse matrices, for example:

    >>> from scipy import sparse
    >>> mat = sparse.csr_matrix([[0, 1], [1, 0]])
    >>> qml.math.expand_matrix(mat, wires=[1], wire_order=[0,1]).toarray()
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.]])

    """

    if (wire_order is None) or (wire_order == wires):
        return base_matrix

    interface = qml.math.get_interface(base_matrix)  # pylint: disable=protected-access

    def eye_interface(dim):
        return qml.math.cast_like(qml.math.eye(2**dim, like=interface), base_matrix)

    if interface == "scipy" and issparse(base_matrix):
        return _sparse_expand_matrix(base_matrix, wires, wire_order, format=sparse_format)

    wire_order = list(wire_order)

    shape = qml.math.shape(base_matrix)
    batch_dim = shape[0] if len(shape) == 3 else None

    # get a subset of `wire_order` values that contain all wire labels inside `wires` argument
    # e.g. wire_order = [0, 1, 2, 3, 4]; wires = [3, 0, 2]
    # --> subset_wire_order = [0, 1, 2, 3]; expanded_wires = [3, 0, 2, 1]
    wire_indices = [wire_order.index(wire) for wire in wires]
    subset_wire_order = wire_order[min(wire_indices) : max(wire_indices) + 1]
    wire_difference = list(set(subset_wire_order) - set(wires))
    expanded_wires = wires + wire_difference

    # expand matrix if needed
    if wire_difference:
        if batch_dim is not None:
            batch_matrices = [
                qml.math.kron(batch, eye_interface(len(wire_difference)), like=interface)
                for batch in base_matrix
            ]
            base_matrix = qml.math.stack(batch_matrices, like=interface)
        else:
            base_matrix = qml.math.kron(
                base_matrix, eye_interface(len(wire_difference)), like=interface
            )

    num_wires = len(subset_wire_order)

    # reshape matrix to match wire values e.g. mat[0, 0, 0, 0] = <0000|mat|0000>
    # with this reshape we can easily swap wires
    shape = [batch_dim] + [2] * (num_wires * 2) if batch_dim else [2] * (num_wires * 2)
    mat = qml.math.reshape(base_matrix, shape)

    # compute the permutations needed to match wire order
    perm = [expanded_wires.index(wire) for wire in subset_wire_order]
    perm += [p + num_wires for p in perm]
    if batch_dim:
        perm = [0] + [p + 1 for p in perm]

    if perm != list(range(num_wires * 2)):
        mat = qml.math.transpose(mat, axes=perm)

    # reshape back
    shape = [batch_dim] + [2**num_wires] * 2 if batch_dim else [2**num_wires] * 2
    mat = qml.math.reshape(mat, shape)

    # expand the matrix even further by adding the missing wires
    if len(wire_order) > len(expanded_wires):
        mats = []  # list of tuples with (Id,) and (base_matrix,) or (mat1, mat2, ...) if batch_dim
        op_wires_in_list = False
        identity_count = 0
        for wire in wire_order:
            if wire not in expanded_wires:
                identity_count += 1
            elif not op_wires_in_list:
                if identity_count > 0:
                    mats.append((eye_interface(identity_count),))
                    identity_count = 0
                mats.append(tuple(mat) if batch_dim else (mat,))
                op_wires_in_list = True

        if identity_count > 0:
            mats.append((eye_interface(identity_count),))

        # itertools.product will create a tuple of matrices for each different batch
        mats_list = list(itertools.product(*mats))
        # here we compute the kron product of each different tuple and stack them back together
        expanded_batch_matrices = [
            reduce(
                lambda i, j: qml.math.kron(
                    i.contiguous() if interface == "torch" else i,
                    j.contiguous() if interface == "torch" else j,
                    like=interface,
                ),
                mats,
            )
            for mats in mats_list
        ]
        mat = (
            qml.math.stack(expanded_batch_matrices, like=interface)
            if len(expanded_batch_matrices) > 1
            else expanded_batch_matrices[0]
        )

    return mat


def reduce_matrices(
    mats_and_wires_gen: Generator[Tuple[np.ndarray, Wires], None, None], reduce_func: callable
) -> Tuple[np.ndarray, Wires]:
    """Apply the given ``reduce_func`` cumulatively to the items of the ``mats_and_wires_gen``
    generator, from left to right, so as to reduce the sequence to a tuple containing a single
    matrix and the wires it acts on.

    Args:
        mats_and_wires_gen (Generator): generator of tuples containing the matrix and the wires of
            each operator
        reduce_func (callable): function used to reduce the sequence of operators

    Returns:
        Tuple[tensor, Wires]: a tuple containing the reduced matrix and the wires it acts on
    """

    def expand_and_reduce(op1_tuple: Tuple[np.ndarray, Wires], op2_tuple: Tuple[np.ndarray, Wires]):
        mat1, wires1 = op1_tuple
        mat2, wires2 = op2_tuple
        expanded_wires = wires1 + wires2
        mat1 = qml.math.expand_matrix(mat1, wires1, wire_order=expanded_wires)
        mat2 = qml.math.expand_matrix(mat2, wires2, wire_order=expanded_wires)
        return reduce_func(mat1, mat2), expanded_wires

    reduced_mat, final_wires = reduce(expand_and_reduce, mats_and_wires_gen)

    return reduced_mat, final_wires


def _local_sparse_swap_mat(i, n, format="csr"):
    """Helper function which generates the sparse matrix of SWAP
    for qubits: i <--> i+1 with final shape (2**n, 2**n)."""
    assert i < n - 1
    swap_mat = csr_matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    j = i + 1  # i is the index of the qubit, j is the number of qubits prior to and include qubit i
    return kron(
        kron(eye(2 ** (j - 1)), swap_mat), eye(2 ** (n - (j + 1))), format=format
    )  # (j - 1) + 2 + (n - (j+1)) = n


def _sparse_swap_mat(i, j, n, format="csr"):
    """Helper function which generates the sparse matrix of SWAP
    for qubits: i <--> j with final shape (2**n, 2**n)."""
    assert i < n and j < n
    if i == j:
        return eye(2**n, format=format)

    (small_i, big_j) = (i, j) if i < j else (j, i)
    store_swaps = [
        _local_sparse_swap_mat(index, n, format=format) for index in range(small_i, big_j)
    ]

    res = eye(2**n, format=format)
    for mat in store_swaps:  # swap i --> j
        res @= mat

    for mat in store_swaps[-2::-1]:  # bring j --> old_i
        res @= mat

    return res


def _sparse_expand_matrix(base_matrix, wires, wire_order, format="csr"):
    """Re-express a sparse base matrix acting on a subspace defined by a set of wire labels
    according to a global wire order.

    Args:
        base_matrix (scipy.sparse.spmatrix): base matrix to expand
        wires (Iterable): wires determining the subspace that base matrix acts on; a base matrix of
            dimension :math:`2^n` acts on a subspace of :math:`n` wires
        wire_order (Iterable): global wire order, which has to contain all wire labels in ``wires``, but can also
            contain additional labels
        format (str): string representing the preferred scipy sparse matrix format to cast the expanded
            matrix too

    Returns:
        tensor_like: expanded matrix
    """
    n_wires = len(wires)
    n_total_wires = len(wire_order)

    if isinstance(wires, qml.wires.Wires):
        expanded_wires = wires.tolist()
    else:
        expanded_wires = list(copy.copy(wires))

    for wire in wire_order:
        if wire not in wires:
            expanded_wires.append(wire)

    num_missing_wires = n_total_wires - n_wires
    if num_missing_wires > 0:
        expanded_matrix = kron(
            base_matrix, eye(2**num_missing_wires, format=format), format=format
        )  # added missing wires at the end
    else:
        expanded_matrix = copy.copy(base_matrix)

    U = eye(2**n_total_wires, format=format)
    for i in range(n_total_wires):
        if expanded_wires[i] != wire_order[i]:
            j = expanded_wires.index(wire_order[i])  # location of correct wire
            U = U @ _sparse_swap_mat(
                i, j, n_total_wires, format=format
            )  # swap incorrect wire for correct wire

            expanded_wires[i], expanded_wires[j] = expanded_wires[j], expanded_wires[i]

    expanded_matrix = U.T @ expanded_matrix @ U
    return expanded_matrix.asformat(format)

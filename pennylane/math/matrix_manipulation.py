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
from functools import reduce
from typing import Generator, Tuple

import numpy as np
from scipy.sparse import coo_matrix, eye, issparse, kron

import pennylane as qml
from pennylane.wires import Wires


def expand_matrix(base_matrix, wires, wire_order=None, sparse_format="csr"):
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
    if interface == "scipy" and issparse(base_matrix):
        return _sparse_expand_matrix(base_matrix, wires, wire_order, format=sparse_format)

    wire_order = qml.wires.Wires(wire_order)
    n = len(wires)
    shape = qml.math.shape(base_matrix)
    batch_dim = shape[0] if len(shape) == 3 else None

    # operator's wire positions relative to wire ordering
    op_wire_pos = wire_order.indices(wires)

    identity = qml.math.reshape(
        qml.math.eye(2 ** len(wire_order), like=interface), [2] * (len(wire_order) * 2)
    )
    # The first axis entries are range(n, 2n) for batch_dim=None and range(n+1, 2n+1) else
    axes = (list(range(-n, 0)), op_wire_pos)

    # reshape op.matrix()
    op_matrix_interface = qml.math.convert_like(base_matrix, identity)
    shape = [batch_dim] + [2] * (n * 2) if batch_dim else [2] * (n * 2)
    mat_op_reshaped = qml.math.reshape(op_matrix_interface, shape)
    mat_tensordot = qml.math.tensordot(
        mat_op_reshaped, qml.math.cast_like(identity, mat_op_reshaped), axes
    )

    unused_idxs = [idx for idx in range(len(wire_order)) if idx not in op_wire_pos]
    # permute matrix axes to match wire ordering
    perm = op_wire_pos + unused_idxs
    sources = wire_order.indices(wire_order)
    if batch_dim:
        perm = [p + 1 for p in perm]
        sources = [s + 1 for s in sources]

    mat = qml.math.moveaxis(mat_tensordot, sources, perm)
    shape = [batch_dim] + [2 ** len(wire_order)] * 2 if batch_dim else [2 ** len(wire_order)] * 2
    mat = qml.math.reshape(mat, shape)

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
        mat1 = expand_matrix(mat1, wires1, wire_order=expanded_wires)
        mat2 = expand_matrix(mat2, wires2, wire_order=expanded_wires)
        return reduce_func(mat1, mat2), expanded_wires

    reduced_mat, final_wires = reduce(expand_and_reduce, mats_and_wires_gen)

    return reduced_mat, final_wires


def _sparse_swap_mat(qubit_i, qubit_j, n):
    """Helper function which generates the sparse matrix of SWAP
    for qubits: i <--> j with final shape (2**n, 2**n)."""

    def swap_qubits(index, i, j):
        s = list(format(index, f"0{n}b"))  # convert to binary
        si, sj = s[i], s[j]
        if si == sj:
            return index
        s[i], s[j] = sj, si  # swap qubits
        return int(f"0b{''.join(s)}", 2)  # convert to int

    data = [1] * (2**n)
    index_i = list(range(2**n))  # bras (we don't change anything)
    index_j = [
        swap_qubits(idx, qubit_i, qubit_j) for idx in index_i
    ]  # kets (we swap qubits i and j): <10| --> <01|
    return coo_matrix((data, (index_i, index_j)))


def _sparse_expand_matrix(base_matrix, wires, wire_order, format="coo"):
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
    base_matrix.eliminate_zeros()

    n_total_wires = len(wire_order)

    wires = wires.tolist() if isinstance(wires, qml.wires.Wires) else list(copy.copy(wires))

    expanded_wires = []
    mats = []
    op_wires_in_list = False
    i_count = 0
    for wire in wire_order:
        if wire not in wires:
            i_count += 1
            expanded_wires.append(wire)
        elif not op_wires_in_list:
            if i_count > 0:
                mats.append(eye(2**i_count, format="coo"))
            i_count = 0
            mats.append(base_matrix)
            op_wires_in_list = True
            expanded_wires.extend(wires)

    if i_count > 0:
        mats.append(eye(2**i_count, format="coo"))

    if len(mats) > 1:
        expanded_matrix = reduce(lambda i, j: kron(i, j, format="coo"), mats)
    else:
        expanded_matrix = copy.copy(base_matrix)

    expanded_matrix.eliminate_zeros()

    U = None
    for i in range(n_total_wires):
        if expanded_wires[i] != wire_order[i]:
            if U is None:
                U = eye(2**n_total_wires, format=format)
            j = expanded_wires.index(wire_order[i])  # location of correct wire
            U = U @ _sparse_swap_mat(i, j, n_total_wires)  # swap incorrect wire for correct wire
            U.eliminate_zeros()

            expanded_wires[i], expanded_wires[j] = expanded_wires[j], expanded_wires[i]
    if U is not None:
        expanded_matrix = U.T @ expanded_matrix @ U
        expanded_matrix.eliminate_zeros()
    return expanded_matrix.asformat(format)

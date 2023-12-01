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
import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple

import numpy as np
from scipy.sparse import csr_matrix, eye, kron

import pennylane as qml
from pennylane.wires import Wires


def expand_matrix(mat, wires, wire_order=None, sparse_format="csr"):
    # pylint: disable=too-many-branches
    """Re-express a matrix acting on a subspace defined by a set of wire labels
    according to a global wire order.

    Args:
        mat (tensor_like): matrix to expand
        wires (Iterable): wires determining the subspace that ``mat`` acts on; a matrix of
            dimension :math:`2^n` acts on a subspace of :math:`n` wires
        wire_order (Iterable): global wire order, which has to contain all wire labels in ``wires``, but can also
            contain additional labels
        sparse_format (str): if ``mat`` is a SciPy sparse matrix then this is the string representing the
            preferred scipy sparse matrix format to cast the expanded matrix too

    Returns:
        tensor_like: expanded matrix

    **Example**

    If the wire order is ``None`` or identical to ``wires``, the original matrix gets returned:

    >>> matrix = np.array([[1, 2, 3, 4],
    ...                    [5, 6, 7, 8],
    ...                    [9, 10, 11, 12],
    ...                    [13, 14, 15, 16]])
    >>> print(expand_matrix(matrix, wires=[0, 2], wire_order=[0, 2]))
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]
     [13 14 15 16]]
    >>> print(expand_matrix(matrix, wires=[0, 2]))
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]
     [13 14 15 16]]

    If the wire order is a permutation of ``wires``, the entries of the matrix get permuted:

    >>> print(expand_matrix(matrix, wires=[0, 2], wire_order=[2, 0]))
    [[ 1  3  2  4]
     [ 9 11 10 12]
     [ 5  7  6  8]
     [13 15 14 16]]

    If the wire order contains wire labels not found in ``wires``, the matrix gets expanded:

    >>> print(expand_matrix(matrix, wires=[0, 2], wire_order=[0, 1, 2]))
    [[ 1  2  0  0  3  4  0  0]
     [ 5  6  0  0  7  8  0  0]
     [ 0  0  1  2  0  0  3  4]
     [ 0  0  5  6  0  0  7  8]
     [ 9 10  0  0 11 12  0  0]
     [13 14  0  0 15 16  0  0]
     [ 0  0  9 10  0  0 11 12]
     [ 0  0 13 14  0  0 15 16]]

    The method works with tensors from all autodifferentiation frameworks, for example:

    >>> matrix_torch = torch.tensor([[1., 2.],
    ...                              [3., 4.]], requires_grad=True)
    >>> res = expand_matrix(matrix_torch, wires=["b"], wire_order=["a", "b"])
    >>> type(res)
    torch.Tensor
    >>> res.requires_grad
    True

    The method works with scipy sparse matrices, for example:

    >>> from scipy import sparse
    >>> mat = sparse.csr_matrix([[0, 1], [1, 0]])
    >>> qml.math.expand_matrix(mat, wires=[1], wire_order=[0,1]).toarray()
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.]])

    """

    if (wire_order is None) or (wire_order == wires):
        return mat

    wires = list(wires)
    wire_order = list(wire_order)

    interface = qml.math.get_interface(mat)
    shape = qml.math.shape(mat)
    batch_dim = shape[0] if len(shape) == 3 else None

    def eye_interface(dim):
        if interface == "scipy":
            return eye(2**dim, format="coo")
        return qml.math.cast_like(qml.math.eye(2**dim, like=interface), mat)

    def kron_interface(mat1, mat2):
        if interface == "scipy":
            res = kron(mat1, mat2, format="coo")
            res.eliminate_zeros()
            return res
        if interface == "torch":
            # these lines are to avoid a crash when the matrices are not contiguous in memory
            mat1 = mat1.contiguous()
            mat2 = mat2.contiguous()
        return qml.math.kron(mat1, mat2, like=interface)

    # get a subset of `wire_order` values that contain all wire labels inside `wires` argument
    # e.g. wire_order = [0, 1, 2, 3, 4]; wires = [3, 0, 2]
    # --> subset_wire_order = [0, 1, 2, 3]; expanded_wires = [3, 0, 2, 1]
    wire_indices = [wire_order.index(wire) for wire in wires]
    subset_wire_order = wire_order[min(wire_indices) : max(wire_indices) + 1]
    wire_difference = list(set(subset_wire_order) - set(wires))
    expanded_wires = wires + wire_difference

    # expand the matrix if the wire subset is larger than the matrix wires
    if wire_difference:
        if batch_dim is not None:
            batch_matrices = [
                kron_interface(batch, eye_interface(len(wire_difference))) for batch in mat
            ]
            mat = qml.math.stack(batch_matrices, like=interface)
        else:
            mat = kron_interface(mat, eye_interface(len(wire_difference)))

    # permute matrix
    if interface == "scipy":
        mat = _permute_sparse_matrix(mat, expanded_wires, subset_wire_order)
    else:
        mat = _permute_dense_matrix(mat, expanded_wires, subset_wire_order, batch_dim)

    # expand the matrix even further if needed
    if len(expanded_wires) < len(wire_order):
        mats = []
        num_pre_identities = min(wire_indices)
        if num_pre_identities > 0:
            mats.append((eye_interface(num_pre_identities),))

        mats.append(tuple(mat) if batch_dim else (mat,))

        num_post_identities = len(wire_order) - max(wire_indices) - 1
        if num_post_identities > 0:
            mats.append((eye_interface(num_post_identities),))

        # itertools.product will create a tuple of matrices for each different batch
        mats_list = list(itertools.product(*mats))
        # here we compute the kron product of each different tuple and stack them back together
        expanded_batch_matrices = [reduce(kron_interface, mats) for mats in mats_list]
        mat = (
            qml.math.stack(expanded_batch_matrices, like=interface)
            if len(expanded_batch_matrices) > 1
            else expanded_batch_matrices[0]
        )
    return mat.asformat(sparse_format) if interface == "scipy" else mat


def _permute_sparse_matrix(matrix, wires, wire_order):
    """Permute the matrix to match the wires given in `wire_order`.

    Args:
        matrix (scipy.sparse.spmatrix): matrix to permute
        wires (list): wires determining the subspace that base matrix acts on; a base matrix of
            dimension :math:`2^n` acts on a subspace of :math:`n` wires
        wire_order (list): global wire order, which has to contain all wire labels in ``wires``,
            but can also contain additional labels

    Returns:
        scipy.sparse.spmatrix: permuted matrix
    """
    U = _permutation_sparse_matrix(wires, wire_order)
    if U is not None:
        matrix = U.T @ matrix @ U
        matrix.eliminate_zeros()
    return matrix


def _permute_dense_matrix(matrix, wires, wire_order, batch_dim):
    """Permute the matrix to match the wires given in `wire_order`.

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
    shape = [batch_dim] + [2] * (num_wires * 2) if batch_dim else [2] * (num_wires * 2)
    matrix = qml.math.reshape(matrix, shape)
    # transpose matrix
    matrix = qml.math.transpose(matrix, axes=perm)
    # reshape back
    shape = [batch_dim] + [2**num_wires] * 2 if batch_dim else [2**num_wires] * 2
    return qml.math.reshape(matrix, shape)


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
    ]  # kets (we swap qubits i and j): |10> --> |01>
    return csr_matrix((data, (index_i, index_j)))


def _permutation_sparse_matrix(expanded_wires: Iterable, wire_order: Iterable) -> csr_matrix:
    """Helper function which generates a permutation matrix in sparse format that swaps the wires
    in ``expanded_wires`` to match the order given by the ``wire_order`` argument.

    Args:
        expanded_wires (Iterable): inital wires
        wire_order (Iterable): final wires

    Returns:
        csr_matrix: permutation matrix in CSR sparse format
    """
    n_total_wires = len(wire_order)
    U = None
    for i in range(n_total_wires):
        if expanded_wires[i] != wire_order[i]:
            if U is None:
                U = eye(2**n_total_wires, format="csr")
            j = expanded_wires.index(wire_order[i])  # location of correct wire
            U = U @ _sparse_swap_mat(i, j, n_total_wires)  # swap incorrect wire for correct wire
            U.eliminate_zeros()

            expanded_wires[i], expanded_wires[j] = expanded_wires[j], expanded_wires[i]

    return U


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


def get_batch_size(tensor, expected_shape, expected_size):
    """
    Determine whether a tensor has an additional batch dimension for broadcasting,
    compared to an expected_shape. Has support for abstract TF tensors.

    Args:
        tensor (TensorLike): A tensor to inspect for batching
        expected_shape (Tuple[int]): The expected shape of the tensor if not batched
        expected_size (int): The expected size of the tensor if not batched

    Returns:
        Optional[int]: The batch size of the tensor if there is one, otherwise None
    """
    try:
        size = qml.math.size(tensor)
        ndim = qml.math.ndim(tensor)
        if ndim > len(expected_shape) or size > expected_size:
            return size // expected_size

    except Exception as err:  # pragma: no cover, pylint:disable=broad-except
        # This except clause covers the usage of tf.function
        if not qml.math.is_abstract(tensor):
            raise err

    return None

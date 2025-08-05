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

"""This module contains methods that manipulates matrices."""

import itertools
import numbers
from collections.abc import Callable, Iterable, Sequence
from functools import reduce

import numpy as np
from scipy.sparse import csr_matrix, eye, kron

from pennylane import math


# pylint: disable=too-many-branches
def expand_matrix(mat, wires: Sequence | int, wire_order=None, sparse_format="csr"):
    """Re-express a matrix acting on a subspace defined by a set of wire labels
    according to a global wire order.

    Args:
        mat (tensor_like): matrix to expand
        wires (Sequence): wires determining the subspace that ``mat`` acts on; a matrix of
            dimension :math:`D^n` acts on a subspace of :math:`n` wires, where :math:`D` is the qudit dimension (2).
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
    if isinstance(wires, int):
        wires = [wires]

    if wires:
        float_dim = math.shape(mat)[-1] ** (1 / (len(wires)))
        qudit_dim = int(math.round(float_dim))
    else:
        qudit_dim = 2  # if no wires, just assume qubit

    if (wire_order is None) or (wire_order == wires):
        return mat

    if not wires and math.shape(mat) == (1, 1):
        return complex(mat[0, 0])

    wires = list(wires)
    wire_order = list(wire_order)

    interface = math.get_interface(mat)
    shape = math.shape(mat)
    batch_dim = shape[0] if len(shape) == 3 else None

    def eye_interface(dim):
        if interface == "scipy":
            return eye(qudit_dim**dim, format="coo")
        return math.cast_like(math.eye(qudit_dim**dim, like=interface), mat)

    def kron_interface(mat1, mat2):
        if interface == "scipy":
            res = kron(mat1, mat2, format="coo")
            res.eliminate_zeros()
            return res
        if interface == "torch":
            # these lines are to avoid a crash when the matrices are not contiguous in memory
            mat1 = mat1.contiguous()
            mat2 = mat2.contiguous()
        return math.kron(mat1, mat2, like=interface)

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
            mat = math.stack(batch_matrices, like=interface)
        else:
            mat = kron_interface(mat, eye_interface(len(wire_difference)))

    # permute matrix
    if interface == "scipy":
        mat = _permute_sparse_matrix(mat, expanded_wires, subset_wire_order)
    else:
        mat = _permute_dense_matrix(
            mat, expanded_wires, subset_wire_order, batch_dim, qudit_dim=qudit_dim
        )

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
            math.stack(expanded_batch_matrices, like=interface)
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


def _permute_dense_matrix(matrix, wires, wire_order, batch_dim, qudit_dim: int = 2):
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
    shape = (
        [batch_dim] + [qudit_dim] * (num_wires * 2) if batch_dim else [qudit_dim] * (num_wires * 2)
    )
    matrix = math.reshape(matrix, shape)
    # transpose matrix
    matrix = math.transpose(matrix, axes=perm)
    # reshape back
    shape = [batch_dim] + [qudit_dim**num_wires] * 2 if batch_dim else [qudit_dim**num_wires] * 2
    return math.reshape(matrix, shape)


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


def _permutation_sparse_matrix(expanded_wires: Sequence, wire_order: Sequence) -> csr_matrix:
    """Helper function which generates a permutation matrix in sparse format that swaps the wires
    in ``expanded_wires`` to match the order given by the ``wire_order`` argument.

    Args:
        expanded_wires (Sequence): inital wires
        wire_order (Sequence): final wires

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
    mats_and_wires_gen: Iterable[tuple[np.ndarray, Sequence]], reduce_func: Callable
) -> tuple[np.ndarray, Sequence]:
    """Apply the given ``reduce_func`` cumulatively to the items of the ``mats_and_wires_gen``
    generator, from left to right, reducing the sequence to a tuple containing a single
    matrix and the wires it acts on.

    Args:
        mats_and_wires_gen (Iterable): tuples containing the matrix and the wires of each operator
        reduce_func (callable): function used to reduce the sequence of operators

    Returns:
        Tuple[tensor, Sequence]: a tuple containing the reduced matrix and the wires it acts on
    """

    def expand_and_reduce(
        op1_tuple: tuple[np.ndarray, Sequence], op2_tuple: tuple[np.ndarray, Sequence]
    ):
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
        size = math.size(tensor)
        ndim = math.ndim(tensor)
        if ndim > len(expected_shape) or size > expected_size:
            return size // expected_size

    except Exception as err:  # pragma: no cover, pylint:disable=broad-except
        # This except clause covers the usage of tf.function
        if not math.is_abstract(tensor):
            raise err

    return None


def expand_vector(vector, original_wires, expanded_wires):
    r"""Expand a vector to more wires.

    Args:
        vector (array): :math:`2^n` vector where n = len(original_wires).
        original_wires (Sequence[int]): original wires of vector
        expanded_wires (Union[Sequence[int], int]): expanded wires of vector, can be shuffled
            If a single int m is given, corresponds to list(range(m))

    Returns:
        array: :math:`2^m` vector where m = len(expanded_wires).
    """
    if len(original_wires) == 0:
        val = math.squeeze(vector)
        return val * math.ones(2 ** len(expanded_wires))
    if isinstance(expanded_wires, numbers.Integral):
        expanded_wires = list(range(expanded_wires))

    N = len(original_wires)
    M = len(expanded_wires)
    D = M - N

    len_vector = math.shape(vector)[0]
    qudit_order = int(2 ** (np.log2(len_vector) / N))

    if not set(expanded_wires).issuperset(original_wires):
        raise ValueError("Invalid target subsystems provided in 'original_wires' argument.")

    if math.shape(vector) != (qudit_order**N,):
        raise ValueError(f"Vector parameter must be of length {qudit_order}**len(original_wires)")

    dims = [qudit_order] * N
    tensor = math.reshape(vector, dims)

    if D > 0:
        extra_dims = [qudit_order] * D
        ones = math.ones(qudit_order**D).reshape(extra_dims)
        expanded_tensor = math.tensordot(tensor, ones, axes=0)
    else:
        expanded_tensor = tensor

    wire_indices = [expanded_wires.index(wire) for wire in original_wires]
    wire_indices = np.array(wire_indices)

    # Order tensor factors according to wires
    original_indices = np.array(range(N))
    expanded_tensor = math.moveaxis(expanded_tensor, tuple(original_indices), tuple(wire_indices))

    return math.reshape(expanded_tensor, (qudit_order**M,))


def convert_to_su2(U, return_global_phase=False):
    r"""Convert a 2x2 unitary matrix to :math:`SU(2)`. (batched operation)

    Args:
        U (array[complex]): A matrix with a batch dimension, presumed to be
            of shape :math:`n \times 2 \times 2` and unitary for any positive integer n.
        return_global_phase (bool): If `True`, the return will include the global phase.
            If `False`, only the :math:`SU(2)` representation is returned.

    Returns:
        array[complex]:
            A :math:`n \times 2 \times 2` matrix in :math:`SU(2)` that is equivalent to U up to a
            global phase. If ``return_global_phase=True``, a 2-element tuple is returned, with
            the first element being the :math:`SU(2)` equivalent and the second, the global phase.

    """
    # Compute the determinant
    U = math.cast(U, "complex128")
    batch_size = get_batch_size(U, (2, 2), 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        determinant = math.linalg.det(U)
    global_phase = math.angle(determinant) / 2
    U = math.cast_like(U, determinant)
    if batch_size:
        c_phase = math.cast_like(global_phase, 1j)
        batched_phase = math.reshape(c_phase, (batch_size, 1, 1))
        U = U * math.exp(-1j * batched_phase)
    else:
        c_phase = math.cast_like(global_phase, 1j)
        U = U * math.exp(-1j * c_phase)
    return (U, global_phase) if return_global_phase else U


def convert_to_su4(U, return_global_phase=False):
    r"""Convert a 4x4 matrix to :math:`SU(4)`.

    Args:
        U (array[complex]): A matrix, presumed to be :math:`4 \times 4` and unitary.
        return_global_phase (bool): If `True`, the return will include the global phase.
            If `False`, only the :math:`SU(4)` representation is returned.

    Returns:
        array[complex]:
            A :math:`4 \times 4` matrix in :math:`SU(4)` that is equivalent to U up to a global
            phase. If ``return_global_phase=True``, a 2-element tuple is returned, with the first
            element being the :math:`SU(4)` equivalent and the second, the global phase.

    """
    # Compute the determinant
    U = math.cast(U, "complex128")
    batch_size = get_batch_size(U, (4, 4), 16)
    with np.errstate(divide="ignore", invalid="ignore"):
        determinant = math.linalg.det(U)
    global_phase = math.angle(determinant) / 4
    if batch_size:
        c_phase = math.cast_like(global_phase, 1j)
        batched_phase = math.reshape(c_phase, (batch_size, 1, 1))
        U = U * math.exp(-1j * batched_phase)
    else:
        c_phase = math.cast_like(global_phase, 1j)
        U = U * math.exp(-1j * c_phase)
    return (U, global_phase) if return_global_phase else U

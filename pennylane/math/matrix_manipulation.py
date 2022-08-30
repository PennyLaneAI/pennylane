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
import pennylane as qml


def expand_matrix(base_matrix, wires, wire_order=None):
    """Re-express a base matrix acting on a subspace defined by a set of wire labels
    according to a global wire order.

    Args:
        base_matrix (tensor_like): base matrix to expand
        wires (Iterable): wires determining the subspace that base matrix acts on; a base matrix of
            dimension :math:`2^n` acts on a subspace of :math:`n` wires
        wire_order (Iterable): global wire order, which has to contain all wire labels in ``wires``, but can also
            contain additional labels

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

    """

    if (wire_order is None) or (wire_order == wires):
        return base_matrix

    wire_order = qml.wires.Wires(wire_order)
    n = len(wires)
    shape = qml.math.shape(base_matrix)
    batch_dim = shape[0] if len(shape) == 3 else None
    interface = qml.math.get_interface(base_matrix)  # pylint: disable=protected-access

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

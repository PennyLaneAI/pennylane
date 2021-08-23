# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Class for framework-agnostic sparse matrix representation."""
from copy import copy
import numpy as np
import pennylane as qml


class SparseMatrix:
    """Framework-agnostic class to represent a sparse matrix.

    The sparse matrix is stored as a dictionary of a tuple of indices as keys, and
    the corresponding value as entries:

    >>> s = SparseMatrix(tf.Variable([[3., 0., 0.],[2., 0., 0.]]))
    >>> print(s.data)
    {(0, 0): <tf.Tensor: shape=(), dtype=float32, numpy=3.0>, (1, 0): <tf.Tensor: shape=(), dtype=float32, numpy=2.0>}
    >>> print(s.shape)
    (2, 3)

    An empty sparse matrix can be initialised by passing a shape tuple:

    >>> s = SparseMatrix((12, 12))
    >>> print(s.data)
    None
    >>> print(s.shape)
    (12, 12)

    Args:
        arg (tuple or tensor_like): if tuple, this is interpreted as the shape of
            the sparse matrix; else it is interpreted as a 2-d tensor which is a dense representation
            of the matrix.

    **Example:**

    To mimic ``scipy.sparse`` objects, sparse matrices give acces to lists of row and column indices,
    as well as the number of non-zero elements:

    >>> s = SparseMatrix(tf.Variable([[3., 0.],[2., 0.]]))
    >>> print(s.row)
    [0, 1]
    >>> print(s.col)
    [0, 0]
    >>> print(s.nnz)
    2

    They support elementary arithmetic such as addition, subtraction and scalar multiplication from both sides:

    >>> s1 = SparseMatrix(torch.tensor([[3., 0.],[2., 0.]]))
    >>> s2 = SparseMatrix(torch.tensor([[-3., 0.],[0., 0.]]))
    >>> res = s1 + s2
    >>> print(res.data)
    {(1, 0): tensor(2.)}

    >>> res2 = s1 - s2
    >>> print(res2.data)
    {(0, 0): tensor(6.), (1, 0): tensor(2.)}

    >>> res3 = 2. * s1
    >>> print(res3.data)
    {(0, 0): tensor(6.), (1, 0): tensor(4.)}

    We can also compute the Kronecker product (or tensor product) of two sparse matrices:
    >>> res4 = s1.kron(s2)
    >>> print(res4.data)
    {(0, 0): -9.0, (2, 0): -6.0}
    >>> print(res4.shape)
    (4, 4)
    """

    def __init__(self, arg):

        if isinstance(arg, tuple):
            self.data = {}
            self.shape = arg

        else:
            self.shape = qml.math.shape(arg)
            self.data = {}
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    entry = arg[i, j]
                    # easiest way to check that entry is not zero
                    if qml.math.count_nonzero(entry) > 0:
                        self.data[(i, j)] = entry

        if len(self.shape) != 2:
            raise ValueError("Expected a 2-dimensional tensor.")

    def row(self):
        """Return a list of the first indices of nonzero entries."""
        return [idx[0] for idx in self.data]

    def col(self):
        """Return a list of the second indices of nonzero entries."""
        return [idx[1] for idx in self.data]

    def nnz(self):
        """Number of nonzero entries."""
        return len(self.data)

    def __add__(self, other):
        """Addition of another SparseMatrix object.
        Args:
            other (SparseMatrix): sparse matrix to add
        """
        if not isinstance(other, type(self)):
            raise ValueError(f"Cannot add SparseMatrix and {type(other)}.")

        new = copy(self)
        for idx, entry in other.data.items():
            if idx in new.data:
                new.data[idx] += entry
                # if the sum cancelled the element, remove it
                if qml.math.count_nonzero(new.data[idx]) == 0:
                    del new.data[idx]
            else:
                new.data[idx] = entry
        return new

    def __sub__(self, other):
        """Subtraction of another SparseMatrix object.

        Args:
            other (SparseMatrix): sparse matrix to add
        """
        return self + -1. * other

    def __mul__(self, other):

        new = copy(self)
        for idx in new.data.keys():
            new.data[idx] *= other

        return new

    __rmul__ = __mul__

    def kron(self, other):
        """An implementation of a sparse kronecker product inspired by ``scipy.sparse.sparse_kron``.

        Args:
            other (SparseMatrix): another sparse matrix

        Returns:
            SparseMatrix: new sparse matrix object representing the kronecker product
        """
        if not isinstance(other, SparseMatrix):
            raise ValueError(f"Can only compute the kronecker product with another SparseMatrix; got {type(other)}")

        output_shape = (self.shape[0] * other.shape[0], self.shape[1] * other.shape[1])

        # note: we can use vanilla np here because we are only bookkeeping indices
        row = np.repeat(self.row(), other.nnz())
        col = np.repeat(self.col(), other.nnz())
        entries = np.repeat(list(self.data.values()), other.nnz())

        row *= other.shape[0]
        col *= other.shape[1]

        # increment block indices
        row, col = row.reshape(-1, other.nnz()), col.reshape(-1, other.nnz())
        row += other.row()
        col += other.col()
        row, col = row.reshape(-1), col.reshape(-1)

        # compute block entries
        entries = qml.math.multiply(qml.math.reshape(entries, (-1, other.nnz())), list(other.data.values()))
        entries = qml.math.reshape(entries, -1)

        res = SparseMatrix(output_shape)
        data = {(i, j): e for i, j, e in zip(row, col, entries)}
        res.data = data

        return res

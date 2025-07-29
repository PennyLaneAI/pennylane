# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper class for generic fragment objects"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

from pennylane.labs.trotter_error import Fragment
from pennylane.labs.trotter_error.abstract import AbstractState


def sparse_fragments(fragments: Sequence[csr_matrix]) -> List[SparseFragment]:
    """Instantiates :class:`~.pennylane.labs.trotter_error.SparseFragment` objects.

    Args:
        fragments (Sequence[csr_matrix]): A sequence of sparse matrices to be used as fragments.

    Returns:
        List[SparseFragment]: A list of :class:`~.pennylane.labs.trotter_error.SparseFragment` objects instantiated from `fragments`.


    **Example**
    This code example demonstrates building fragments from scipy sparse matrices.

    >>> from pennylane.labs.trotter_error import sparse_fragments
    >>> from scipy.sparse import csr_matrix
    >>> matrices = [csr_matrix([[1, 0], [0, 1]]), csr_matrix([[0, 1], [1, 0]])]
    >>> fragments = sparse_fragments(matrices)
    >>> fragments
    [SparseFragment(shape=(2, 2), dtype=int64), SparseFragment(shape=(2, 2), dtype=int64)]
    >>> fragments[0].norm()
    1.4142135623730951
    """

    if len(fragments) == 0:
        return []

    if not any(isinstance(fragment, csr_matrix) for fragment in fragments):
        raise TypeError("Fragments must be csr_matrix objects")

    return [SparseFragment(fragment) for fragment in fragments]


class SparseFragment(Fragment):
    """A wrapper class to allow scipy sparse matrices to be used in the Trotter error functions.

    Args:
        fragment (csr_matrix): The `csr_matrix` to be used as a `~.pennylane.labs.trotter_error.abstract.Fragment`.

    .. note:: :class:`~.pennylane.labs.trotter_error.SparseFragment` objects should be instantated through the ``~.pennylane.labs.trotter_error.sparse_fragments`` function.

    **Example**

    >>> from pennylane.labs.trotter_error import sparse_fragments
    >>> from scipy.sparse import csr_matrix
    >>> matrices = [csr_matrix([[1, 0], [0, 1]]), csr_matrix([[0, 1], [1, 0]])]
    >>> sparse_fragments(matrices)
    [SparseFragment(shape=(2, 2), dtype=int64), SparseFragment(shape=(2, 2), dtype=int64)]
    """

    def __init__(self, fragment: csr_matrix):
        self.fragment = fragment

    def __add__(self, other: SparseFragment):
        new_fragment = self.fragment + other.fragment
        return SparseFragment(new_fragment)

    def __sub__(self, other: SparseFragment):
        return SparseFragment(self.fragment + (-1) * other.fragment)

    def __mul__(self, scalar: float):
        return SparseFragment(scalar * self.fragment)

    def __eq__(self, other: SparseFragment):
        if not isinstance(other, SparseFragment):
            raise TypeError(f"Cannot compare SparseFragment with type {type(other)}.")

        if not np.all(self.fragment.indices == other.fragment.indices):
            return False

        if not np.all(self.fragment.indptr == other.fragment.indptr):
            return False

        return np.allclose(self.fragment.data, other.fragment.data)

    __rmul__ = __mul__

    def __matmul__(self, other: SparseFragment):
        return SparseFragment(self.fragment.dot(other.fragment))

    def apply(self, state: SparseState) -> Any:
        return SparseState(self.fragment.dot(state.csr_matrix.transpose()).transpose())

    def expectation(self, left: SparseState, right: Any) -> complex:
        result = left.csr_matrix.conjugate().dot(self.fragment.dot(right.csr_matrix.transpose()))
        return complex(result.toarray().flatten()[0])

    def norm(self, params: Dict = None) -> float:
        return sp.sparse.linalg.norm(self.fragment)

    def __repr__(self):
        return f"SparseFragment(shape={self.fragment.shape}, dtype={self.fragment.dtype})"


class SparseState(AbstractState):
    """A wrapper class to allow scipy sparse vectors to be used in the Trotter error esimation functions.
    This class is intended to instantiate states to be used along with the `SparseFragment` class.
    
    """
    def __init__(self, matrix: csr_matrix):
        """Initialize the SparseState."""
        self.csr_matrix = matrix

    def __add__(self, other: SparseState) -> SparseState:
        return SparseState(self.csr_matrix + other.csr_matrix)

    def __sub__(self, other: SparseState) -> SparseState:
        return SparseState(self.csr_matrix - other.csr_matrix)

    def __mul__(self, scalar: float) -> SparseState:
        return SparseState(scalar * self.csr_matrix)

    def __rmul__(self, scalar: float) -> SparseState:
        return self.__mul__(scalar)

    @classmethod
    def zero_state(cls, dim: int) -> SparseState: #pylint: disable=arguments-differ
        """Return a representation of the zero state.

        Returns:
            SparseState: an ``SparseState`` representation of the zero state
        """
        return csr_matrix((dim, dim))

    def dot(self, other) -> complex:
        """Compute the dot product of two states.

        Args:
            other: the state to take the dot product with

        Returns:
            complex: the dot product of self and other
        """

        if isinstance(other, SparseState):
            result = self.csr_matrix.conjugate().dot(other.csr_matrix.transpose())
            return complex(result.toarray().flatten()[0])

        raise TypeError(f"Cannot compute dot product between SparseState and {type(other)}")

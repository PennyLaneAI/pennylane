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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

from pennylane.labs.trotter_error import Fragment
from pennylane.labs.trotter_error.abstract import AbstractState

def sparse_fragments(fragments: Sequence[csr_matrix]) -> List[SparseFragment]:
    """Instantiates :class:`~.pennylane.labs.trotter_error.SparseFragment` objects.

    Args:
        fragments (Sequence[Any]): A sequence of objects of the same type. The type is assumed to implement ``__add__``, ``__mul__``, and ``__matmul__``.
        norm_fn (Callable): A function that computes the norm of the fragments.

    Returns:
        List[SparseFragment]: A list of :class:`~.pennylane.labs.trotter_error.SparseFragment` objects instantiated from `fragments`.


    **Example**
    This code example demonstrates building fragments from scipy sparse matrices.

    >>> from pennylane.labs.trotter_error import sparse_fragments
    >>> from scipy.sparse import csr_matrix
    >>> matrices = [csr_matrix([[1, 0], [0, 1]]), csr_matrix([[0, 1], [1, 0]])]
    >>> fragments = sparse_fragments(matrices)
    >>> fragments
    [SparseFragment(type=<class 'scipy.sparse.csr_matrix'>), SparseFragment(type=<class 'scipy.sparse.csr_matrix'>)]
    >>> fragments[0].norm()
    1.4142135623730951
    """

    if len(fragments) == 0:
        return []

    if not any(isinstance(fragment, csr_matrix) for fragment in fragments):
        raise TypeError("Fragments must be csr_matrix objects")

    return [SparseFragment(fragment) for fragment in fragments]


class SparseFragment(Fragment):
    """Abstract class used to define a scipy spares fragment object for product formula error estimation.

    This class allows using any object implementing arithmetic dunder methods to be used
    for product formula error estimation.

    Args:
        fragment (Any): An object that implements the following arithmetic methods:
            ``__add__``, ``__mul__``, and ``__matmul__``.

    .. note:: :class:`~.pennylane.labs.trotter_error.SparseFragment` objects should be instantated through the ``sparse_fragments`` function.

    **Example**

    >>> from pennylane.labs.trotter_error import sparse_fragments
    >>> from scipy.sparse import csr_matrix
    >>> matrices = [csr_matrix([[1, 0], [0, 1]]), csr_matrix([[0, 1], [1, 0]])]
    >>> sparse_fragments(matrices)
    [SparseFragment(type=<class 'scipy.sparse.csr_matrix'>), SparseFragment(type=<class 'scipy.sparse.csr_matrix'>)]
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

        return self.fragment == other.fragment

    __rmul__ = __mul__

    def __matmul__(self, other: SparseFragment):
        return SparseFragment(self.fragment.dot(other.fragment))

    def apply(self, state: SparseState) -> Any:
        """Apply the fragment to a state using the underlying object's ``__matmul__`` method."""
        return SparseState(self.fragment.dot(state.csr_matrix.transpose()).transpose())

    def expectation(self, left: SparseState, right: Any) -> complex:
        """Compute the expectation value using the underlying object's ``__matmul__`` method."""
        result = left.csr_matrix.conjugate().transpose().dot(self.fragment.dot(right.csr_matrix))
        # Convert to scalar - handle both sparse matrix and array cases
        return complex(result.toarray().flatten()[0])

    def norm(self, params: Dict = None) -> float:
        return norm(self.fragment)

    def dot(self, other: Any) -> float:
        """Compute the dot product with another SparseFragment."""
        return SparseFragment(self.fragment.dot(other.fragment))

    def __repr__(self):
        return self.fragment.__repr__()


class SparseState(AbstractState):
    """Abstract class used to define a state object for product formula error estimation.

    A class inheriting from ``MPSState`` must implement the following dunder methods.

    * ``__add__``: implements addition
    * ``__mul__``: implements multiplication

    Additionally, it requires the following methods.

    * ``zero_state``: returns a representation of the zero state
    * ``dot``: implments the dot product of two states
    """
    def __init__(self, matrix: csr_matrix):
        """Initialize the SparseState.
        """
        self.csr_matrix = matrix

    def __add__(self, other: SparseState) -> SparseState:
        return SparseState(self.csr_matrix + other.csr_matrix)

    def __sub__(self, other: SparseState) -> SparseState:
        return SparseState(self.csr_matrix - other.csr_matrix)

    def __mul__(self, scalar: float) -> SparseState:
        return  SparseState(scalar * self.csr_matrix)

    def __rmul__(self, scalar: float) -> SparseState:
        return self.__mul__(scalar)

    @classmethod
    def zero_state(cls) -> SparseState:
        """Return a representation of the zero state.

        Returns:
            SparseState: an ``SparseState`` representation of the zero state
        """
        raise NotImplementedError

    def dot(self, other) -> complex:
        """Compute the dot product of two states.

        Args:
            other: the state to take the dot product with

        Returns:
        complex: the dot product of self and other
        """
        # Handle _AdditiveIdentity (zero state)
        if hasattr(other, '__class__') and 'AdditiveIdentity' in other.__class__.__name__:
            return 0.0

        # Handle SparseState objects
        if isinstance(other, SparseState):
            # For row vectors (1,n), dot product is self.conj() @ other.T
            result = self.csr_matrix.conjugate().dot(other.csr_matrix.transpose())
            # Convert to scalar - handle both sparse matrix and array cases
            return complex(result.toarray().flatten()[0])

        raise TypeError(f"Cannot compute dot product between SparseState and {type(other)}")

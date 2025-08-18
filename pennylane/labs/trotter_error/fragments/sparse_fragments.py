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
"""Wrapper class for Scipy sparse matrices."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import scipy as sp
from scipy.sparse import csr_array

from pennylane.labs.trotter_error import Fragment
from pennylane.labs.trotter_error.abstract import AbstractState


def sparse_fragments(fragments: Sequence[csr_array]) -> List[SparseFragment]:
    """Instantiates :class:`~.pennylane.labs.trotter_error.SparseFragment` objects.

    Args:
        fragments (Sequence[csr_array]): A sequence of sparse matrices to be used as fragments.

    Returns:
        List[SparseFragment]: A list of :class:`~.pennylane.labs.trotter_error.SparseFragment` objects instantiated from `fragments`.


    **Example**
    This code example demonstrates building fragments from scipy sparse matrices.

    >>> from pennylane.labs.trotter_error import sparse_fragments
    >>> from scipy.sparse import csr_array
    >>> matrices = [csr_array([[1, 0], [0, 1]]), csr_array([[0, 1], [1, 0]])]
    >>> fragments = sparse_fragments(matrices)
    >>> fragments
    [SparseFragment(shape=(2, 2), dtype=int64), SparseFragment(shape=(2, 2), dtype=int64)]
    >>> fragments[0].norm()
    1.4142135623730951
    """

    if len(fragments) == 0:
        return []

    if not all(isinstance(fragment, csr_array) for fragment in fragments):
        raise TypeError("Fragments must be csr_array objects")

    return [SparseFragment(fragment) for fragment in fragments]


class SparseFragment(Fragment):
    """A wrapper class to allow scipy sparse matrices to be used in the Trotter error functions.

    Args:
        fragment (csr_array): The `csr_array` to be used as a `~.pennylane.labs.trotter_error.abstract.Fragment`.

    .. note:: :class:`~.pennylane.labs.trotter_error.SparseFragment` objects should be instantated through the ``~.pennylane.labs.trotter_error.sparse_fragments`` function.

    **Example**

    >>> from pennylane.labs.trotter_error import sparse_fragments
    >>> from scipy.sparse import csr_array
    >>> matrices = [csr_array([[1, 0], [0, 1]]), csr_array([[0, 1], [1, 0]])]
    >>> sparse_fragments(matrices)
    [SparseFragment(shape=(2, 2), dtype=int64), SparseFragment(shape=(2, 2), dtype=int64)]
    """

    def __init__(self, fragment: csr_array):
        self.fragment = fragment

    def __add__(self, other: SparseFragment):
        new_fragment = self.fragment + other.fragment
        return SparseFragment(new_fragment)

    def __sub__(self, other: SparseFragment):
        return SparseFragment(self.fragment - other.fragment)

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

    def apply(self, state: SparseState) -> SparseState:
        result = self.fragment.dot(state.state.transpose()).transpose()
        return SparseState(csr_array(result))

    def expectation(self, left: SparseState, right: SparseFragment) -> complex:
        result = left.state.conjugate().dot(self.fragment.dot(right.state.transpose()))
        return complex(result.toarray().flatten()[0])

    def norm(self, params: Dict = None) -> float:
        if params is None:
            params = {}

        ord = params.get("ord")
        return sp.sparse.linalg.norm(self.fragment, ord=ord)

    def __repr__(self):
        return f"SparseFragment(shape={self.fragment.shape}, dtype={self.fragment.dtype})"


class SparseState(AbstractState):
    """A wrapper class to allow scipy sparse vectors to be used in the Trotter error esimation functions.
    This class is intended to instantiate states to be used along with the `SparseFragment` class.
    """

    def __init__(self, state: csr_array):

        if not isinstance(state, csr_array):
            raise TypeError(
                f"SparseState must be instantiated from a csr_array. Got {type(state)}."
            )

        shape = state.shape

        if not len(shape) == 2 or not shape[0] == 1:
            raise ValueError(
                f"Input csr_array must be one-dimensional with shape (1, k). Got shape {shape}."
            )

        self.state = state

    def __add__(self, other: SparseState) -> SparseState:
        return SparseState(self.state + other.state)

    def __sub__(self, other: SparseState) -> SparseState:
        return SparseState(self.state - other.state)

    def __mul__(self, scalar: float) -> SparseState:
        return SparseState(scalar * self.state)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"SparseState({self.state.__repr__()})"

    def __eq__(self, other: SparseState) -> SparseState:
        if not isinstance(other, SparseState):
            raise TypeError(f"Cannot compare SparseFragment with type {type(other)}.")

        if not np.all(self.state.indices == other.state.indices):
            return False

        if not np.all(self.state.indptr == other.state.indptr):
            return False

        return np.allclose(self.state.data, other.state.data)

    @classmethod
    def zero_state(cls, dim: int) -> SparseState:  # pylint: disable=arguments-differ
        """Return a representation of the zero state.

        Returns:
            SparseState: an ``SparseState`` representation of the zero state
        """
        return csr_array((dim, dim))

    def dot(self, other) -> complex:
        """Compute the dot product of two states.

        Args:
            other: the state to take the dot product with

        Returns:
            complex: the dot product of self and other
        """

        if isinstance(other, SparseState):
            result = self.state.conjugate().dot(other.state.transpose())
            return complex(result.toarray().flatten()[0])

        raise TypeError(f"Cannot compute dot product between SparseState and {type(other)}")

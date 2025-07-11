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

from collections.abc import Callable, Sequence
from typing import Any

from pennylane.labs.trotter_error import Fragment


def generic_fragments(fragments: Sequence[Any], norm_fn: Callable = None) -> list[GenericFragment]:
    """Instantiates :class:`~.pennylane.labs.trotter_error.GenericFragment` objects.

    Args:
        fragments (Sequence[Any]): A sequence of objects of the same type. The type is assumed to implement ``__add__``, ``__mul__``, and ``__matmul__``.
        norm_fn (Callable): A function that computes the norm of the fragments.

    Returns:
        List[GenericFragment]: A list of :class:`~.pennylane.labs.trotter_error.GenericFragment` objects instantiated from `fragments`.


    **Example**

    This code example demonstrates building fragments from numpy matrices.

    >>> from pennylane.labs.trotter_error import generic_fragments
    >>> import numpy as np
    >>> matrices = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
    >>> fragments = generic_fragments(matrices, norm_fn=np.linalg.norm)
    >>> fragments
    [GenericFragment(type=<class 'numpy.ndarray'>), GenericFragment(type=<class 'numpy.ndarray'>)]
    >>> fragments[0].norm()
    1.4142135623730951
    """

    if len(fragments) == 0:
        return []

    frag_type = type(fragments[0])

    if not all(isinstance(fragment, frag_type) for fragment in fragments):
        raise TypeError("All fragments must be of the same type.")

    if not hasattr(frag_type, "__add__"):
        raise TypeError(f"Fragment of type {frag_type} does not implement __add__.")

    if not hasattr(frag_type, "__mul__"):
        raise TypeError(f"Fragment of type {frag_type} does not implement __mul__.")

    if not hasattr(frag_type, "__matmul__"):
        raise TypeError(f"Fragment of type {frag_type} does not implement __matmul__.")

    return [GenericFragment(fragment, norm_fn=norm_fn) for fragment in fragments]


class GenericFragment(Fragment):
    """Abstract class used to define a generic fragment object for product formula error estimation.

    This class allows using any object implementing arithmetic dunder methods to be used
    for product formula error estimation.

    Args:
        fragment (Any): An object that implements the following arithmetic methods:
            ``__add__``, ``__mul__``, and ``__matmul__``.
        norm_fn (optional, Callable): A function used to compute the norm of ``fragment``.

    .. note:: :class:`~.pennylane.labs.trotter_error.GenericFragment` objects should be instantated through the ``generic_fragments`` function.

    **Example**

    >>> from pennylane.labs.trotter_error import generic_fragments
    >>> import numpy as np
    >>> matrices = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
    >>> generic_fragments(matrices)
    [GenericFragment(type=<class 'numpy.ndarray'>), GenericFragment(type=<class 'numpy.ndarray'>)]
    """

    def __init__(self, fragment: Any, norm_fn: Callable = None):
        self.fragment = fragment
        self.norm_fn = norm_fn

    def __add__(self, other: GenericFragment):
        return GenericFragment(self.fragment + other.fragment, norm_fn=self.norm_fn)

    def __sub__(self, other: GenericFragment):
        return GenericFragment(self.fragment + (-1) * other.fragment, norm_fn=self.norm_fn)

    def __mul__(self, scalar: float):
        return GenericFragment(scalar * self.fragment, norm_fn=self.norm_fn)

    def __eq__(self, other: GenericFragment):
        if not isinstance(self.fragment, type(other)):
            return False

        return self.fragment == other.fragment

    __rmul__ = __mul__

    def __matmul__(self, other: GenericFragment):
        return GenericFragment(self.fragment @ other.fragment, norm_fn=self.norm_fn)

    def apply(self, state: Any) -> Any:
        """Apply the fragment to a state using the underlying object's ``__matmul__`` method."""
        return self.fragment @ state

    def expectation(self, left: Any, right: Any) -> float:
        """Compute the expectation value using the underlying object's ``__matmul__`` method."""
        return left @ self.fragment @ right

    def norm(self, params: dict = None) -> float:
        """Compute the norm of the fragment."""
        if self.norm_fn:
            params = params or {}
            return self.norm_fn(self.fragment, **params)

        raise NotImplementedError(
            "GenericFragment was constructed without specifying the norm function."
        )

    def __repr__(self):
        return f"GenericFragment(type={type(self.fragment)})"

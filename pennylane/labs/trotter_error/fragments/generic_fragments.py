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

from typing import Any, Callable, Dict, List, Sequence

from pennylane.labs.trotter_error import Fragment


def generic_fragments(fragments: Sequence[Any], norm_fn: Callable = None) -> List[GenericFragment]:
    """Instantiates ``GenericFragment`` objects.

    Args:
        fragments (Sequence[Any]): A sequence of Python objects of the same type
        norm_fn (Callable): A function that computes the norm of the fragments.

    Returns:
        List[GenericFragment]: A list of GenericFragment objects instantiated from `fragments`.


    **Example**

    This code example demonstrates building fragments from numpy matrices.

    >>> from pennylane.labs.trotter import generic_fragments
    >>> import numpy as np
    >>> matrices = [np.random.random(size=(10, 10)) for _ in range(100)]
    >>> fragments = generic_fragments(matrices, norm_fn=np.linalg.norm)
    """

    if len(fragments) == 0:
        return []

    frag_type = type(fragments[0])

    if not all(isinstance(fragment, frag_type) for fragment in fragments):
        raise TypeError("All fragments must be of the same type.")

    if not hasattr(frag_type, "__add__"):
        raise TypeError(f"Fragment of type {frag_type} does not implement __add__.")

    if not hasattr(frag_type, "__sub__"):
        raise TypeError(f"Fragment of type {frag_type} does not implement __sub__.")

    if not hasattr(frag_type, "__mul__"):
        raise TypeError(f"Fragment of type {frag_type} does not implement __mul__.")

    if not hasattr(frag_type, "__matmul__"):
        raise TypeError(f"Fragment of type {frag_type} does not implement __matmul__.")

    return [GenericFragment(fragment, norm_fn=norm_fn) for fragment in fragments]


class GenericFragment(Fragment):
    """This class allows users to use any Python object implementing arithmetic dunder methods to be used
    in the Trotter error workflow.

    Args:
        fragment (Any): Any Python object. The object is assumed to implement the following methods:
            ``__add__``, ``__sub__``, ``__mul__``, and ``__matmul__``.
        norm_fn (optional, Callable): This is a function used to compute the norm of `fragment`, which is
            needed for some Trotter error functionality.

    ``GenericFragment`` objects should be instantated through the ``generic_fragments`` function.
    """

    def __init__(self, fragment: Any, norm_fn: Callable = None):
        self.fragment = fragment
        self.norm_fn = norm_fn

    def __add__(self, other: GenericFragment):
        return GenericFragment(self.fragment + other.fragment, norm_fn=self.norm_fn)

    def __sub__(self, other: GenericFragment):
        return GenericFragment(self.fragment - other.fragment, norm_fn=self.norm_fn)

    def __mul__(self, scalar: float):
        return GenericFragment(scalar * self.fragment, norm_fn=self.norm_fn)

    __rmul__ = __mul__

    def __matmul__(self, other: GenericFragment):
        return GenericFragment(self.fragment @ other.fragment, norm_fn=self.norm_fn)

    def apply(self, state: Any) -> Any:
        """Apply the Fragment to a state using the underlying object's __matmul__ method."""
        return self.fragment @ state

    def expectation(self, left: Any, right: Any) -> float:
        """Compute the expectation value using the underlying object's __matmul__ method."""
        return left @ self.fragment @ right

    def norm(self, params: Dict = None) -> float:
        """Compute the norm of the ``GenericFragment`` by calling ``norm_fn``."""
        if self.norm_fn:
            params = params or {}
            return self.norm_fn(self.fragment, **params)

        raise NotImplementedError(
            "GenericFragment was constructed without specifying the norm function."
        )

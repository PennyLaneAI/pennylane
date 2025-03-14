"""Wrapper class for generic fragment objects"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from pennylane.labs.trotter import Fragment


def generic_fragments(fragments: Sequence[Any], norm_fn: Callable = None) -> List[GenericFragment]:
    """Wrapper function for GenericFragment"""

    if len(fragments) > 0:
        frag_type = type(fragments[0])
    else:
        return []

    if not all(isinstance(fragment, frag_type) for fragment in fragments):
        raise TypeError("All fragments must be of the same type.")

    if not hasattr(frag_type, "__add__"):
        raise TypeError(f"Fragment type {frag_type} does not implement __add__.")

    if not hasattr(frag_type, "__sub__"):
        raise TypeError(f"Fragment type {frag_type} does not implement __sub__.")

    if not hasattr(frag_type, "__mul__"):
        raise TypeError(f"Fragment type {frag_type} does not implement __mul__.")

    if not hasattr(frag_type, "__matmul__"):
        raise TypeError(f"Fragment type {frag_type} does not implement __matmul__.")

    return [GenericFragment(fragment, norm_fn=norm_fn) for fragment in fragments]


class GenericFragment(Fragment):
    """Wrapper class to support any Python object implementing arithmetic dunder methods."""

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
        if self.norm_fn:
            return self.norm_fn(self.fragment, **params)

        raise NotImplementedError

"""The VibronicTerm class"""

from __future__ import annotations

from itertools import product
from typing import Sequence, Tuple

from vibronic_tree import Node


class VibronicTerm:
    """The VibronicTerm class"""

    def __init__(self, ops: Tuple[str], coeffs: Node) -> VibronicTerm:
        self.ops = ops
        self.coeffs = coeffs

    def __add__(self, other: VibronicTerm) -> VibronicTerm:
        if self.ops != other.ops:
            raise ValueError(f"Cannot add term {self.ops} with term {other.ops}.")

        return VibronicTerm(self.ops, Node.sum_node(self.coeffs, other.coeffs))

    def __sub__(self, other: VibronicTerm) -> VibronicTerm:
        if self.ops != other.ops:
            raise ValueError(f"Cannot subtract term {self.ops} with term {other.ops}.")

        return VibronicTerm(
            self.ops, Node.sum_node(self.coeffs, Node.scalar_node(-1, other.coeffs))
        )

    def __mul__(self, scalar: float) -> VibronicTerm:
        return VibronicTerm(self.ops, Node.scalar_node(scalar, self.coeffs))

    __rmul__ = __mul__

    def __matmul__(self, other: VibronicTerm) -> VibronicTerm:
        return VibronicTerm(self.ops + other.ops, Node.outer_node(self.coeffs, other.coeffs))

    def __repr__(self) -> str:
        return f"({self.ops.__repr__()}, {self.coeffs.__repr__()})"

    def __eq__(self, other: VibronicTerm) -> bool:
        if self.ops != other.ops:
            return False

        return self.coeffs == other.coeffs


class VibronicWord:
    """The VibronicWord class"""

    def __init__(self, terms: Sequence[VibronicTerm]) -> VibronicWord:
        self.terms = tuple(terms)
        self._lookup = {term.ops: term for term in terms}

    def __add__(self, other: VibronicWord) -> VibronicWord:
        l_ops = {term.ops for term in self.terms}
        r_ops = {term.ops for term in other.terms}

        new_terms = []

        for op in l_ops.intersection(r_ops):
            new_terms.append(self._lookup[op] + other._lookup[op])

        for op in l_ops.difference(r_ops):
            new_terms.append(self._lookup[op])

        for op in r_ops.difference(l_ops):
            new_terms.append(other._lookup[op])

        return VibronicWord(new_terms)

    def __sub__(self, other: VibronicWord) -> VibronicWord:
        l_ops = {term.ops for term in self.terms}
        r_ops = {term.ops for term in other.terms}

        new_terms = []

        for op in l_ops.intersection(r_ops):
            new_terms.append(self._lookup[op] - other._lookup[op])

        for op in l_ops.difference(r_ops):
            new_terms.append(self._lookup[op])

        for op in r_ops.difference(l_ops):
            new_terms.append((-1) * other._lookup[op])

        return VibronicWord(new_terms)

    def __mul__(self, scalar: float) -> VibronicWord:
        return VibronicWord([scalar * term for term in self.terms])

    __rmul__ = __mul__

    def __matmul__(self, other: VibronicWord) -> VibronicWord:
        return VibronicWord(
            [
                VibronicTerm(l_term.ops + r_term.ops, l_term.coeffs @ r_term.coeffs)
                for l_term, r_term in product(self.terms, other.terms)
            ]
        )

    def __repr__(self) -> str:
        return self.terms.__repr__()

    def __eq__(self, other: VibronicWord) -> bool:
        return self._lookup == other._lookup

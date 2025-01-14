"""The VibronicTerm class"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Sequence, Tuple

import numpy as np

from .vibronic_tree import Node


class VibronicTerm:
    """The VibronicTerm class"""

    def __init__(self, ops: Tuple[str], coeffs: Node) -> VibronicTerm:
        self.ops = ops
        self.coeffs = coeffs

    def __add__(self, other: VibronicTerm) -> VibronicTerm:
        if self.is_zero:
            return other

        if other.is_zero:
            return self

        if self.ops != other.ops:
            raise ValueError(f"Cannot add term {self.ops} with term {other.ops}.")

        return VibronicTerm(self.ops, Node.sum_node(self.coeffs, other.coeffs))

    def __sub__(self, other: VibronicTerm) -> VibronicTerm:
        if self.is_zero:
            return (-1) * other

        if other.is_zero:
            return self

        if self.ops != other.ops:
            raise ValueError(f"Cannot subtract term {self.ops} with term {other.ops}.")

        return VibronicTerm(
            self.ops, Node.sum_node(self.coeffs, Node.scalar_node(-1, other.coeffs))
        )

    def __mul__(self, scalar: float) -> VibronicTerm:
        if np.isclose(scalar, 0):
            return VibronicTerm.zero_term()

        return VibronicTerm(self.ops, Node.scalar_node(scalar, self.coeffs))

    __rmul__ = __mul__

    def __imul__(self, scalar: float) -> VibronicTerm:
        if np.isclose(scalar, 0):
            return VibronicTerm.zero_term()

        self.coeffs = Node.scalar_node(scalar, self.coeffs)
        return self

    def __matmul__(self, other: VibronicTerm) -> VibronicTerm:
        if other.is_zero:
            return self

        return VibronicTerm(self.ops + other.ops, Node.outer_node(self.coeffs, other.coeffs))

    def __repr__(self) -> str:
        return f"({self.ops.__repr__()}, {self.coeffs.__repr__()})"

    def __eq__(self, other: VibronicTerm) -> bool:
        if self.ops != other.ops:
            return False

        return self.coeffs == other.coeffs

    @property
    def is_zero(self) -> bool:
        """If is_zero returns true the term evaluates to zero, however there are false negatives"""
        return self.coeffs.is_zero

    @classmethod
    def zero_term(cls) -> VibronicTerm:
        """Returns a VibronicTerm representing 0"""
        return VibronicTerm(tuple(), Node.tensor_node(np.array(0)))


class VibronicWord:
    """The VibronicWord class"""

    def __init__(self, terms: Sequence[VibronicTerm]) -> VibronicWord:
        terms = tuple(filter(lambda term: not term.is_zero, terms))
        self.is_zero = len(terms) == 0

        self._lookup = defaultdict(lambda: VibronicTerm.zero_term()) #pylint: disable=unnecessary-lambda
        for term in terms:
            self._lookup[term.ops] += term

        self.terms = tuple(self._lookup.values())


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

    def __imul__(self, scalar: float) -> VibronicWord:
        for term in self.terms:
            term *= scalar

        return self

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

    @classmethod
    def zero_word(cls) -> VibronicWord:
        """Return a VibronicWord representing 0"""
        return VibronicWord([VibronicTerm.zero_term()])

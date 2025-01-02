"""The VibronicTerm class"""

from __future__ import annotations

from itertools import product
from typing import Sequence, Tuple

from vibronic_tree import Node


class VibronicTerm:
    """The VibronicTerm class"""

    def __init__(self, term: Tuple[str], coeffs: Node) -> VibronicTerm:
        self.term = term
        self.coeffs = coeffs

    def __add__(self, other: VibronicTerm) -> VibronicTerm:
        if self.term != other.term:
            raise ValueError(f"Cannot add term {self.term} with term {other.term}.")

        return VibronicTerm(self.term, Node.sum_node(self.coeffs, other.coeffs))

    def __sub__(self, other: VibronicTerm) -> VibronicTerm:
        if self.term != other.term:
            raise ValueError(f"Cannot subtract term {self.term} with term {other.term}.")

        return VibronicTerm(
            self.term, Node.sum_node(self.coeffs, Node.scalar_node(-1, other.coeffs))
        )

    def __mul__(self, scalar: float) -> VibronicTerm:
        return VibronicTerm(self.term, Node.scalar_node(scalar, self.coeffs))

    __rmul__ = __mul__

    def __matmul__(self, other: VibronicTerm) -> VibronicTerm:
        return VibronicTerm(self.term + other.term, Node.outer_node(self.coeffs, other.coeffs))

    def __repr__(self) -> str:
        return f"({self.term.__repr__()}, {self.coeffs.__repr__()})"

    def __eq__(self, other: VibronicTerm) -> bool:
        if self.term != other.term:
            return False

        return self.coeffs == other.coeffs


class VibronicWord:
    """The VibronicWord class"""

    def __init__(self, terms: Sequence[VibronicTerm]) -> VibronicWord:
        self.terms = tuple(terms)
        self._lookup = {term.term: term for term in terms}

    def __add__(self, other: VibronicWord) -> VibronicWord:
        l_terms = {term.term for term in self.terms}
        r_terms = {term.term for term in other.terms}

        new_terms = []

        for term in l_terms.intersection(r_terms):
            new_terms.append(self._lookup[term] + other._lookup[term])

        for term in l_terms.difference(r_terms):
            new_terms.append(self._lookup[term])

        for term in r_terms.difference(l_terms):
            new_terms.append(other._lookup[term])

        return VibronicWord(new_terms)

    def __sub__(self, other: VibronicWord) -> VibronicWord:
        l_terms = {term.term for term in self.terms}
        r_terms = {term.term for term in other.terms}

        new_terms = []

        for term in l_terms.intersection(r_terms):
            new_terms.append(self._lookup[term] - other._lookup[term])

        for term in l_terms.difference(r_terms):
            new_terms.append(self._lookup[term])

        for term in r_terms.difference(l_terms):
            new_terms.append((-1) * other._lookup[term])

        return VibronicWord(new_terms)

    def __mul__(self, scalar: float) -> VibronicWord:
        return VibronicWord([scalar * term for term in self.terms])

    __rmul__ = __mul__

    def __matmul__(self, other: VibronicWord) -> VibronicWord:
        return VibronicWord(
            [
                VibronicTerm(l_term.term + r_term.term, l_term.coeffs @ r_term.coeffs)
                for l_term, r_term in product(self.terms, other.terms)
            ]
        )

    def __repr__(self) -> str:
        return self.terms.__repr__()

    def __eq__(self, other: VibronicWord) -> bool:
        return self._lookup == other._lookup

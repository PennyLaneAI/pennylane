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
"""The ProductFormula class"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Hashable
from itertools import permutations
from typing import Dict, Generator, List, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import expm

from pennylane.labs.trotter_error.abstract import Fragment


class ProductFormula:
    """Class for representing product formulas"""

    def __init__(
        self,
        terms: Union[Sequence[Hashable], Sequence[ProductFormula]],
        coeffs: Sequence[float] = None,
        exponent: float = 1.0,
        label: str = None,
        include_i: bool = True,
    ):

        if any(not isinstance(term, type(terms[0])) for term in terms):
            raise TypeError("All terms must have the same type.")

        if not isinstance(terms[0], (Hashable, ProductFormula)):
            raise TypeError(
                f"Terms must either be Hashable type or of type ProductFormula. Got {type(terms[0])} instead."
            )

        if isinstance(terms[0], ProductFormula):
            self.recursive = True
        elif isinstance(terms[0], Hashable):
            if coeffs is None:
                raise ValueError("List of coefficients must be given.")
            if len(coeffs) != len(terms):
                raise ValueError("Number of coefficients must equal number of terms.")

            self.recursive = False

        self.fragments = set()
        for term in terms:
            if isinstance(term, ProductFormula):
                self.fragments = set.union(self.fragments, term.fragments)
            else:
                self.fragments.add(term)

        self.terms = terms

        if coeffs and include_i:
            self.coeffs = [1j * coeff for coeff in coeffs]
        elif coeffs:
            self.coeffs = coeffs
        else:
            self.coeffs = [1] * len(self.terms)

        self.exponent = exponent
        self.label = label

        self._ordered_terms = {}
        position = 0
        for term in self.terms:
            if term not in self._ordered_terms:
                self._ordered_terms[term] = position
                position += 1

    def __call__(self, t: float):
        if self.recursive:
            return ProductFormula(
                [term(t) for term in self.terms], exponent=self.exponent, label=f"{self.label}({t})"
            )

        ret = copy.copy(self)
        ret.coeffs = [t * coeff for coeff in self.coeffs]

        if self.label:
            ret.label = f"{self.label}({t})"

        return ret

    def __eq__(self, other: ProductFormula) -> bool:
        if self.terms != other.terms:
            return False

        if not self.recursive and self.coeffs != other.coeffs:
            return False

        return self.exponent == other.exponent

    def __hash__(self) -> int:
        terms = tuple(self.terms)
        return hash((terms, self.exponent))

    def __matmul__(self, other: ProductFormula) -> ProductFormula:
        return ProductFormula([self, other], label=f"{self.label}@{other.label}")

    def __pow__(self, z: float) -> ProductFormula:
        ret = copy.copy(self)
        ret.exponent = z * self.exponent
        ret.label = f"{self.label}**{z * self.exponent}"

        return ret

    def __repr__(self) -> str:
        if self.label:
            return f"{self.label}**{self.exponent}" if self.exponent != 1 else f"{self.label}"

        if self.recursive:
            return "@".join(term.__repr__() for term in self.terms)

        return "@".join([f"Exp({coeff}*H_{term})" for coeff, term in zip(self.coeffs, self.terms)])

    def bch_approx(self, max_order: int) -> Dict[Tuple[int], float]:
        """Returns an approximation of the BCH expansion in terms of right-nested commutators up to order `max_order`.
        This method follows the procedure outlined in `arXiv:2006.15869 <https://arxiv.org/pdf/2006.15869>`.
        """

        return _remove_redundancies(
            [_kth_order_terms(self.terms, self.coeffs, k) for k in range(1, max_order + 1)],
            self._ordered_terms,
        )

    def to_matrix(self, fragments: Dict[Hashable, Fragment], accumulator: Fragment) -> np.ndarray:
        """Returns a numpy representation of the product formula"""
        acc = copy.copy(accumulator)
        for term, coeff in zip(self.terms, self.coeffs):
            if isinstance(term, ProductFormula):
                accumulator @= term.to_matrix(fragments, copy.copy(acc))
            else:
                accumulator @= expm(coeff * fragments[term])

        return accumulator


def _kth_order_terms(
    fragments: Sequence[int], coeffs: Sequence[float], k: int
) -> Dict[Tuple[int], float]:
    n = len(fragments)

    terms = defaultdict(float)

    for partition in _partitions(n, k):
        args = tuple()
        coeff = 1 / math.prod(math.factorial(i) for i in partition)

        for i, j in enumerate(partition):
            args += (fragments[i],) * j
            coeff *= coeffs[i] ** j

        for key, value in _phi(args).items():
            terms[key] += coeff * value

    return terms


def _partitions(n: int, m: int) -> Generator[Tuple[int]]:
    """Return tuples containing number of ways n ordered integers can sum to m"""

    if n == 1:
        yield (m,)
    elif m == 0:
        yield (0,) * n
    else:
        for i in range(m + 1):
            for partition in _partitions(n - 1, m - i):
                yield (i,) + partition


def _phi(fragments: Sequence[int]) -> Dict[Tuple[int], float]:
    n = len(fragments)
    terms = defaultdict(float)

    for permutation in permutations(range(n - 1)):
        d = _n_descents(permutation)
        commutator = tuple(fragments[i] for i in permutation) + (fragments[n - 1],)
        terms[commutator] += ((-1) ** d) / n / math.comb(n - 1, d)

    return terms


def _n_descents(permutation: Sequence[int]) -> int:
    n = 0

    for i in range(len(permutation) - 1):
        if permutation[i] > permutation[i + 1]:
            n += 1

    return n


def _remove_redundancies(
    term_dicts: List[Dict[Tuple[int], float]],
    term_order: Dict[Hashable, int],
) -> List[Dict[Tuple[int], float]]:

    max_order = len(term_dicts)

    for terms in term_dicts[1:]:
        delete = []
        swap = []

        for commutator in terms.keys():
            if commutator[-1] == commutator[-2]:
                delete.append(commutator)
            if term_order[commutator[-1]] < term_order[commutator[-2]]:
                swap.append(commutator)

        for commutator in delete:
            del terms[commutator]

        for commutator in swap:
            new_commutator = list(commutator)
            new_commutator[-1] = commutator[-2]
            new_commutator[-2] = commutator[-1]
            new_commutator = tuple(new_commutator)

            terms[new_commutator] -= terms[commutator]
            del terms[commutator]

    if max_order < 4:
        return _drop_zeros(term_dicts)

    swap = []
    for commutator in term_dicts[3].keys():
        if term_order[commutator[1]] < term_order[commutator[0]]:
            swap.append(commutator)

    for commutator in swap:
        new_commutator = list(commutator)
        new_commutator[0] = commutator[1]
        new_commutator[1] = commutator[0]
        new_commutator = tuple(new_commutator)

        term_dicts[3][new_commutator] += term_dicts[3][commutator]
        del term_dicts[3][commutator]

    return _drop_zeros(term_dicts)


def _drop_zeros(term_dicts: List[Dict[Tuple[int], float]]) -> List[Dict[Tuple[int], float]]:
    for terms in term_dicts:
        delete = []
        for commutator, coeff in terms.items():
            if np.isclose(coeff, 0):
                delete.append(commutator)

        for commutator in delete:
            del terms[commutator]

    return term_dicts

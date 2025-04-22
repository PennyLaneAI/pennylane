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

import math
from collections import defaultdict
from itertools import permutations
from typing import Dict, Generator, List, Sequence, Tuple

import numpy as np


class ProductFormula:
    """Class for representing product formulas"""

    def __init__(
        self,
        coeffs: Sequence[float],
        fragments: Sequence[int],
        exponent: int = 1,
    ):

        if len(coeffs) != len(fragments):
            raise ValueError(f"Got {len(coeffs)} coefficients and {len(fragments)} fragments.")

        for fragment in fragments:
            if not isinstance(fragment, int):
                raise TypeError(f"Fragments must have type int, got {type(fragment)} instead.")

        self.labels = set()
        for fragment in fragments:
            if isinstance(fragment, ProductFormula):
                self.labels = set.union(self.labels, fragment.labels)
            else:
                self.labels.add(fragment)

        self.coeffs = coeffs
        self.fragments = fragments
        self.exponent = exponent

    def __repr__(self) -> str:
        reps = [
            f"Exp({coeff}*H_{fragment})" for coeff, fragment in zip(self.coeffs, self.fragments)
        ]

        return "*".join(reps)

    def bch_approx(self, max_order: int) -> Dict[Tuple[int], float]:
        terms = []

        for k in range(1, max_order + 1):
            terms.append(_kth_order_terms(self.fragments, k))

        return _remove_redundancies(terms)


def _kth_order_terms(fragments: Sequence[int], k: int) -> Dict[Tuple[int], float]:
    n = len(fragments)

    terms = defaultdict(float)

    for partition in _partitions(n, k):
        args = tuple()
        coeff = 1 / math.prod(math.factorial(i) for i in partition)

        for i, j in enumerate(partition):
            args += (fragments[i],) * j

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
) -> List[Dict[Tuple[int], float]]:

    max_order = len(term_dicts)

    for terms in term_dicts[1:]:
        delete = []
        swap = []

        for commutator in terms.keys():
            if commutator[-1] == commutator[-2]:
                delete.append(commutator)
            if commutator[-1] < commutator[-2]:
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
        if commutator[1] < commutator[0]:
            swap.append(commutator)

    for commutator in swap:
        new_commutator = list(commutator)
        new_commutator[0] = commutator[1]
        new_commutator[1] = commutator[0]
        new_commutator = tuple(new_commutator)

        term_dicts[3][new_commutator] -= term_dicts[3][commutator]
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

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
"""This file contains the BCH computation"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable
from itertools import permutations, product
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from pennylane.labs.trotter_error import Fragment, ProductFormula


def bch_expansion(product_formula: ProductFormula, order: int) -> Dict[Tuple[int], complex]:
    return _drop_zeros(_bch_expansion(product_formula, order, {}))


def _bch_expansion(
    product_formula: ProductFormula, order: int, term_dict: Dict[Tuple[Hashable], complex]
) -> Dict[Tuple[Hashable], complex]:
    """Returns the commutators of order ``order`` of the BCH expansion of ``product_formula``.
    This method follows the procedure outlined in `arXiv:2006.15869 <https://arxiv.org/pdf/2006.15869>`.
    """

    bch = _bch(
        product_formula.terms,
        [coeff * product_formula.timestep for coeff in product_formula.coeffs],
        product_formula._ordered_terms,
        order,
    )

    if not product_formula.recursive:
        return _apply_exponent(bch, product_formula.exponent)

    for pf in product_formula.terms:
        if pf in term_dict:
            continue
        term_dict[pf] = _bch_expansion(pf, order, term_dict)

    merged_bch = [defaultdict(complex) for _ in range(order)]

    for ith_order_commutators in bch:
        for commutator, coeff in ith_order_commutators.items():
            for j, merged in enumerate(_merge_commutators(commutator, term_dict, order, coeff)):
                merged_bch[j] = _add_dicts(merged_bch[j], merged)

    return _remove_redundancies(
        _apply_exponent(merged_bch, product_formula.exponent), product_formula.ordered_fragments()
    )


def _bch(
    fragments: Sequence[Fragment],
    coeffs: Sequence[complex],
    term_order: Dict[Hashable, int],
    order: int,
):

    if len(fragments) < 3:
        return _remove_redundancies(
            [_kth_order_terms(fragments, coeffs, k) for k in range(1, order + 1)], term_order
        )

    terms = {
        "head": _bch([fragments[0]], [coeffs[0]], term_order, order),
        "tail": _bch(fragments[1:], coeffs[1:], term_order, order),
    }

    bch = _bch(["head", "tail"], [1, 1], {"head": 0, "tail": 1}, order)
    merged_bch = [defaultdict(complex) for _ in range(order)]

    for ith_order_commutators in bch:
        for commutator, coeff in ith_order_commutators.items():
            for j, merged in enumerate(_merge_commutators(commutator, terms, order, coeff)):
                merged_bch[j] = _add_dicts(merged_bch[j], merged)

    return _remove_redundancies(merged_bch, term_order)


def _apply_exponent(bch, exponent):
    for i, ith_order_commutators in enumerate(bch):
        for commutator, coeff in ith_order_commutators.items():
            bch[i][commutator] = coeff * exponent

    return bch


def _kth_order_terms(
    fragments: Sequence[Hashable], coeffs: Sequence[complex], k: int
) -> Dict[Tuple[int], complex]:
    n = len(fragments)

    terms = defaultdict(complex)

    for partition in _partitions_nonnegative(n, k):
        args = tuple()
        coeff = 1 / math.prod(math.factorial(i) for i in partition)

        for i, j in enumerate(partition):
            args += (fragments[i],) * j
            coeff *= coeffs[i] ** j

        for key, value in _phi(args).items():
            terms[key] += coeff * value

    return terms


def _merge_commutators(commutator, terms, order, bch_coeff):
    merged = [defaultdict(complex) for _ in range(order)]

    for x in _commutator_terms(commutator, terms, order):
        new_commutator = tuple(y[0] for y in x)
        term_coeff = math.prod(y[1] for y in x) * bch_coeff
        commutator_order = _commutator_order(new_commutator)

        merged[commutator_order - 1][new_commutator] += term_coeff

    return merged


def _commutator_terms(commutator, terms, order):
    for i in range(order):
        for partition in _partitions_positive(len(commutator), i + 1):
            kept_terms = [
                terms[term][partition[j] - 1].items() for j, term in enumerate(commutator)
            ]
            yield from product(*kept_terms)


def _commutator_order(commutator):
    order = 0

    for term in commutator:
        if isinstance(term, tuple):
            order += _commutator_order(term)
        else:
            order += 1

    return order


def _add_dicts(d1, d2):
    for key, value in d2.items():
        d1[key] += value

    return d1


def _partitions_nonnegative(n: int, m: int) -> Generator[Tuple[int]]:
    """Return tuples containing number of ways n ordered integers can sum to m"""

    if n == 1:
        yield (m,)
    elif m == 0:
        yield (0,) * n
    else:
        for i in range(m + 1):
            for partition in _partitions_nonnegative(n - 1, m - i):
                yield (i,) + partition


def _partitions_positive(m: int, n: int):
    """number of ways to sum m positive integers to n"""

    if m == 1:
        yield (n,)
    elif m == n:
        yield (1,) * m
    elif n < m:
        yield from []
    else:
        for i in range(1, n - m + 2):
            for partition in _partitions_positive(m - 1, n - i):
                yield (i,) + partition


def _phi(fragments: Sequence[int]) -> Dict[Tuple[int], float]:
    n = len(fragments)
    terms = defaultdict(complex)

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
    """Applies the following identites to the dictionary of commutators:

    1. [A, A] = 0
    2. [A, [C, B]] = -[A, [B, C]]
    3. [B, [A, [C, D]]] = [A, [B, [C, D]]]
    """

    def less_than(x, y):
        """Define an ordering on nested tuples"""

        if isinstance(x, tuple) and isinstance(y, tuple):
            if len(x) < len(y):
                return True

            for a, b in zip(x, y):
                if a == b:
                    continue

                return less_than(a, b)

            return False

        if isinstance(x, tuple):
            return False

        if isinstance(y, tuple):
            return True

        return term_order[x] < term_order[y]

    max_order = len(term_dicts)

    term_dicts = [_make_right_nested(terms) for terms in term_dicts]

    for terms in term_dicts[1:]:
        _delete(terms)
        _swap(terms, less_than)

    if max_order < 4:
        return term_dicts

    for terms in term_dicts[3:]:
        swap = []
        for commutator in terms.keys():
            if commutator[-1] == commutator[-4] and commutator[-2] == commutator[-3]:
                swap.append(commutator)

        for commutator in swap:
            new_commutator = list(commutator)
            new_commutator[-4] = commutator[-3]
            new_commutator[-3] = commutator[-4]
            new_commutator = tuple(new_commutator)

            terms[new_commutator] += terms[commutator]
            del terms[commutator]

    return term_dicts


def _swap(terms, less_than):
    swap = []
    for commutator in terms.keys():
        if less_than(commutator[-1], commutator[-2]):
            swap.append(commutator)

    for commutator in swap:
        new_commutator = list(commutator)
        new_commutator[-1] = commutator[-2]
        new_commutator[-2] = commutator[-1]
        new_commutator = tuple(new_commutator)

        terms[new_commutator] -= terms[commutator]
        del terms[commutator]


def _delete(terms):
    delete = []
    for commutator in terms.keys():
        if commutator[-1] == commutator[-2]:
            delete.append(commutator)

    for commutator in delete:
        del terms[commutator]


def _make_right_nested(terms):
    ret = defaultdict(complex)

    for commutator1, coeff1 in terms.items():
        for commutator2, coeff2 in _right_nested(commutator1).items():
            ret[commutator2] += coeff1 * coeff2

    return ret


def _right_nested(commutator) -> Dict[Tuple, float]:
    if isinstance(commutator[-1], tuple):
        commutator = commutator[:-1] + commutator[-1]

    if not any(isinstance(x, tuple) for x in commutator):
        return {commutator: 1}

    if (
        len(commutator) == 2
        and isinstance(commutator[0], tuple)
        and isinstance(commutator[1], tuple)
    ):
        return _right_nest_two_comms(commutator)

    for i in range(len(commutator) - 1, -1, -1):
        if isinstance(commutator[i], tuple):
            break

    coc = (commutator[i], commutator[i + 1 :])
    partially_nested = {
        commutator[:i] + nested: coeff for nested, coeff in _right_nest_two_comms(coc).items()
    }

    ret = {}

    for partial, coeff1 in partially_nested.items():
        for nested, coeff2 in _right_nested(partial).items():
            ret[nested] = coeff1 * coeff2

    return ret


def _right_nest_two_comms(commutator) -> Dict[Tuple, float]:
    """Assume commutator is the commutator of two right-nested commutators
    Apply the Jacobi identity [A, [B, C]] = [C, [B, A]] - [B, [C, A]]
    """

    if len(commutator[0]) == 1:
        return {commutator[0] + commutator[1]: 1}

    if len(commutator[1]) == 1:
        return {commutator[1] + commutator[0]: -1}

    a = commutator[0]
    b = commutator[1][0]
    c = commutator[1][1:]

    comm_bca = {(b,) + comm: -coeff for comm, coeff in _right_nest_two_comms((c, a)).items()}
    comm_cab = {comm: -coeff for comm, coeff in _right_nest_two_comms(((b,) + a, c)).items()}

    commutators = defaultdict(int)

    for comm, coeff in comm_bca.items():
        commutators[comm] += coeff

    for comm, coeff in comm_cab.items():
        commutators[comm] += coeff

    return commutators


def _drop_zeros(term_dicts: List[Dict[Tuple[int], float]]) -> List[Dict[Tuple[int], float]]:
    """Remove any terms whose coefficient is close to zero"""
    for terms in term_dicts:
        delete = []
        for commutator, coeff in terms.items():
            if np.isclose(coeff, 0):
                delete.append(commutator)

        for commutator in delete:
            del terms[commutator]

    return term_dicts

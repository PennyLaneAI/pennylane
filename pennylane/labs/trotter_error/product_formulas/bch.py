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
from collections.abc import Generator, Hashable, Sequence
from functools import cache
from itertools import permutations, product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pennylane.labs.trotter_error import ProductFormula


def bch_expansion(
    product_formula: ProductFormula, order: int
) -> list[dict[tuple[Hashable], complex]]:
    r"""Compute the Baker-Campbell-Hausdorff expansion of a :class:`~.pennylane.labs.trotter_error.ProductFormula` object.

    Args:
        product_formula (ProductFormula): The :class:`~.pennylane.labs.trotter_error.ProductFormula` object whose BCH expansion will be computed.
        order (int): The maximum order of the expansion to return.
    Returns:
        List[Dict[Tuple[Hashable], complex]]: A list of dictionaries. The ``ith`` dictionary contains the ``ith`` order commutators and their coefficients.

    **Example**

    In this example we compute the BCH expansion of the second order Trotter-Suzuki formula. The output is a list of dictionaries where each dictionary is indexed by
    a tuple representing a right-nested commutator. For example, ``('A', 'A', 'B')`` represents the commutator :math:`[A, [A, B]]`.

    >>> from pprint import pp
    >>> from pennylane.labs.trotter_error import ProductFormula, bch_expansion
    >>> frag_labels = ["A", "B", "C", "B", "A"]
    >>> frag_coeffs = [1/2, 1/2, 1, 1/2, 1/2]
    >>> second_order = ProductFormula(frag_labels, frag_coeffs)
    >>> pp(bch_expansion(second_order, order=3))
    [defaultdict(<class 'complex'>,
                 {('A',): (1+0j),
                  ('B',): (1+0j),
                  ('C',): (1+0j)}),
     defaultdict(<class 'complex'>, {}),
     defaultdict(<class 'complex'>,
                 {('A', 'A', 'B'): (-0.04166666666666667+0j),
                  ('B', 'A', 'B'): (-0.08333333333333333+0j),
                  ('C', 'A', 'B'): (-0.08333333333333334+0j),
                  ('A', 'A', 'C'): (-0.04166666666666667+0j),
                  ('B', 'A', 'C'): (-0.08333333333333334+0j),
                  ('B', 'B', 'C'): (-0.04166666666666667+0j),
                  ('C', 'A', 'C'): (-0.08333333333333333+0j),
                  ('C', 'B', 'C'): (-0.08333333333333333+0j)})]
    """
    return _drop_zeros(_bch_expansion(product_formula, order, {}))


def _bch_expansion(
    product_formula: ProductFormula, order: int, term_dict: dict[tuple[Hashable], complex]
) -> list[dict[tuple[Hashable], complex]]:
    """Recursively applies BCH to the product formula. The terms of ProductFormula objects are either
    hashable labels for fragments, or ProductFormula objects. The hashable labels are the base case,
    and the ProductFormula objects are the recursive case.

     Base case:
         Directly compute BCH on the fragment labels and return the result

     Recursive case:
         1. Compute BCH on the ProductFormula objects as if they were the labels for fragments.
         2. Call _bch_expansion recursively on each ProductFormula object
         5. Merge the results in the following way:
             If (PF1, PF2) is a commutator obtained from step 1, substitute the labels PF1 and PF2
             with the commutators returned by calling _bch_expansion on PF1 and PF2
         4. Simplify the commutators and return
    """

    bch = _bch(
        product_formula.terms,
        product_formula.coeffs,
        product_formula.ordered_terms,
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
        _apply_exponent(merged_bch, product_formula.exponent), product_formula.ordered_fragments
    )


def _bch(
    fragments: Sequence[Hashable],
    coeffs: Sequence[complex],
    term_order: dict[Hashable, int],
    order: int,
) -> list[dict[tuple[Hashable], complex]]:
    """Computes BCH on a list of labels by recursively applying BCH to the head and tail of the list.
    For a list [A, B, C] we compute BCH(A, BCH(B, C)). This is done with the following steps.

    1. Set head = BCH(A) = A
    2. Set tail = BCH(B, C)
    3. Compute BCH(head, tail)
    4. Subtitute A for head, and BCH(B, C) for tail in BCH(head, tail)
    """

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
    """Scale the coefficients in the dictionary by the exponent"""
    for i, ith_order_commutators in enumerate(bch):
        for commutator, coeff in ith_order_commutators.items():
            bch[i][commutator] = coeff * exponent

    return bch


def _kth_order_terms(
    fragments: Sequence[Hashable], coeffs: Sequence[complex], k: int
) -> dict[tuple[Hashable], complex]:
    r"""Computes the kth order commutators of the BCH expansion of the product formula.
    See Proposition 1 of `arXiv:2006.15869 <https://arxiv.org/pdf/2006.15869>`."""

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
    """Substitute the labels in a commutator with the BCH expansion of the label, but only include
    commutators whose order is at most ``order``"""

    merged = [defaultdict(complex) for _ in range(order)]

    for x in _generate_merged_commutators(commutator, terms, order):
        new_commutator = tuple(y[0] for y in x)
        term_coeff = math.prod(y[1] for y in x) * bch_coeff
        commutator_order = _commutator_order(new_commutator)

        merged[commutator_order - 1][new_commutator] += term_coeff

    return merged


def _generate_merged_commutators(commutator, terms, order):
    r"""Yields the commutators obtained by substituting the commutators in ``terms`` with the labels
    in ``commutator`` whose order is at most ``order``. The algorithm is based on the identity
    [\sum_i A_i, [\sum_j B_j, \sum_k C_k]]] = \sum_{i,j,k} [A_k, [B_j, C_k]].

    """

    for i in range(order):
        for partition in _partitions_positive(len(commutator), i + 1):
            kept_terms = [
                terms[term][partition[j] - 1].items() for j, term in enumerate(commutator)
            ]
            yield from product(*kept_terms)


def _commutator_order(commutator):
    """Returns the order of the commutator"""
    order = 0

    for term in commutator:
        if isinstance(term, tuple):
            order += _commutator_order(term)
        else:
            order += 1

    return order


def _add_dicts(d1, d2):
    """Add two defaultdicts"""
    for key, value in d2.items():
        d1[key] += value

    return d1


def _partitions_nonnegative(m: int, n: int) -> Generator[tuple[int]]:
    """Yields tuples of m nonnegative integers that sum to n"""

    if m == 1:
        yield (n,)
    elif m == 0:
        yield (0,) * m
    else:
        for i in range(n + 1):
            for partition in _partitions_nonnegative(m - 1, n - i):
                yield (i,) + partition


def _partitions_positive(m: int, n: int) -> Generator[tuple[int]]:
    """Yields tuples of m positive integers that sum to n"""

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


def _phi(fragments: Sequence[Hashable]) -> dict[tuple[Hashable], float]:
    """Implements equation 13 from `arXiv:2006.15869 <https://arxiv.org/pdf/2006.15869>`."""
    n = len(fragments)
    terms = defaultdict(complex)

    for permutation in permutations(range(n - 1)):
        d = _n_descents(permutation)
        commutator = tuple(fragments[i] for i in permutation) + (fragments[n - 1],)
        terms[commutator] += ((-1) ** d) / n / math.comb(n - 1, d)

    return terms


def _n_descents(permutation: Sequence[Hashable]) -> int:
    """Returns the number of descents in a permutation. For a permutation sigma, a descent in sigma
    is a pair i,i+1 such that sigma(i) > sigma(i+1)."""
    n = 0

    for i in range(len(permutation) - 1):
        if permutation[i] > permutation[i + 1]:
            n += 1

    return n


def _remove_redundancies(
    term_dicts: list[dict[tuple[Hashable], float]],
    term_order: dict[Hashable, int],
) -> list[dict[tuple[Hashable], float]]:
    """Applies the following identities to the commutators

    1. Express the commutator as a linear combination of right-nested commutators
    2. Apply [A, A] = 0
    3. Apply [A, B] = -[A, B]
    4. Apply [A, B, B, A] = [B, A, B, A]

    A derivation of identity 4 can be found in the appendix of `arXiv:2006.15869 <https://arxiv.org/pdf/2006.15869>`.
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
        _commute_with_self(terms)
        _antisymmetry(terms, less_than)

    if max_order < 4:
        return term_dicts

    for terms in term_dicts[3:]:
        _fourth_order_simplification(terms)

    return term_dicts


def _antisymmetry(terms, less_than):
    """Apply the identity [A, B] = -[B, A]"""
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


def _commute_with_self(terms):
    """Apply the identity [A, A] = 0"""
    delete = []
    for commutator in terms.keys():
        if commutator[-1] == commutator[-2]:
            delete.append(commutator)

    for commutator in delete:
        del terms[commutator]


def _fourth_order_simplification(terms):
    """Apply the identity [A, B, B, A] = [B, A, B, A]"""
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


def _make_right_nested(terms):
    """Transforms the commutators into linear combinations of right-nested commutators"""
    ret = defaultdict(complex)

    for commutator1, coeff1 in terms.items():
        for commutator2, coeff2 in _right_nested(commutator1).items():
            ret[commutator2] += coeff1 * coeff2

    return ret


@cache
def _right_nested(commutator: tuple[tuple | Hashable]) -> dict[tuple[Hashable], float]:
    """Express the commutator as a linear combation of right-nested commutators.

    Find the i such that commutaor[i] is a tuple representing a commutator, and commutator[j] is a hashable
    label for each j > i. Then the commutator has the form [..., [[X_1,...,X_n], [Y_1,...,Y_m]]]
    and the commutator of nested commutators [[X_1,...,X_n], [Y_1,...,Y_m]] is transformed into a linear
    combination of right-nested commutators by calling _right_nest_two_comms. Repeat the process recursively.
    """
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

    two_comms = (commutator[i], commutator[i + 1 :])
    partially_nested = {
        commutator[:i] + nested: coeff for nested, coeff in _right_nest_two_comms(two_comms).items()
    }

    ret = defaultdict(complex)

    for partial, coeff1 in partially_nested.items():
        for nested, coeff2 in _right_nested(partial).items():
            ret[nested] += coeff1 * coeff2

    return ret


@cache
def _right_nest_two_comms(commutator: tuple[tuple | Hashable]) -> dict[tuple[Hashable], float]:
    """Express a commutator of two right-nested commutators [[X_1, ..., X_n], [Y_1, ..., Y_m]] as a
    linear combination of right-nested commutators [Z_1, ..., Z_{n+m}].

    Set A = [X_1, ..., X_n], B = Y_1, and C = [Y_2, ..., Y_m] and apply the Jacobi identity [A, B, C] = [B, A, C] - [C, A, B]
    to obtain [Y_1, [[X_1, ..., X_n], [Y_2, ..., Y_m]] - [[Y_1, X_1, ..., X_n], [Y_2, ..., Y_m]] and recurse on both commutators.
    """

    if len(commutator[0]) == 0:
        return _right_nested(commutator[1])

    if len(commutator[1]) == 0:
        return _right_nested(commutator[0])

    if len(commutator[0]) == 1:
        return {commutator[0] + commutator[1]: 1}

    if len(commutator[1]) == 1:
        return {commutator[1] + commutator[0]: -1}

    a = commutator[0]
    b = commutator[1][0]
    c = commutator[1][1:]

    comm_bac = {(b,) + comm: coeff for comm, coeff in _right_nest_two_comms((a, c)).items()}
    comm_cab = {comm: -coeff for comm, coeff in _right_nest_two_comms(((b,) + a, c)).items()}

    commutators = defaultdict(int)

    for comm, coeff in comm_bac.items():
        commutators[comm] += coeff

    for comm, coeff in comm_cab.items():
        commutators[comm] += coeff

    return commutators


def _drop_zeros(
    term_dicts: list[dict[tuple[Hashable], complex]],
) -> list[dict[tuple[Hashable], float]]:
    """Remove any terms whose coefficient is close to zero"""
    for terms in term_dicts:
        delete = []
        for commutator, coeff in terms.items():
            if np.isclose(coeff, 0):
                delete.append(commutator)

        for commutator in delete:
            del terms[commutator]

    return term_dicts

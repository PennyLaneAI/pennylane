# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Baker--Campbell--Hausdorff (BCH) expansion of a product formula.

This module computes the BCH expansion of a :class:`~.ProductFormula` in the
Hall basis of the free Lie algebra. The BCH formula expresses the product of
exponentials as a single exponential of a sum of nested commutators,

.. math::

    e^{A} e^{B} = \exp\!\Big(A + B + \tfrac{1}{2}[A, B]
        + \tfrac{1}{12}\big([A, [A, B]] + [B, [B, A]]\big) + \cdots\Big).

The Trotter error of a product formula is given by the commutator terms.

The implementation is recursive. A product formula is reduced to a sequence of
symbols, and the expansion of that sequence is built up by repeatedly combining
a single symbol (the *head*) with the expansion of the remainder (the *tail*).
The base case, the BCH expansion of two symbols, read from a precomputed lookup
table of Hall-basis commutators and their rational coefficients (see :func:`_bch`).
The lookup tables were obtained from Tables 3 and 4 `Casas and Murua (2008) arXiv:0810.2656  <https://arxiv.org/abs/0810.2656>`_

Recursive product formulas are expanded by first expanding each sub-formula and then substituting
those expansions into the outer expansion via a bilinear substitution.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pennylane.labs.trotter_error.product_formulas.commutator import (
    ASTNode,
    CommutatorNode,
    SymbolNode,
    bilinear_expansion,
    is_mergeable,
    is_tree_isomorphic,
    merge,
)

if TYPE_CHECKING:
    from pennylane.labs.trotter_error.product_formulas.product_formula import ProductFormula


def bch_expansion(product_formula: ProductFormula, max_order: int) -> dict[ASTNode, float]:
    r"""Return the Baker--Campbell--Hausdorff (BCH) expansion of a product formula.

    Computes the single-exponential generator equivalent to the given product
    formula, expanded in the Hall basis and truncated at ``max_order``.

    Args:
        product_formula (ProductFormula): the product formula to expand. May be a
            flat formula or a recursive composition of sub-formulas.
        max_order (int): the maximum commutator order (nesting depth) to retain
            in the expansion.

    Returns:
        dict[ASTNode, float]: a mapping from each retained Hall-basis commutator
        to its (nonzero) coefficient.
    """
    return _drop_zeros(_bch_product_formula(product_formula, max_order, {}))


def _bch_product_formula(
    product_formula: ProductFormula,
    max_order: int,
    term_dict: dict[ProductFormula, dict[ASTNode, float]],
) -> dict[ASTNode, float]:
    r"""Compute the BCH expansion of a (possibly recursive) product formula.

    The formula is first reduced to a sequence of ``(symbol, coefficient)``
    factors whose BCH expansion is computed by :func:`_bch_symbols`.
    For a non-recursive formula, the symbol expansion is scaled by the
    formula's ``exponent`` and returned directly.

    For a recursive formula, each sub-formula is expanded once, and those
    sub-expansions are substituted into the outer expansion via
    :func:`bilinear_expansion`.

    Args:
        product_formula (ProductFormula): the formula to expand.
        max_order (int): the maximum commutator order to retain.
        term_dict (dict[ProductFormula, dict[ASTNode, float]]): a memoization
            cache mapping each already-expanded sub-formula symbol to its scaled
            expansion. Mutated in place to share work across the recursion.

    Returns:
        dict[ASTNode, float]: the BCH expansion of the formula,
        scaled by its exponent.
    """

    if product_formula.is_symmetric:
        k = math.ceil(len(product_formula.symbols) / 2)
        symbols = product_formula.symbols[0:k]

        if len(product_formula.symbols) % 2 == 0:
            coeffs = [2 * coeff for coeff in product_formula.coeffs[0:k]]
        else:
            coeffs = [2 * coeff for coeff in product_formula.coeffs[0 : k - 1]] + [
                product_formula.coeffs[k - 1]
            ]

    else:
        symbols = product_formula.symbols
        coeffs = product_formula.coeffs

    bch = _bch_symbols(
        [SymbolNode(symbol, coeff) for symbol, coeff in zip(symbols, coeffs)],
        max_order,
        product_formula.is_symmetric,
    )

    if not product_formula.is_recursive:
        return _scale_dict(bch, product_formula.exponent)

    for pf, coeff in zip(symbols, coeffs):
        symbol = SymbolNode(pf, coeff)
        if symbol in term_dict:
            continue
        term_dict[SymbolNode(pf, coeff)] = _scale_dict(
            _bch_product_formula(pf, max_order, term_dict), coeff
        )

    ret = defaultdict(float)
    for comm1, coeff1 in bch.items():
        comm_dict = {
            leaf: _add_dicts([term_dict[x] for x in leaf.split()]) for leaf in comm1.leaves()
        }
        for comm2, coeff2 in bilinear_expansion(comm1, comm_dict, max_order, coeff1).items():
            ret[comm2] += coeff2

    ret = _cancel_terms(ret)
    ret = _group_terms(ret, max_order)
    ret = _scale_dict(ret, product_formula.exponent)

    return ret


def _bch_symbols(
    symbols: Sequence[SymbolNode], max_order: int, symmetric: bool
) -> dict[ASTNode, float]:
    r"""Compute the BCH expansion of a sequence of symbols.

    Expands :math:`e^{s_0} e^{s_1} \cdots e^{s_{n-1}}` into Hall-basis
    commutators. The recursion peels off the first symbol as the *head* and
    treats the expansion of the remaining symbols as the *tail*:

    * a single symbol expands to itself with coefficient one;
    * two symbols are looked up directly from the precomputed table via
      :func:`_bch`;
    * for more than two symbols, the two-element expansion of ``(head, tail)``
      is computed and the placeholder head/tail leaves are substituted, via
      :func:`bilinear_expansion`, by the head symbol and the recursively
      expanded tail.

    After substitution the result is normalized at order one, has vanishing
    terms cancelled, and has tree-isomorphic terms grouped/merged.

    Args:
        symbols (Sequence[SymbolNode]): the ordered symbols to expand.
        max_order (int): the maximum commutator order to retain.
        symmetric (bool): whether to use the symmetric coefficient table (and
            its associated cancellation of even-order terms).

    Returns:
        dict[ASTNode, float]: the BCH expansion of the symbol sequence.
    """
    if len(symbols) == 1:
        return {symbols[0]: 1}

    if len(symbols) == 2:
        return _bch(symbols[0], symbols[1], max_order, symmetric)

    head = SymbolNode("Head")
    tail = SymbolNode("Tail")

    terms = {
        head: {symbols[0]: 1.0},
        tail: _bch_symbols(symbols[1:], max_order, symmetric),
    }

    bch_ht = _bch(head, tail, max_order, symmetric)
    bch_final = defaultdict(float)
    for comm1, coeff1 in bch_ht.items():
        for comm2, coeff2 in bilinear_expansion(comm1, terms, max_order, coeff1).items():
            bch_final[comm2] += coeff2

    bch_final = _cancel_terms(bch_final)
    bch_final = _group_terms(bch_final, max_order)

    return bch_final


def _bch(x: SymbolNode, y: SymbolNode, max_order: int, symmetric: bool) -> dict[ASTNode, float]:
    r"""Read the two-symbol BCH expansion of ``x`` and ``y`` from a lookup table.

    This is the base case of the recursion. The Hall-basis commutators and their
    rational coefficients are precomputed and stored in a NumPy archive next to
    this file: ``sbch.npz`` for the symmetric expansion and ``bch.npz`` for the
    general one. The precomputed data was obtained from Tables 3 and 4 of Casas and Murua (2008) arXiv:0810.2656  <https://arxiv.org/abs/0810.2656>_.
    The table contains:

    * ``l_comm`` / ``r_comm``: 1-based indices into the running list of Hall
      commutators, identifying the left and right operands of each new
      commutator;
    * ``p`` / ``q``: the integer numerator and denominator of the rational
      coefficient ``p / q`` for that commutator.

    Starting from ``[x, y]`` (each with coefficient one), each subsequent row
    builds a new :class:`CommutatorNode` from two previously constructed Hall
    commutators and appends it with its coefficient. The first two rows of the
    table are skipped because they correspond to ``x`` and ``y`` themselves.
    Construction stops as soon as a commutator would exceed ``max_order``
    (the table is ordered by nondecreasing commutator order).

    Args:
        x (SymbolNode): the first (left) symbol.
        y (SymbolNode): the second (right) symbol.
        max_order (int): the maximum commutator order to retain.
        symmetric (bool): if ``True``, read from the symmetric table
            (``sbch.npz``); otherwise read from ``bch.npz``.

    Returns:
        dict[ASTNode, float]: a mapping from each Hall-basis commutator (up to
        ``max_order``) to its coefficient, including ``x`` and ``y`` at order
        one.
    """
    hall_commutators = [x, y]
    bch_coeffs = [1, 1]

    rows = zip(*load_hall_basis(symmetric))

    for l_comm, r_comm, p, q in rows:
        left, right = int(l_comm), int(r_comm)
        bch_coeff = int(p) / int(q)

        left = copy.deepcopy(hall_commutators[left - 1])
        right = copy.deepcopy(hall_commutators[right - 1])

        basis_element = CommutatorNode(left, right)

        if basis_element.order > max_order:
            break

        hall_commutators.append(basis_element)
        bch_coeffs.append(bch_coeff)

    return dict(zip(hall_commutators, bch_coeffs))


def _scale_dict(d: dict, k: float) -> dict:
    """Multiply every value of ``d`` by ``k`` in place.

    Args:
        d (dict): the dictionary whose values are scaled. Mutated in place.
        k (float): the scalar multiplier.

    Returns:
        dict: the same dictionary ``d``, with each value scaled by ``k``.
    """

    new_dict = {}

    for key in d:
        new_dict[key] = k * d[key]

    return new_dict


def _add_dicts(ds: Sequence[dict]) -> dict:
    """Sum a sequence of dictionaries key-wise.

    Args:
        ds (Sequence[dict]): the dictionaries to combine. Values are summed for
            keys that appear in more than one dictionary.

    Returns:
        dict: a ``defaultdict(float)`` mapping each key to the sum of its values
        across all inputs.
    """
    ret = defaultdict(float)
    for d in ds:
        for k, v in d.items():
            ret[k] += v

    return ret


def _drop_zeros(d: dict) -> dict:
    """Remove entries whose value is numerically close to zero.

    Args:
        d (dict): the dictionary to prune.

    Returns:
        dict: a new dictionary containing only the entries whose absolute value
        is not close to zero (per ``numpy.isclose``).
    """
    return {key: value for key, value in d.items() if not np.isclose(np.abs(value), 0)}


def _cancel_terms(d: dict[ASTNode, float]) -> dict[ASTNode, float]:
    """Remove commutators that are identically zero.

    Drops any key whose ``is_zero()`` method reports the commutator as
    structurally vanishing (e.g. a commutator of an operand with itself).

    Args:
        d (dict[ASTNode, float]): the expansion to filter.

    Returns:
        dict[ASTNode, float]: a new dictionary without the zero commutators.
    """
    return {key: value for key, value in d.items() if not key.is_zero()}


def _group_terms(d: dict[ASTNode, float], max_order: int) -> dict[ASTNode, float]:
    """Group tree-isomorphic commutators with equal coefficients and merge them.

    Commutators that are tree-isomorphic (same nesting structure, see
    :func:`is_tree_isomorphic`) and that carry numerically equal coefficients
    are collected into groups. Within each group, the constituent commutators
    are merged order by order, up to ``max_order``, using :func:`is_mergeable`
    and :func:`merge`; the shared coefficient is then assigned to each merged
    representative.

    Args:
        d (dict[ASTNode, float]): the expansion to group and merge.
        max_order (int): the number of merge passes / maximum order considered
            when merging.

    Returns:
        dict[ASTNode, float]: the expansion with isomorphic, equal-coefficient
        terms merged.
    """
    groups = []
    coeffs = []

    for comm, coeff1 in d.items():
        found = False
        for group, coeff2 in zip(groups, coeffs):
            rep = group[0]
            if is_tree_isomorphic(comm, rep) and np.isclose(coeff1, coeff2):
                group.append(comm)
                found = True
                break

        if not found:
            groups.append([comm])
            coeffs.append(coeff1)

    ret = defaultdict(float)
    for group, coeff in zip(groups, coeffs):
        unmerged = set(group)
        for k in range(max_order):
            merged = set()
            for comm1 in unmerged:
                comm2 = next((c for c in merged if is_mergeable(comm1, c, k)), None)

                if comm2 is not None:
                    new_comm = merge(comm1, comm2, k)
                    merged.add(new_comm)
                    merged.remove(comm2)
                else:
                    merged.add(comm1)

            unmerged = copy.deepcopy(merged)

        for comm in merged:
            ret[comm] += coeff

    return ret


@cache
def load_hall_basis(symmetric: bool):
    """Load the Hall basis data from a precomputed numpy archive"""

    file = "sbch.npz" if symmetric else "bch.npz"
    filepath = Path(__file__).parent / file

    with np.load(filepath) as bch:
        l_comms = bch["l_comm"][2:]
        r_comms = bch["r_comm"][2:]
        ps = bch["p"][2:]
        qs = bch["q"][2:]

    return l_comms, r_comms, ps, qs

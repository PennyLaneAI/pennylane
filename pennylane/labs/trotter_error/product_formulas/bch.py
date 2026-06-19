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
"""This file contains the BCH computation"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Sequence
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
    """Return the Baker-Campbell-Hausdorff (BCH) expansion."""
    return _drop_zeros(_bch_product_formula(product_formula, max_order, {}))


def _bch_product_formula(
    product_formula: ProductFormula,
    max_order: int,
    term_dict: dict[ProductFormula, dict[ASTNode, float]],
) -> dict[ASTNode, float]:
    """Apply BCH to a product formula"""

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
    """Apply BCH to a sequence of symbols"""
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
    """Reads the BCH expansion of `x` and `y` from a precomputed lookup table"""
    hall_commutators = [x, y]
    bch_coeffs = [1, 1]

    file = "sbch.npz" if symmetric else "bch.npz"
    filepath = Path(__file__).parent / file

    with np.load(filepath) as bch:

        l_comms = bch["l_comm"][2:]
        r_comms = bch["r_comm"][2:]
        ps = bch["p"][2:]
        qs = bch["q"][2:]

        rows = zip(l_comms, r_comms, ps, qs)

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
    for key in d:
        d[key] *= k

    return d


def _add_dicts(ds: Sequence[dict]) -> dict:
    ret = defaultdict(float)
    for d in ds:
        for k, v in d.items():
            ret[k] += v

    return ret


def _drop_zeros(d: dict) -> dict:
    """Remove terms close to zero"""
    return {key: value for key, value in d.items() if not np.isclose(np.abs(value), 0)}


def _cancel_terms(d: dict[ASTNode, float]) -> dict[ASTNode, float]:
    return {key: value for key, value in d.items() if not key.is_zero()}


def _group_terms(d: dict[ASTNode, float], max_order: int) -> dict[ASTNode, float]:
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

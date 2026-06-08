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

import copy
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Sequence

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


def bch_expansion(product_formula: ProductFormula, max_order: int) -> Dict[ASTNode, float]:
    """Return the BCH expansion"""
    return _drop_zeros(_bch_product_formula(product_formula, max_order, {}))


def _bch_product_formula(
    product_formula: ProductFormula,
    max_order: int,
    term_dict: Dict[ProductFormula, Dict[ASTNode, float]],
) -> Dict[ASTNode, float]:
    """Apply BCH to a product formula"""

    if product_formula.is_symmetric:
        k = math.ceil(len(product_formula.symbols) / 2)
        symbols = product_formula.symbols[0:k]
        coeffs = [2 * coeff for coeff in product_formula.coeffs[0 : k - 1]]

        if len(product_formula.symbols) % 2 == 0:
            coeffs.append(2 * product_formula.coeffs[k - 1])
        else:
            coeffs.append(product_formula.coeffs[k - 1])

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
        if pf in term_dict:
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

    ret = _normalize_order_1(ret)
    ret = _cancel_terms(ret)
    ret = _group_terms(ret, max_order)
    ret = _scale_dict(ret, product_formula.exponent)

    return ret


def _bch_symbols(
    symbols: Sequence[SymbolNode], max_order: int, symmetric: bool
) -> Dict[ASTNode, float]:
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

    bch_final = _normalize_order_1(bch_final)
    bch_final = _cancel_terms(bch_final)
    bch_final = _group_terms(bch_final, max_order)

    return bch_final


def _bch(x: SymbolNode, y: SymbolNode, max_order: int, symmetric: bool) -> Dict[ASTNode, float]:
    hall_commutators = [x, y]
    bch_coeffs = [1, 1]

    filepath = (
        Path(__file__).parent / "sbch.csv" if symmetric else Path(__file__).parent / "bch.csv"
    )

    with open(filepath, "r") as csv_file:
        bch_reader = csv.reader(csv_file)

        next(bch_reader)
        next(bch_reader)

        for row in bch_reader:
            left, right = int(row[1]), int(row[2])
            bch_coeff = int(row[3]) / int(row[4])

            left = copy.deepcopy(hall_commutators[left - 1])
            right = copy.deepcopy(hall_commutators[right - 1])

            basis_element = CommutatorNode(left, right)

            if basis_element.order > max_order:
                break

            hall_commutators.append(basis_element)
            bch_coeffs.append(bch_coeff)

    return dict(zip(hall_commutators, bch_coeffs))


def _scale_dict(d: Dict, k: float) -> Dict:
    for key in d:
        d[key] *= k

    return d


def _add_dicts(ds: Sequence[Dict]) -> Dict:
    ret = defaultdict(float)
    for d in ds:
        for k, v in d.items():
            ret[k] += v

    return ret


def _drop_zeros(d: Dict) -> Dict:
    """Remove terms close to zero"""
    return {key: value for key, value in d.items() if not np.isclose(np.abs(value), 0)}


def _cancel_terms(d: Dict[ASTNode, float]) -> Dict[ASTNode, float]:
    return {key: value for key, value in d.items() if not key.is_zero()}


def _group_terms(d: Dict[ASTNode, float], max_order: int) -> Dict[ASTNode, float]:
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
                is_merged = False
                for comm2 in merged:
                    if is_mergeable(comm1, comm2, k):
                        new_comm = merge(comm1, comm2, k)
                        is_merged = True
                        break

                if is_merged:
                    merged.add(new_comm)
                    merged.remove(comm2)
                else:
                    merged.add(comm1)

            unmerged = copy.deepcopy(merged)

        for comm in merged:
            ret[comm] += coeff

    return ret


def _normalize_order_1(d: dict[ASTNode, float]) -> dict[ASTNode, float]:
    order_1 = {comm: coeff for comm, coeff in d.items() if comm.order == 1}
    ret = defaultdict(float, {comm: coeff for comm, coeff in d.items() if comm.order > 1})

    all_symbols = defaultdict(float)
    for lincomb, coeff in order_1.items():
        for symbol, sym_coeff in lincomb.symbols:
            all_symbols[symbol] += coeff * sym_coeff

    symbols, coeffs = zip(*all_symbols.items())
    new_order_1_comm = SymbolNode(symbols, coeffs)
    ret[new_order_1_comm] = 1

    return ret

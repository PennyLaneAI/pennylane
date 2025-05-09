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
"""Functions for retreiving effective error from fragments"""

import math
from collections import defaultdict
from collections.abc import Hashable
from itertools import product
from typing import Dict, List, Sequence

from pennylane.labs.trotter_error import AbstractState, Fragment
from pennylane.labs.trotter_error.abstract import nested_commutator
from pennylane.labs.trotter_error.product_formulas.product_formula import ProductFormula


class _AdditiveIdentity:
    """Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other


def effective_hamiltonian(
    product_formula: ProductFormula, fragments: Dict[Hashable, Fragment], order: int = 3
):
    """Compute the effective Hamiltonian according to the product formula

    **Example**

    >>> import numpy as np
    >>> from pennylane.labs.trotter_error.fragments import vibrational_fragments
    >>> from pennylane.labs.trotter_error.product_formulas import ProductFormula, effective_hamiltonian

    >>> n_modes = 4
    >>> r_state = np.random.RandomState(42)
    >>> freqs = r_state.random(4)
    >>> taylor_coeffs = [
    >>>     np.array(0),
    >>>     r_state.random(size=(n_modes, )),
    >>>     r_state.random(size=(n_modes, n_modes)),
    >>>     r_state.random(size=(n_modes, n_modes, n_modes))
    >>> ]
    >>>
    >>> delta = 0.001
    >>> frag_labels = [0, 1, 1, 0]
    >>> frag_coeffs = [delta/2, delta/2, delta/2, delta/2]

    >>> pf = ProductFormula(frag_labels, coeffs=frag_coeffs)
    >>> frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))
    >>> type(effective_hamiltonian(pf, frags, order=5))
    <class 'pennylane.labs.trotter_error.realspace.realspace_operator.RealspaceSum'>
    """

    if not product_formula.fragments.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")

    bch = _recursive_bch(product_formula, order)
    eff = _AdditiveIdentity()

    for commutator, coeff in bch.items():
        eff += coeff * nested_commutator(_insert_fragments(commutator, fragments))

    return eff


def _insert_fragments(commutator, fragments):
    ret = tuple()
    for term in commutator:
        if isinstance(term, tuple):
            ret += (_insert_fragments(term, fragments),)
        else:
            ret += (fragments[term],)

    return ret


def _recursive_bch(product_formula: ProductFormula, order: int = 3):

    bch = {}
    for commutator_order in product_formula.bch_approx(order):
        for commutator, coeff in commutator_order.items():
            bch[commutator] = product_formula.exponent * coeff

    if not product_formula.recursive:
        return bch

    terms = {pf: _recursive_bch(pf, order) for pf in product_formula.terms}

    merged_bch = defaultdict(complex)
    for commutator, coeff in bch.items():
        merged_bch = _add_dicts(merged_bch, _merge_commutators(commutator, terms, order, coeff))

    return merged_bch


def _merge_commutators(commutator, terms, order, bch_coeff):
    commutator_terms = [terms[term].items() for term in commutator]

    merged = defaultdict(complex)

    if len(commutator_terms) > order:
        return merged

    for x in product(*commutator_terms):
        if len(x) > 1 and x[-1][0] == x[-2][0]:
            continue

        new_commutator = x[0][0] if len(x) == 1 else tuple(_flatten_commutator(y[0]) for y in x)

        if _commutator_order(new_commutator) > order:
            continue
        term_coeff = math.prod(y[1] for y in x)

        merged[new_commutator] += bch_coeff * term_coeff

    return merged


def _flatten_commutator(commutator):
    if isinstance(commutator, tuple) and len(commutator) == 1:
        return commutator[0]

    return commutator


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


def perturbation_error(
    product_formula: ProductFormula,
    fragments: Sequence[Fragment],
    states: Sequence[AbstractState],
    order: int = 3,
) -> List[float]:
    r"""Computes the perturbation theory error using the second-order Trotter error operator.

    The second-order Trotter error operator, :math:`\hat{\epsilon}`, is given by the expression

    .. math:: \hat{\epsilon} = \frac{- \Delta t^2}{24} \sum_{i=1}^{L-1} \sum_{j = i + 1}^L \left[ H_i + 2 \sum_{k = j + 1}^L H_k, \left[ H_i, H_j \right] \right].

    For a state :math:`\left| \psi \right\rangle` the perturbation theory error is given by the expectation value :math:`\left\langle \psi \right| \hat{\epsilon} \left| \psi \right\rangle`.

    Args:
        fragments (Sequence[Fragments]): the set of :class:`~.pennylane.labs.trotter_error.Fragment`
            objects to compute Trotter error from
        states: (Sequence[AbstractState]): the states to compute expectation values from
        delta (float): time step for the trotter error operator.

    Returns:
        List[float]: the list of expectation values computed from the Trotter error operator and the input states

    **Example**

    >>> import numpy as np
    >>> from pennylane.labs.trotter_error import HOState, ProductFormula, vibrational_fragments, perturbation_error

    >>> frag_labels = [0, 1, 1, 0]
    >>> frag_coeffs = [1/2, 1/2, 1/2, 1/2]
    >>> pf = ProductFormula(frag_labels, coeffs=frag_coeffs)

    >>> n_modes = 2
    >>> r_state = np.random.RandomState(42)
    >>> freqs = r_state.random(n_modes)
    >>> taylor_coeffs = [
    >>>     np.array(0),
    >>>     r_state.random(size=(n_modes, )),
    >>>     r_state.random(size=(n_modes, n_modes)),
    >>>     r_state.random(size=(n_modes, n_modes, n_modes))
    >>> ]
    >>> frags = dict(enumerate(vibrational_fragments(n_modes, freqs, taylor_coeffs)))

    >>> gridpoints = 5
    >>> state1 = HOState(n_modes, gridpoints, {(0, 0): 1})
    >>> state2 = HOState(n_modes, gridpoints, {(1, 1): 1})

    >>> errors = perturbation_error(pf, frags, [state1, state2])
    [(-0.9189251160920879+0j), (-4.797716682426851+0j)]
    """

    eff = effective_hamiltonian(product_formula, fragments, order=order)
    error = eff - sum(fragments.values(), _AdditiveIdentity())

    return [error.expectation(state, state) for state in states]

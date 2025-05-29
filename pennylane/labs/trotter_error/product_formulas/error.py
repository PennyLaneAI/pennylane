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

import copy
from collections import defaultdict
from collections.abc import Hashable
from typing import Dict, List, Sequence, Tuple

from pennylane.labs.trotter_error import AbstractState, Fragment
from pennylane.labs.trotter_error.abstract import nested_commutator
from pennylane.labs.trotter_error.product_formulas.bch import bch_expansion
from pennylane.labs.trotter_error.product_formulas.product_formula import ProductFormula


class _AdditiveIdentity:
    """Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other


def effective_hamiltonian(
        product_formula: ProductFormula, fragments: Dict[Hashable, Fragment], order: int, timestep: float = 1.0
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

    bch = bch_expansion(product_formula(timestep), order)
    eff = _AdditiveIdentity()

    for ith_order in bch:
        for commutator, coeff in ith_order.items():
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


def perturbation_error(
    product_formula: ProductFormula,
    fragments: Dict[Hashable, Fragment],
    states: Sequence[AbstractState],
    order: int,
    timestep: float = 1.0
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

    if not product_formula.fragments.issubset(fragments.keys()):
        raise ValueError("Fragments do not match product formula")

    bch = bch_expansion(product_formula(timestep), order)
    commutators = {
        commutator: coeff for comm_dict in bch[1:] for commutator, coeff in comm_dict.items()
    }

    expectations = []
    for state in states:
        new_state = _AdditiveIdentity()
        for commutator, coeff in commutators.items():
            new_state += coeff * _apply_commutator(commutator, fragments, state)

        expectations.append(state.dot(new_state))

    return expectations


def _apply_commutator(
    commutator: Tuple[Hashable], fragments: Dict[Hashable, Fragment], state: AbstractState
) -> AbstractState:

    new_state = _AdditiveIdentity()

    for term, coeff in _op_list(commutator).items():
        tmp_state = copy.copy(state)
        for frag in reversed([fragments[x] for x in term]):
            tmp_state = frag.apply(tmp_state)

        new_state += coeff * tmp_state

    return new_state


def _op_list(commutator):
    """assume right nested"""

    commutator = tuple(commutator)

    if len(commutator) == 0:
        return {}

    if len(commutator) == 1:
        return {commutator: 1}

    if len(commutator) == 2:
        return {
            (commutator[0], commutator[1]): 1,
            (commutator[1], commutator[0]): -1,
        }

    head, *tail = commutator

    ops1 = defaultdict(int, {(head,) + ops: coeff for ops, coeff in _op_list(tail).items()})
    ops2 = defaultdict(int, {ops + (head,): -coeff for ops, coeff in _op_list(tail).items()})

    for op, coeff in ops2.items():
        ops1[op] += coeff

    return ops1

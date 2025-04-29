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

from collections.abc import Hashable
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

    if product_formula.recursive:
        fragments = {
            pf: effective_hamiltonian(pf, fragments, order) for pf in product_formula.terms
        }

    bch = product_formula.bch_approx(order)
    eff = _AdditiveIdentity()

    for commutator_order in bch:
        for commutator, coeff in commutator_order.items():
            eff += coeff * nested_commutator(
                [frag_coeff * fragments[label] for label, frag_coeff in commutator]
            )

    return product_formula.exponent * eff


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

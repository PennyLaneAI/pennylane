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

from typing import List, Sequence

from pennylane.labs.trotter_error import AbstractState, Fragment
from pennylane.labs.trotter_error.abstract import nested_commutator


class _AdditiveIdentity:
    """Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other


def trotter_error(fragments: Sequence[Fragment], delta: float) -> Fragment:
    r"""Compute the second-order Trotter error operator.

    For a Hamiltonian :math:`H` expressed as a sum of
    fragments :math:`\sum_{m=1}^L H_m`, the second order Trotter formula is given by

    .. math:: e^{iH\Delta t} \approx \prod_{m=1}^L e^{iH_m\Delta t / 2} \prod_{m=L}^1 e^{iH_m \Delta t / 2} = e^{i \tilde{H} \Delta t},

    where :math:`\tilde{H} = H + \hat{\epsilon}`. The leading term of the error operator :math:`\hat{\epsilon}` is given by

    .. math:: \hat{\epsilon} = \frac{- \Delta t^2}{24} \sum_{i=1}^{L-1} \sum_{j = i + 1}^L \left[ H_i + 2 \sum_{k = j + 1}^L H_k, \left[ H_i, H_j \right] \right].

    Args:
        fragments (Sequence[Fragments]): the set of :class:`~.pennylane.labs.trotter_error.Fragment`
            objects to compute Trotter error from
        delta (float): time step for the trotter formula.

    Returns:
        Fragment: the Trotter error operator

    **Example**

    >>> import numpy as np
    >>> from pennylane.labs.trotter_error.fragments import vibrational_fragments
    >>> from pennylane.labs.trotter_error.product_formulas import trotter_error
    >>> n_modes = 4
    >>> r_state = np.random.RandomState(42)
    >>> freqs = r_state.random(4)
    >>> taylor_coeffs = [
    >>>     np.array(0),
    >>>     r_state.random(size=(n_modes, )),
    >>>     r_state.random(size=(n_modes, n_modes)),
    >>>     r_state.random(size=(n_modes, n_modes, n_modes))
    >>> ]
    >>> frags = vibrational_fragments(n_modes, freqs, taylor_coeffs)
    >>> delta = 0.001
    >>> type(trotter_error(frags, delta))
    <class 'pennylane.labs.trotter_error.realspace.realspace_operator.RealspaceSum'>

    """

    if len(fragments) == 0:
        return fragments

    eff = _AdditiveIdentity()
    n_frags = len(fragments)
    scalar = -(delta**2) / 24

    for i in range(n_frags - 1):
        for j in range(i + 1, n_frags):
            eff += nested_commutator([fragments[i], fragments[i], fragments[j]])
            for k in range(i + 1, n_frags):
                eff += 2 * nested_commutator([fragments[k], fragments[i], fragments[j]])

    eff = scalar * eff
    return eff


def perturbation_error(
    fragments: Sequence[Fragment], states: Sequence[AbstractState], delta: float = 1
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

    >>> from pennylane.labs.trotter_error import HOState, vibrational_fragments, perturbation_error
    >>> import numpy as np
    >>> n_modes = 2
    >>> r_state = np.random.RandomState(42)
    >>> freqs = r_state.random(n_modes)
    >>> taylor_coeffs = [
    >>>     np.array(0),
    >>>     r_state.random(size=(n_modes, )),
    >>>     r_state.random(size=(n_modes, n_modes)),
    >>>     r_state.random(size=(n_modes, n_modes, n_modes))
    >>> ]
    >>> frags = vibrational_fragments(n_modes, freqs, taylor_coeffs)
    >>> gridpoints = 5
    >>> state1 = HOState(n_modes, gridpoints, {(0, 0): 1})
    >>> state2 = HOState(n_modes, gridpoints, {(1, 1): 1})
    >>> perturbation_error(frags, [state1, state2])
    [(-0.9189251160920879+0j), (-4.797716682426851+0j)]
    """

    error = trotter_error(fragments, delta)

    return [error.expectation(state, state) for state in states]

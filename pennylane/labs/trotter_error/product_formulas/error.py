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

from pennylane.labs.trotter_error import AbstractState, Fragment, nested_commutator


class _AdditiveIdentity:
    """Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other


def trotter_error(fragments: Sequence[Fragment], delta: float) -> Fragment:
    """Return the second order trotter error.

    Args:
        fragments (Sequence[Fragments]): the set of fragments to compute Trotter error from
        delta (float): the time step parameter

    Returns:
        Fragment: a representation of the Trotter error operator

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

    eff *= scalar
    return eff


def perturbation_error(
    fragments: Sequence[Fragment], states: Sequence[AbstractState], delta: float = 1
) -> List[float]:
    """Return the perturbation theory error obtained from second order Trotter.

    Args:
        fragments (Sequence[Fragments]): the set of fragments to compute Trotter error from
        states: (Sequence[AbstractState]): the states to compute expectation values from
        delta (float): the time step parameter

    Returns:
        List[float]: the list of expectation values computed from the Trotter error operator and the input states
    """

    error = trotter_error(fragments, delta)

    return [error.expectation(state, state) for state in states]

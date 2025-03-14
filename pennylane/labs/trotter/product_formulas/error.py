"""Functions for retreiving effective error from fragments"""

from typing import List, Sequence

import numpy as np

from pennylane.labs.trotter import Fragment, State, nested_commutator


class AdditiveIdentity:
    """Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other


def trotter_error(fragments: Sequence[Fragment], delta: float) -> Fragment:
    """Return the second order trotter error"""
    eff = AdditiveIdentity()
    n_frags = len(fragments)
    scalar = -(delta**2) / 24

    for i in range(n_frags - 1):
        for j in range(i + 1, n_frags):
            eff += nested_commutator([fragments[i], fragments[i], fragments[j]])
            for k in range(i + 1, n_frags):
                eff += 2 * nested_commutator([fragments[k], fragments[i], fragments[j]])

    eff *= scalar
    return eff


def pt_error(fragments: List[Fragment], states: List[State], delta: float = 1) -> np.ndarray:
    """Return the perturbation theory error"""

    error = trotter_error(fragments, delta)

    n_states = len(states)
    expectations = np.zeros(shape=(n_states, n_states))

    for i, left in enumerate(states):
        for j, right in enumerate(states):
            expectations[i, j] = error.expectation(left, right)

    return expectations

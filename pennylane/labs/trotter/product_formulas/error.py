"""Functions for retreiving effective error from fragments"""

from typing import List, Sequence

import numpy as np

from pennylane.labs.trotter import AbstractState, Fragment, nested_commutator


class _AdditiveIdentity:
    """Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other


def trotter_error(fragments: Sequence[Fragment], delta: float) -> Fragment:
    """Return the second order trotter error"""
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
    fragments: List[Fragment], states: List[AbstractState], delta: float = 1
) -> List[float]:
    """Return the perturbation theory error"""

    error = trotter_error(fragments, delta)
    expectations = []

    for state in states:
        expectations.append(error.expectation(state, state))

    return expectations

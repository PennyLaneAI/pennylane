from typing import Sequence

from pennylane.labs.pf import Fragment, nested_commutator

fourth_order_coeffs_2_frags = {}
fourth_order_coeffs_3_frags = {}


class AdditiveIdentity:
    """Only used to initialize accumulators for summing Fragments"""

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other


def trotter(fragments: Sequence[Fragment], delta: float, order: int = 2):
    """Compute effective Hamiltonian from Trotter"""

    if order == 2:
        return _second_order(fragments, delta)

    if order == 4:
        return _fourth_order(fragments, delta)

    raise ValueError("Only second and fourth order Trotter are supported.")


def _second_order(fragments: Sequence[Fragment], delta: float) -> Fragment:
    eff = AdditiveIdentity()
    n_frags = len(fragments)
    scalar = -(delta**2) / 24

    for i in range(n_frags - 1):
        for j in (i + 1, n_frags):
            eff += nested_commutator([fragments[i], fragments[i], fragments[j]])
            for k in range(i + 1, n_frags):
                eff += 2 * nested_commutator([fragments[k], fragments[i], fragments[j]])

    eff *= scalar
    return eff


def _fourth_order(fragments: Sequence[Fragment], delta: float) -> Fragment:
    pass


# def epsilon(self, delta: float) -> VibronicMatrix:
#    # pylint: disable=arguments-out-of-order
#    """Compute the error matrix"""
#    scalar = -(delta**2) / 24
#    epsilon = VibronicMatrix(self.states, self.modes, sparse=self.sparse)
#
#    next_pow_2 = 2 ** (self.states - 1).bit_length()
#
#    for i in range(next_pow_2):
#        for j in range(i + 1, next_pow_2 + 1):
#            epsilon += self._commute_fragments(i, i, j)
#            for k in range(i + 1, next_pow_2 + 1):
#                epsilon += 2 * self._commute_fragments(k, i, j)
#
#    epsilon *= scalar
#    return epsilon

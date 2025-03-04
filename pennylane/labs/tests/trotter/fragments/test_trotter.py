import pytest

from pennylane.labs.trotter import nested_commutator
from pennylane.labs.trotter.fragments import vibronic_fragments
from pennylane.labs.trotter.product_formulas import trotter
from pennylane.labs.trotter.utils import coeffs
from pennylane.labs.trotter.realspace import VibronicMatrix

@pytest.mark.parametrize("modes", range(5))
def test_epsilon(modes):
    """Test that epsilon is correct for 2 states"""
    states = 2
    delta = 0.72
    scalar = -(delta**2) / 24
    fragments = vibronic_fragments(states, modes, *coeffs(states, modes, order=2))

    terms = [
        nested_commutator([fragments[0], fragments[0], fragments[1]]),
        2 * nested_commutator([fragments[1], fragments[0], fragments[1]]),
        2 * nested_commutator([fragments[2], fragments[0], fragments[1]]),
        nested_commutator([fragments[0], fragments[0], fragments[2]]),
        2 * nested_commutator([fragments[1], fragments[0], fragments[2]]),
        2 * nested_commutator([fragments[2], fragments[0], fragments[2]]),
        nested_commutator([fragments[1], fragments[1], fragments[2]]),
        2 * nested_commutator([fragments[2], fragments[1], fragments[2]])
    ]

    #terms = [
    #    vham._commute_fragments(0, 0, 1),
    #    2 * vham._commute_fragments(1, 0, 1),
    #    2 * vham._commute_fragments(2, 0, 1),
    #    vham._commute_fragments(0, 0, 2),
    #    2 * vham._commute_fragments(1, 0, 2),
    #    2 * vham._commute_fragments(2, 0, 2),
    #    vham._commute_fragments(1, 1, 2),
    #    2 * vham._commute_fragments(2, 1, 2),
    #]

    actual = trotter(fragments, delta, order=2)
    expected = scalar * sum(terms, VibronicMatrix(states, modes))

    assert actual == expected

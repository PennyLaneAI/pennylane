import pytest

@pytest.mark.parametrize("modes", range(5))
def test_epsilon(modes):
    """Test that epsilon is correct for 2 states"""
    states = 2
    delta = 0.72
    scalar = -(delta**2) / 24
    vham = VibronicHamiltonian(states, modes, *coeffs(states, modes, order=2))

    terms = [
        vham._commute_fragments(0, 0, 1),
        2 * vham._commute_fragments(1, 0, 1),
        2 * vham._commute_fragments(2, 0, 1),
        vham._commute_fragments(0, 0, 2),
        2 * vham._commute_fragments(1, 0, 2),
        2 * vham._commute_fragments(2, 0, 2),
        vham._commute_fragments(1, 1, 2),
        2 * vham._commute_fragments(2, 1, 2),
    ]

    actual = vham.epsilon(delta)
    expected = scalar * sum(terms, VibronicMatrix(states, modes))

    assert actual == expected

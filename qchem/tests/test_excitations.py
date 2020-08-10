import os

import pytest

from pennylane import qchem


@pytest.mark.parametrize(
    ("electrons", "orbitals", "delta_sz", "n_singles", "n_doubles", "singles_exp", "doubles_exp",),
    [
        (1, 5, 0, 2, 0, [[0, 2], [0, 4]], []),
        (1, 5, 1, 0, 0, [], []),
        (1, 5, -1, 2, 0, [[0, 1], [0, 3]], []),
        (2, 5, 0, 3, 2, [[0, 2], [0, 4], [1, 3]], [[0, 1, 2, 3], [0, 1, 3, 4]]),
        (2, 5, 1, 2, 1, [[1, 2], [1, 4]], [[0, 1, 2, 4]]),
        (2, 5, -1, 1, 0, [[0, 3]], []),
        (2, 5, 2, 0, 0, [], []),
        (3, 6, 1, 1, 0, [[1, 4]], []),
        (
            3,
            6,
            -1,
            4,
            4,
            [[0, 3], [0, 5], [2, 3], [2, 5]],
            [[0, 1, 3, 5], [0, 2, 3, 4], [0, 2, 4, 5], [1, 2, 3, 5]],
        ),
        (3, 6, -2, 0, 1, [], [[0, 2, 3, 5]]),
        (3, 4, 0, 1, 0, [[1, 3]], []),
        (3, 4, 1, 0, 0, [], []),
        (3, 4, -1, 2, 0, [[0, 3], [2, 3]], []),
        (3, 4, 2, 0, 0, [], []),
    ],
)
def test_excitations(electrons, orbitals, delta_sz, n_singles, n_doubles, singles_exp, doubles_exp):
    r"""Test the correctness of the generated configurations"""

    singles, doubles = qchem.excitations(electrons, orbitals, delta_sz)

    assert len(singles) == len(singles_exp)
    assert len(doubles) == len(doubles_exp)
    assert singles == singles_exp
    assert doubles == doubles_exp


@pytest.mark.parametrize(
    ("electrons", "orbitals", "delta_sz", "message_match"),
    [
        (0, 4, 0, "number of active electrons has to be greater than 0"),
        (3, 2, 0, "has to be greater than the number of active electrons"),
        (2, 4, 3, "Expected values for 'delta_sz'"),
        (2, 4, 1.5, "Expected values for 'delta_sz'"),
    ],
)
def test_inconsistent_excitations(electrons, orbitals, delta_sz, message_match):
    r"""Test that an error is raised if a set of inconsistent arguments is input"""

    with pytest.raises(ValueError, match=message_match):
        qchem.excitations(electrons, orbitals, delta_sz)

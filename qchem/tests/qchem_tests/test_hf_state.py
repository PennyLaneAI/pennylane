import numpy as np
import pytest

from pennylane import qchem


@pytest.mark.parametrize(
    ("electrons", "orbitals", "exp_state"),
    [
        (2, 5, np.array([1, 1, 0, 0, 0])),
        (1, 5, np.array([1, 0, 0, 0, 0])),
        (5, 5, np.array([1, 1, 1, 1, 1])),
    ],
)
def test_hf_state(electrons, orbitals, exp_state):
    r"""Test the correctness of the generated occupation-number vector"""

    res_state = qchem.hf_state(electrons, orbitals)

    assert len(res_state) == len(exp_state)
    assert np.allclose(res_state, exp_state)


@pytest.mark.parametrize(
    ("electrons", "orbitals", "msg_match"),
    [
        (0, 5, "number of active electrons has to be larger than zero"),
        (-1, 5, "number of active electrons has to be larger than zero"),
        (6, 5, "number of active orbitals cannot be smaller than the number of active"),
    ],
)
def test_inconsistent_input(electrons, orbitals, msg_match):
    r"""Test that an error is raised if a set of inconsistent arguments is input"""

    with pytest.raises(ValueError, match=msg_match):
        qchem.hf_state(electrons, orbitals)

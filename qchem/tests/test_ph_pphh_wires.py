import os

import pytest

from pennylane import qchem

@pytest.mark.parametrize(
    ("ph_confs", "pphh_confs", "wires", "ph_expected", "pphh_expected"),
    [
        ([[0, 2]], [], None, [[0, 1, 2]], []),
        ([], [[0, 1, 2, 3]], None, [], [[[0, 1], [2, 3]]]),
        ([[0, 1]], [[0, 1, 2, 4]], None, [[0, 1]], [[[0, 1], [2, 3, 4]]]),
        ([[0, 1], [2, 4]], [[0, 1, 2, 3], [0, 2, 4, 6]], None,
         [[0, 1], [2, 3, 4]], [[[0, 1], [2, 3]], [[0, 1, 2], [4, 5, 6]]]),
        ([[0, 1], [2, 4]], [[0, 1, 2, 3], [0, 2, 4, 6]], ['a0', 'b1', 'c2', 'd3', 'e4', 'f5','g6'],
         [['a0', 'b1'], ['c2', 'd3', 'e4']], [[['a0', 'b1'], ['c2', 'd3']],
          [['a0', 'b1', 'c2'], ['e4', 'f5', 'g6']]]),
    ]
)
def test_mapping_from_ph_to_wires(
    ph_confs, pphh_confs, wires, ph_expected, pphh_expected
):

    r"""Test the correctness of the mapping between indices of the particle-hole
    configurations and the list of wires to be passed to the quantum circuit"""

    ph_res, pphh_res = qchem.ph_pphh_wires(ph_confs, pphh_confs, wires=wires)

    assert len(ph_res) == len(ph_expected)
    assert len(pphh_res) == len(pphh_expected)
    assert ph_res == ph_expected
    assert pphh_res == pphh_expected


@pytest.mark.parametrize(
    ("ph_confs", "pphh_confs", "wires", "message_match"),
    [
        ([], [], None, "'ph_confs' and 'pphh_confs' lists can not be both empty"),
        ([[0, 2, 3]], [], None, "expected entries of 'ph_confs' to be of shape"),
        ([[0, 2], [3]], [], None, "expected entries of 'ph_confs' to be of shape"),
        ([], [[0, 1, 2, 3], [1, 3]], None, "expected entries of 'pphh_confs' to be of shape"),
        ([], [[0, 1, 2, 3], [1, 3, 4, 5, 6]], None, 
            "expected entries of 'pphh_confs' to be of shape"),
        ([[0,2]], [[0, 1, 2, 3], [0, 2, 4, 6]], ['a0', 'b1', 'c2', 'd3', 'e4', 'f5'], 
            "Expected number of wires is"),
    ]
)
def test_ph_pphh_exceptions(ph_confs, pphh_confs, wires, message_match):

    r"""Test that the function 'ph_pphh_wires()' throws an exception if ``ph_confs``,
    ``pphh_confs`` or ``wires`` parameter has illegal shapes or size"""

    with pytest.raises(ValueError, match=message_match):
        qchem.ph_pphh_wires(ph_confs, pphh_confs, wires=wires)

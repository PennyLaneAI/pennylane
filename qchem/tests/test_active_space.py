import os

import pytest

from pennylane import qchem

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")


@pytest.mark.parametrize(
    ("mol_name", "n_act_electrons", "n_act_orbitals", "docc_indices_ref", "act_indices_ref"),
    [
        ("lih", None, None, [], list(range(6))),
        ("lih", 4, None, [], list(range(6))),
        ("lih", 2, None, [0], list(range(1, 6))),
        ("lih", None, 4, [], list(range(4))),
        ("lih", 2, 3, [0], list(range(1, 4))),
        ("lih_anion", 3, 4, [0], list(range(1, 5))),
        ("lih_anion", 1, 4, [0, 1], list(range(2, 6))),
    ],
)
def test_active_spaces(
    mol_name, n_act_electrons, n_act_orbitals, docc_indices_ref, act_indices_ref
):

    r"""Test the correctness of the generated active spaces"""

    docc_indices, active_indices = qchem.active_space(
        mol_name, ref_dir, n_act_electrons, n_act_orbitals
    )

    assert docc_indices == docc_indices_ref
    assert active_indices == act_indices_ref


@pytest.mark.parametrize(
    ("mol_name", "n_act_electrons", "n_act_orbitals", "message_match"),
    [
        ("lih", 6, 5, "greater than the total number of electrons"),
        ("lih", 1, 5, "should be even"),
        ("lih", -1, 5, "has to be greater than 0."),
        ("lih", 2, 6, "can not be greater than the number " "of molecular orbitals"),
        ("lih", 2, 1, "there are no virtual orbitals"),
        ("lih_anion", 2, 5, "should be odd"),
        ("lih_anion", 3, -2, "has to be greater than 0."),
        ("lih_anion", 3, 6, "can not be greater than the number " "of molecular orbitals"),
        ("lih_anion", 3, 2, "there are no virtual orbitals"),
        ("lih_anion_2", 1, 2, "greater than or equal to"),
    ],
)
def test_inconsistent_active_spaces(mol_name, n_act_electrons, n_act_orbitals, message_match):

    r"""Test that an error is raised if an inconsistent active space is generated"""

    with pytest.raises(ValueError, match=message_match):
        qchem.active_space(mol_name, ref_dir, n_act_electrons, n_act_orbitals)

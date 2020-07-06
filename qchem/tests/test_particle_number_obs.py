import os

import numpy as np
import pytest

from pennylane import qchem

from openfermion.ops._qubit_operator import QubitOperator
from openfermion.hamiltonians import MolecularData

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")

terms_h20_jw_full = {
    (): (7 + 0j),
    ((0, "Z"),): (-0.5 + 0j),
    ((1, "Z"),): (-0.5 + 0j),
    ((2, "Z"),): (-0.5 + 0j),
    ((3, "Z"),): (-0.5 + 0j),
    ((4, "Z"),): (-0.5 + 0j),
    ((5, "Z"),): (-0.5 + 0j),
    ((6, "Z"),): (-0.5 + 0j),
    ((7, "Z"),): (-0.5 + 0j),
    ((8, "Z"),): (-0.5 + 0j),
    ((9, "Z"),): (-0.5 + 0j),
    ((10, "Z"),): (-0.5 + 0j),
    ((11, "Z"),): (-0.5 + 0j),
    ((12, "Z"),): (-0.5 + 0j),
    ((13, "Z"),): (-0.5 + 0j),
}

terms_h20_jw_23 = {
    (): (11 + 0j),
    ((0, "Z"),): (-0.5 + 0j),
    ((1, "Z"),): (-0.5 + 0j),
    ((2, "Z"),): (-0.5 + 0j),
    ((3, "Z"),): (-0.5 + 0j),
    ((4, "Z"),): (-0.5 + 0j),
    ((5, "Z"),): (-0.5 + 0j),
}

terms_h20_bk_44 = {
    (): (10 + 0j),
    ((0, "Z"),): (-0.5 + 0j),
    ((0, "Z"), (1, "Z")): (-0.5 + 0j),
    ((2, "Z"),): (-0.5 + 0j),
    ((1, "Z"), (2, "Z"), (3, "Z")): (-0.5 + 0j),
    ((4, "Z"),): (-0.5 + 0j),
    ((4, "Z"), (5, "Z")): (-0.5 + 0j),
    ((6, "Z"),): (-0.5 + 0j),
    ((3, "Z"), (5, "Z"), (6, "Z"), (7, "Z")): (-0.5 + 0j),
}

terms_lih_anion_bk = {
    (): (7 + 0j),
    ((0, "Z"),): (-0.5 + 0j),
    ((0, "Z"), (1, "Z")): (-0.5 + 0j),
    ((2, "Z"),): (-0.5 + 0j),
    ((1, "Z"), (2, "Z"), (3, "Z")): (-0.5 + 0j),
    ((4, "Z"),): (-0.5 + 0j),
    ((4, "Z"), (5, "Z")): (-0.5 + 0j),
    ((6, "Z"),): (-0.5 + 0j),
    ((3, "Z"), (5, "Z"), (6, "Z"), (7, "Z")): (-0.5 + 0j),
    ((8, "Z"),): (-0.5 + 0j),
    ((8, "Z"), (9, "Z")): (-0.5 + 0j),
}


@pytest.mark.parametrize(
    ("mol_name", "n_act_elect", "n_act_orb", "mapping", "terms_exp"),
    [
        ("h2o_psi4", None, None, "JORDAN_wigner", terms_h20_jw_full),
        ("h2o_psi4", 2, 3, "JORDAN_wigner", terms_h20_jw_23),
        ("h2o_psi4", 4, 4, "bravyi_KITAEV", terms_h20_bk_44),
        ("lih_anion", 3, None, "bravyi_KITAEV", terms_lih_anion_bk),
    ],
)
def test_particle_number_observable(
    mol_name, n_act_elect, n_act_orb, mapping, terms_exp, monkeypatch
):
    r"""Tests the correctness of the particle number observable :math:`\hat{N}` generated
    by the ``'particle_number'`` function.

    The parametrized inputs are `.terms` attribute of the particle number `QubitOperator`.
    The equality checking is implemented in the `qchem` module itself as it could be
    something useful to the users as well.
    """

    mol_data = MolecularData(filename=os.path.join(ref_dir.strip(), mol_name.strip()))

    docc, act = qchem.active_space(
        mol_name, ref_dir, n_active_electrons=n_act_elect, n_active_orbitals=n_act_orb,
    )

    pn_obs = qchem.particle_number(mol_data, docc_orb=docc, act_orb=act, mapping=mapping)

    particle_number_qubit_op = QubitOperator()
    monkeypatch.setattr(particle_number_qubit_op, "terms", terms_exp)

    assert qchem._qubit_operators_equivalent(particle_number_qubit_op, pn_obs)


@pytest.mark.parametrize(
    ("docc_orb", "act_orb", "msg_match"),
    [(2, [1, 3], "'docc_orb' must be a list"), ([0, 1], 4, "'act_orb' must be a list"),],
)
def test_exceptions_particle_number(docc_orb, act_orb, msg_match):
    r"""Tests that the 'particle_number' function throws an exception if 'docc_orb' or 
    'act_orb' are not lists."""

    mol_name = "h2o_psi4"
    mapping = "jordan_wigner"
    mol_data = mol_data = MolecularData(filename=os.path.join(ref_dir.strip(), mol_name.strip()))
    with pytest.raises(ValueError, match=msg_match):
        qchem.particle_number(mol_data, docc_orb=docc_orb, act_orb=act_orb, mapping=mapping)

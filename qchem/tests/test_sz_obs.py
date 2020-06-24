import os

import numpy as np
import pytest

from pennylane import qchem

from openfermion.ops._qubit_operator import QubitOperator

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")

me = np.array([[0.0, 0.0, 0.5], [1.0, 1.0, -0.5], [2.0, 2.0, 0.5], [3.0, 3.0, -0.5]])

me_lih = np.array(
    [
        [0.0, 0.0, 0.5],
        [1.0, 1.0, -0.5],
        [2.0, 2.0, 0.5],
        [3.0, 3.0, -0.5],
        [4.0, 4.0, 0.5],
        [5.0, 5.0, -0.5],
    ]
)


@pytest.mark.parametrize(
    ("mol_name", "n_act_elect", "n_act_orb", "sz_me_exp"),
    [
        ("h2_pyscf", None, None, me),
        ("h2_pyscf", 2, None, me),
        ("lih", 2, 2, me),
        ("lih", None, 3, me_lih),
    ],
)
def test_get_spinZ_matrix_elements(mol_name, n_act_elect, n_act_orb, sz_me_exp, tol):
    r"""Test that the table of matrix elements
    :math:`\langle \alpha \vert \hat{s}_z \vert \beta \rangle` are computed correctly
    for different choices of the active space."""

    sz_me_res = qchem.get_spinZ_matrix_elements(
        mol_name, ref_dir, n_active_electrons=n_act_elect, n_active_orbitals=n_act_orb
    )

    assert np.allclose(sz_me_res, sz_me_exp, **tol)


terms_lih_jw = {
    ((0, "Z"),): (-0.25 + 0j),
    ((1, "Z"),): (0.25 + 0j),
    ((2, "Z"),): (-0.25 + 0j),
    ((3, "Z"),): (0.25 + 0j),
}

terms_lih_anion_bk = {
    ((0, "Z"),): (-0.25 + 0j),
    ((0, "Z"), (1, "Z")): (0.25 + 0j),
    ((2, "Z"),): (-0.25 + 0j),
    ((1, "Z"), (2, "Z"), (3, "Z")): (0.25 + 0j),
    ((4, "Z"),): (-0.25 + 0j),
    ((4, "Z"), (5, "Z")): (0.25 + 0j),
}


@pytest.mark.parametrize(
    ("mol_name", "n_act_elect", "n_act_orb", "mapping", "terms_exp"),
    [
        ("lih", 2, 2, "JORDAN_wigner", terms_lih_jw),
        ("lih_anion", 3, 3, "bravyi_KITAEV", terms_lih_anion_bk),
    ],
)
def test_build_sz_observable(mol_name, n_act_elect, n_act_orb, mapping, terms_exp, monkeypatch):
    r"""Tests the correctness of the built total-spin projection observable :math:`\hat{S}_z`.

    The parametrized inputs are `.terms` attribute of the total spin `QubitOperator.
    The equality checking is implemented in the `qchem` module itself as it could be
    something useful to the users as well.
    """

    sz_me_table = qchem.get_spinZ_matrix_elements(
        mol_name, ref_dir, n_active_electrons=n_act_elect, n_active_orbitals=n_act_orb
    )

    sz_obs = qchem.observable(sz_me_table, mapping=mapping)

    sz_qubit_op = QubitOperator()
    monkeypatch.setattr(sz_qubit_op, "terms", terms_exp)

    assert qchem._qubit_operators_equivalent(sz_qubit_op, sz_obs)

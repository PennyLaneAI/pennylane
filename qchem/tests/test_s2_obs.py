import os

import numpy as np
import pytest

from pennylane import qchem

from openfermion.ops._qubit_operator import QubitOperator

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")


@pytest.mark.parametrize(
    ("n_spin_orbs", "s2_me_expected"),
    [
        (1, np.array([[0.0, 0.0, 0.0, 0.0, 0.25]])),
        (
            3,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.25],
                    [0.0, 1.0, 1.0, 0.0, -0.25],
                    [0.0, 2.0, 2.0, 0.0, 0.25],
                    [1.0, 0.0, 0.0, 1.0, -0.25],
                    [1.0, 1.0, 1.0, 1.0, 0.25],
                    [1.0, 2.0, 2.0, 1.0, -0.25],
                    [2.0, 0.0, 0.0, 2.0, 0.25],
                    [2.0, 1.0, 1.0, 2.0, -0.25],
                    [2.0, 2.0, 2.0, 2.0, 0.25],
                    [0.0, 1.0, 0.0, 1.0, 0.5],
                    [1.0, 0.0, 1.0, 0.0, 0.5],
                ]
            ),
        ),
    ],
)
def test_spin2_matrix_elements(n_spin_orbs, s2_me_expected, tol):
    r"""Test the calculation of the matrix elements of the two-particle spin operator
    :math:`\hat{s}_1 \cdot \hat{s}_2`"""

    sz = np.where(np.arange(n_spin_orbs) % 2 == 0, 0.5, -0.5)

    s2_me_result = qchem.spin2_matrix_elements(sz, n_spin_orbs)

    assert np.allclose(s2_me_result, s2_me_expected, **tol)


def test_exception_spin2_me(message_match="Size of 'sz' must be equal to 'n_spin_orbs'"):
    """Test that the 'spin2_matrix_elements' function throws an exception if the
    size of 'sz' is not equal to 'n_spin_orbs'."""

    n_spin_orbs = 4
    sz = np.where(np.arange(n_spin_orbs + 1) % 2 == 0, 0.5, -0.5)

    with pytest.raises(ValueError, match=message_match):
        qchem.spin2_matrix_elements(sz, n_spin_orbs)


me = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.25],
        [0.0, 1.0, 1.0, 0.0, -0.25],
        [0.0, 2.0, 2.0, 0.0, 0.25],
        [0.0, 3.0, 3.0, 0.0, -0.25],
        [1.0, 0.0, 0.0, 1.0, -0.25],
        [1.0, 1.0, 1.0, 1.0, 0.25],
        [1.0, 2.0, 2.0, 1.0, -0.25],
        [1.0, 3.0, 3.0, 1.0, 0.25],
        [2.0, 0.0, 0.0, 2.0, 0.25],
        [2.0, 1.0, 1.0, 2.0, -0.25],
        [2.0, 2.0, 2.0, 2.0, 0.25],
        [2.0, 3.0, 3.0, 2.0, -0.25],
        [3.0, 0.0, 0.0, 3.0, -0.25],
        [3.0, 1.0, 1.0, 3.0, 0.25],
        [3.0, 2.0, 2.0, 3.0, -0.25],
        [3.0, 3.0, 3.0, 3.0, 0.25],
        [0.0, 1.0, 0.0, 1.0, 0.5],
        [0.0, 3.0, 2.0, 1.0, 0.5],
        [1.0, 0.0, 1.0, 0.0, 0.5],
        [1.0, 2.0, 3.0, 0.0, 0.5],
        [2.0, 1.0, 0.0, 3.0, 0.5],
        [2.0, 3.0, 2.0, 3.0, 0.5],
        [3.0, 0.0, 1.0, 2.0, 0.5],
        [3.0, 2.0, 3.0, 2.0, 0.5],
    ]
)

me_lih = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.25],
        [0.0, 1.0, 1.0, 0.0, -0.25],
        [0.0, 2.0, 2.0, 0.0, 0.25],
        [0.0, 3.0, 3.0, 0.0, -0.25],
        [0.0, 4.0, 4.0, 0.0, 0.25],
        [0.0, 5.0, 5.0, 0.0, -0.25],
        [1.0, 0.0, 0.0, 1.0, -0.25],
        [1.0, 1.0, 1.0, 1.0, 0.25],
        [1.0, 2.0, 2.0, 1.0, -0.25],
        [1.0, 3.0, 3.0, 1.0, 0.25],
        [1.0, 4.0, 4.0, 1.0, -0.25],
        [1.0, 5.0, 5.0, 1.0, 0.25],
        [2.0, 0.0, 0.0, 2.0, 0.25],
        [2.0, 1.0, 1.0, 2.0, -0.25],
        [2.0, 2.0, 2.0, 2.0, 0.25],
        [2.0, 3.0, 3.0, 2.0, -0.25],
        [2.0, 4.0, 4.0, 2.0, 0.25],
        [2.0, 5.0, 5.0, 2.0, -0.25],
        [3.0, 0.0, 0.0, 3.0, -0.25],
        [3.0, 1.0, 1.0, 3.0, 0.25],
        [3.0, 2.0, 2.0, 3.0, -0.25],
        [3.0, 3.0, 3.0, 3.0, 0.25],
        [3.0, 4.0, 4.0, 3.0, -0.25],
        [3.0, 5.0, 5.0, 3.0, 0.25],
        [4.0, 0.0, 0.0, 4.0, 0.25],
        [4.0, 1.0, 1.0, 4.0, -0.25],
        [4.0, 2.0, 2.0, 4.0, 0.25],
        [4.0, 3.0, 3.0, 4.0, -0.25],
        [4.0, 4.0, 4.0, 4.0, 0.25],
        [4.0, 5.0, 5.0, 4.0, -0.25],
        [5.0, 0.0, 0.0, 5.0, -0.25],
        [5.0, 1.0, 1.0, 5.0, 0.25],
        [5.0, 2.0, 2.0, 5.0, -0.25],
        [5.0, 3.0, 3.0, 5.0, 0.25],
        [5.0, 4.0, 4.0, 5.0, -0.25],
        [5.0, 5.0, 5.0, 5.0, 0.25],
        [0.0, 1.0, 0.0, 1.0, 0.5],
        [0.0, 3.0, 2.0, 1.0, 0.5],
        [0.0, 5.0, 4.0, 1.0, 0.5],
        [1.0, 0.0, 1.0, 0.0, 0.5],
        [1.0, 2.0, 3.0, 0.0, 0.5],
        [1.0, 4.0, 5.0, 0.0, 0.5],
        [2.0, 1.0, 0.0, 3.0, 0.5],
        [2.0, 3.0, 2.0, 3.0, 0.5],
        [2.0, 5.0, 4.0, 3.0, 0.5],
        [3.0, 0.0, 1.0, 2.0, 0.5],
        [3.0, 2.0, 3.0, 2.0, 0.5],
        [3.0, 4.0, 5.0, 2.0, 0.5],
        [4.0, 1.0, 0.0, 5.0, 0.5],
        [4.0, 3.0, 2.0, 5.0, 0.5],
        [4.0, 5.0, 4.0, 5.0, 0.5],
        [5.0, 0.0, 1.0, 4.0, 0.5],
        [5.0, 2.0, 3.0, 4.0, 0.5],
        [5.0, 4.0, 5.0, 4.0, 0.5],
    ]
)


@pytest.mark.parametrize(
    ("mol_name", "n_act_elect", "n_act_orb", "s2_me_exp", "init_term_exp"),
    [
        ("h2_pyscf", None, None, me, 1.5),
        ("h2_pyscf", 2, None, me, 1.5),
        ("lih", 2, 2, me, 1.5),
        ("lih", None, 3, me_lih, 3.0),
    ],
)
def test_get_spin2_matrix_elements(mol_name, n_act_elect, n_act_orb, s2_me_exp, init_term_exp, tol):
    r"""Test that the table of matrix elements and the term use to initialize the
    FermionOperator are computed correctly for different active spaces."""

    s2_me_res, init_term_res = qchem.get_spin2_matrix_elements(
        mol_name, ref_dir, n_active_electrons=n_act_elect, n_active_orbitals=n_act_orb
    )

    assert np.allclose(s2_me_res, s2_me_exp, **tol)
    assert init_term_res == init_term_exp


terms_lih_jw = {
    (): (0.75 + 0j),
    ((1, "Z"),): (0.375 + 0j),
    ((0, "Z"), (1, "Z")): (-0.375 + 0j),
    ((0, "Z"), (2, "Z")): (0.125 + 0j),
    ((0, "Z"),): (0.375 + 0j),
    ((0, "Z"), (3, "Z")): (-0.125 + 0j),
    ((1, "Z"), (2, "Z")): (-0.125 + 0j),
    ((1, "Z"), (3, "Z")): (0.125 + 0j),
    ((2, "Z"),): (0.375 + 0j),
    ((3, "Z"),): (0.375 + 0j),
    ((2, "Z"), (3, "Z")): (-0.375 + 0j),
    ((0, "Y"), (1, "X"), (2, "Y"), (3, "X")): (0.125 + 0j),
    ((0, "Y"), (1, "Y"), (2, "X"), (3, "X")): (0.125 + 0j),
    ((0, "Y"), (1, "Y"), (2, "Y"), (3, "Y")): (0.125 + 0j),
    ((0, "Y"), (1, "X"), (2, "X"), (3, "Y")): (-0.125 + 0j),
    ((0, "X"), (1, "Y"), (2, "Y"), (3, "X")): (-0.125 + 0j),
    ((0, "X"), (1, "X"), (2, "X"), (3, "X")): (0.125 + 0j),
    ((0, "X"), (1, "X"), (2, "Y"), (3, "Y")): (0.125 + 0j),
    ((0, "X"), (1, "Y"), (2, "X"), (3, "Y")): (0.125 + 0j),
}

terms_lih_anion_bk = {
    (): (1.125 + 0j),
    ((0, "Z"), (1, "Z")): (0.375 + 0j),
    ((1, "Z"),): (-0.375 + 0j),
    ((0, "Z"), (2, "Z")): (0.125 + 0j),
    ((0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")): (-0.125 + 0j),
    ((0, "Z"), (4, "Z")): (0.125 + 0j),
    ((0, "Z"),): (0.375 + 0j),
    ((0, "Z"), (4, "Z"), (5, "Z")): (-0.125 + 0j),
    ((0, "Z"), (1, "Z"), (2, "Z")): (-0.125 + 0j),
    ((0, "Z"), (2, "Z"), (3, "Z")): (0.125 + 0j),
    ((0, "Z"), (1, "Z"), (4, "Z")): (-0.125 + 0j),
    ((0, "Z"), (1, "Z"), (4, "Z"), (5, "Z")): (0.125 + 0j),
    ((1, "Z"), (2, "Z"), (3, "Z")): (0.375 + 0j),
    ((1, "Z"), (3, "Z")): (-0.375 + 0j),
    ((2, "Z"), (4, "Z")): (0.125 + 0j),
    ((2, "Z"),): (0.375 + 0j),
    ((2, "Z"), (4, "Z"), (5, "Z")): (-0.125 + 0j),
    ((1, "Z"), (2, "Z"), (3, "Z"), (4, "Z")): (-0.125 + 0j),
    ((1, "Z"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "Z")): (0.125 + 0j),
    ((4, "Z"),): (0.375 + 0j),
    ((4, "Z"), (5, "Z")): (0.375 + 0j),
    ((5, "Z"),): (-0.375 + 0j),
    ((0, "Y"), (2, "Y")): (0.125 + 0j),
    ((0, "X"), (1, "Z"), (2, "X")): (-0.125 + 0j),
    ((0, "X"), (2, "X"), (3, "Z")): (0.125 + 0j),
    ((0, "Y"), (1, "Z"), (2, "Y"), (3, "Z")): (-0.125 + 0j),
    ((0, "Y"), (1, "Z"), (2, "Y")): (-0.125 + 0j),
    ((0, "X"), (2, "X")): (0.125 + 0j),
    ((0, "X"), (1, "Z"), (2, "X"), (3, "Z")): (-0.125 + 0j),
    ((0, "Y"), (2, "Y"), (3, "Z")): (0.125 + 0j),
    ((0, "Y"), (4, "Y")): (0.125 + 0j),
    ((0, "X"), (1, "Z"), (4, "X")): (-0.125 + 0j),
    ((0, "X"), (1, "Z"), (4, "X"), (5, "Z")): (0.125 + 0j),
    ((0, "Y"), (4, "Y"), (5, "Z")): (-0.125 + 0j),
    ((0, "Y"), (1, "Z"), (4, "Y")): (-0.125 + 0j),
    ((0, "X"), (4, "X")): (0.125 + 0j),
    ((0, "X"), (4, "X"), (5, "Z")): (-0.125 + 0j),
    ((0, "Y"), (1, "Z"), (4, "Y"), (5, "Z")): (0.125 + 0j),
    ((2, "Y"), (4, "Y")): (0.125 + 0j),
    ((1, "Z"), (2, "X"), (3, "Z"), (4, "X")): (-0.125 + 0j),
    ((1, "Z"), (2, "X"), (3, "Z"), (4, "X"), (5, "Z")): (0.125 + 0j),
    ((2, "Y"), (4, "Y"), (5, "Z")): (-0.125 + 0j),
    ((1, "Z"), (2, "Y"), (3, "Z"), (4, "Y")): (-0.125 + 0j),
    ((2, "X"), (4, "X")): (0.125 + 0j),
    ((2, "X"), (4, "X"), (5, "Z")): (-0.125 + 0j),
    ((1, "Z"), (2, "Y"), (3, "Z"), (4, "Y"), (5, "Z")): (0.125 + 0j),
}


@pytest.mark.parametrize(
    ("mol_name", "n_act_elect", "n_act_orb", "mapping", "terms_exp"),
    [
        ("lih", 2, 2, "JORDAN_wigner", terms_lih_jw),
        ("lih_anion", 3, 3, "bravyi_KITAEV", terms_lih_anion_bk),
    ],
)
def test_build_s2_observable(mol_name, n_act_elect, n_act_orb, mapping, terms_exp, monkeypatch):
    r"""Tests the correctness of the built total-spin observable.

    The parametrized inputs are `.terms` attribute of the total spin `QubitOperator.
    The equality checking is implemented in the `qchem` module itself as it could be
    something useful to the users as well.
    """

    s2_me_table, init_term = qchem.get_spin2_matrix_elements(
        mol_name, ref_dir, n_active_electrons=n_act_elect, n_active_orbitals=n_act_orb
    )

    s2_obs = qchem.observable(s2_me_table, init_term=init_term, mapping=mapping)

    s2_qubit_op = QubitOperator()
    monkeypatch.setattr(s2_qubit_op, "terms", terms_exp)

    assert qchem._qubit_operators_equivalent(s2_qubit_op, s2_obs)

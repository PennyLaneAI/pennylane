import os

import numpy as np
import pytest

from pennylane import qchem

from openfermion.ops._qubit_operator import QubitOperator

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")

table_h2o_1 = np.array([[ 0.,  0.,  1.],
                        [ 1.,  1.,  1.],
                        [ 2.,  2.,  1.],
                        [ 3.,  3.,  1.],
                        [ 4.,  4.,  1.],
                        [ 5.,  5.,  1.],
                        [ 6.,  6.,  1.],
                        [ 7.,  7.,  1.],
                        [ 8.,  8.,  1.],
                        [ 9.,  9.,  1.],
                        [10., 10.,  1.],
                        [11., 11.,  1.],
                        [12., 12.,  1.],
                        [13., 13.,  1.]])

table_h2o_2 = np.array([[0., 0., 1.],
                        [1., 1., 1.],
                        [2., 2., 1.],
                        [3., 3., 1.],
                        [4., 4., 1.],
                        [5., 5., 1.],
                        [6., 6., 1.],
                        [7., 7., 1.]])

table_lih_anion = np.array([[0., 0., 1.],
                            [1., 1., 1.],
                            [2., 2., 1.],
                            [3., 3., 1.],
                            [4., 4., 1.],
                            [5., 5., 1.],
                            [6., 6., 1.],
                            [7., 7., 1.],
                            [8., 8., 1.],
                            [9., 9., 1.]])

@pytest.mark.parametrize(
    ("mol_name", "n_act_elect", "n_act_orb", "pn_table_exp", "pn_docc_exp"),
    [
        ("h2o_psi4" , None, None, table_h2o_1    , 0),
        ("h2o_psi4" , 4   , 4   , table_h2o_2    , 6),
        ("lih_anion", 3   , None, table_lih_anion, 2),
    ],
)
def test_get_particle_number_table(
    mol_name,
    n_act_elect,
    n_act_orb,
    pn_table_exp,
    pn_docc_exp,
    tol
):
    r"""Test the correctness of the table required to build the particle number
    operator for different choices of the active space."""

    pn_table_res, pn_docc_res = qchem.get_particle_number_table(
        mol_name, ref_dir, n_active_electrons=n_act_elect, n_active_orbitals=n_act_orb
    )

    assert np.allclose(pn_table_res, pn_table_exp, **tol)
    assert pn_docc_res == pn_docc_exp


terms_h20_jw = {
    (): (7+0j),
    ((0, 'Z'),): (-0.5+0j),
    ((1, 'Z'),): (-0.5+0j),
    ((2, 'Z'),): (-0.5+0j),
    ((3, 'Z'),): (-0.5+0j),
    ((4, 'Z'),): (-0.5+0j),
    ((5, 'Z'),): (-0.5+0j), 
    ((6, 'Z'),): (-0.5+0j),
    ((7, 'Z'),): (-0.5+0j),
    ((8, 'Z'),): (-0.5+0j),
    ((9, 'Z'),): (-0.5+0j),
    ((10, 'Z'),): (-0.5+0j),
    ((11, 'Z'),): (-0.5+0j),
    ((12, 'Z'),): (-0.5+0j),
    ((13, 'Z'),): (-0.5+0j)
}

terms_lih_anion_bk = {
    (): (7+0j),
    ((0, 'Z'),): (-0.5+0j),
    ((0, 'Z'), (1, 'Z')): (-0.5+0j),
    ((2, 'Z'),): (-0.5+0j),
    ((1, 'Z'), (2, 'Z'), (3, 'Z')): (-0.5+0j),
    ((4, 'Z'),): (-0.5+0j),
    ((4, 'Z'), (5, 'Z')): (-0.5+0j),
    ((6, 'Z'),): (-0.5+0j),
    ((3, 'Z'), (5, 'Z'), (6, 'Z'), (7, 'Z')): (-0.5+0j),
    ((8, 'Z'),): (-0.5+0j),
    ((8, 'Z'), (9, 'Z')): (-0.5+0j)
}


@pytest.mark.parametrize(
    ("mol_name", "n_act_elect", "n_act_orb", "mapping", "terms_exp"),
    [
        ("h2o_psi4" , None, None, "JORDAN_wigner", terms_h20_jw),
        ("lih_anion", 3   , None, "bravyi_KITAEV", terms_lih_anion_bk),
    ],
)
def test_build_particle_number_observable(
    mol_name,
    n_act_elect,
    n_act_orb,
    mapping,
    terms_exp,
    monkeypatch
):
    r"""Tests the correctness of the built particle number observable :math:`\hat{N}`.

    The parametrized inputs are `.terms` attribute of the particle number `QubitOperator.
    The equality checking is implemented in the `qchem` module itself as it could be
    something useful to the users as well.
    """

    pn_table, pn_docc = qchem.get_particle_number_table(
        mol_name, ref_dir, n_active_electrons=n_act_elect, n_active_orbitals=n_act_orb
    )

    particle_number_obs = qchem.observable(pn_table, init_term=pn_docc, mapping=mapping)

    particle_number_qubit_op = QubitOperator()
    monkeypatch.setattr(particle_number_qubit_op, "terms", terms_exp)

    assert qchem._qubit_operators_equivalent(particle_number_qubit_op, particle_number_obs)    

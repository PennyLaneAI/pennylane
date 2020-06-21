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

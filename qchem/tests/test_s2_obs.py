import os

import numpy as np
import pytest

#from pennylane import obs
from pennylane_qchem import obs
#from pennylane import obs

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")

@pytest.mark.parametrize(
    ("n_spin_orbs", "s2_me_expected"),
    [
        (1, np.array([[0., 0., 0., 0., 0.25]])),
        (3, np.array([[ 0., 0., 0., 0., 0.25],
                      [ 0., 1., 1., 0., -0.25],
                      [ 0., 2., 2., 0., 0.25],
                      [ 1., 0., 0., 1., -0.25],
                      [ 1., 1., 1., 1., 0.25],
                      [ 1., 2., 2., 1., -0.25],
                      [ 2., 0., 0., 2., 0.25],
                      [ 2., 1., 1., 2., -0.25],
                      [ 2., 2., 2., 2., 0.25],
                      [ 0., 1., 0., 1., 0.5 ],
                      [ 1., 0., 1., 0., 0.5 ]])),
    ],
)
def test_s2_matrix_elements(n_spin_orbs, s2_me_expected, tol):
    r"""Test the calculation of the matrix elements of the two-particle spin operator
    :math:`\hat{s}_1 \cdot \hat{s}_2`"""

    sz = np.where(np.arange(n_spin_orbs) % 2 == 0, 0.5, -0.5)

    s2_me_result = obs.s2_me_table(sz, n_spin_orbs)

    assert np.allclose(s2_me_result, s2_me_expected, **tol)


me = np.array([[ 0.,    0.,    0.,    0.,    0.25], [ 0.,    1.,    1.,    0.,   -0.25],
               [ 0.,    2.,    2.,    0.,    0.25], [ 0.,    3.,    3.,    0.,   -0.25],
               [ 1.,    0.,    0.,    1.,   -0.25], [ 1.,    1.,    1.,    1.,    0.25],
               [ 1.,    2.,    2.,    1.,   -0.25], [ 1.,    3.,    3.,    1.,    0.25],
               [ 2.,    0.,    0.,    2.,    0.25], [ 2.,    1.,    1.,    2.,   -0.25],
               [ 2.,    2.,    2.,    2.,    0.25], [ 2.,    3.,    3.,    2.,   -0.25],
               [ 3.,    0.,    0.,    3.,   -0.25], [ 3.,    1.,    1.,    3.,    0.25],
               [ 3.,    2.,    2.,    3.,   -0.25], [ 3.,    3.,    3.,    3.,    0.25],
               [ 0.,    1.,    0.,    1.,    0.5 ], [ 0.,    3.,    2.,    1.,    0.5 ],
               [ 1.,    0.,    1.,    0.,    0.5 ], [ 1.,    2.,    3.,    0.,    0.5 ],
               [ 2.,    1.,    0.,    3.,    0.5 ], [ 2.,    3.,    2.,    3.,    0.5 ],
               [ 3.,    0.,    1.,    2.,    0.5 ], [ 3.,    2.,    3.,    2.,    0.5 ]])

me_lih = np.array([[ 0.,    0.,    0.,    0.,    0.25], [ 0.,    1.,    1.,    0.,   -0.25],
                   [ 0.,    2.,    2.,    0.,    0.25], [ 0.,    3.,    3.,    0.,   -0.25],
                   [ 0.,    4.,    4.,    0.,    0.25], [ 0.,    5.,    5.,    0.,   -0.25],
                   [ 1.,    0.,    0.,    1.,   -0.25], [ 1.,    1.,    1.,    1.,    0.25],
                   [ 1.,    2.,    2.,    1.,   -0.25], [ 1.,    3.,    3.,    1.,    0.25],
                   [ 1.,    4.,    4.,    1.,   -0.25], [ 1.,    5.,    5.,    1.,    0.25],
                   [ 2.,    0.,    0.,    2.,    0.25], [ 2.,    1.,    1.,    2.,   -0.25],
                   [ 2.,    2.,    2.,    2.,    0.25], [ 2.,    3.,    3.,    2.,   -0.25],
                   [ 2.,    4.,    4.,    2.,    0.25], [ 2.,    5.,    5.,    2.,   -0.25],
                   [ 3.,    0.,    0.,    3.,   -0.25], [ 3.,    1.,    1.,    3.,    0.25],
                   [ 3.,    2.,    2.,    3.,   -0.25], [ 3.,    3.,    3.,    3.,    0.25],
                   [ 3.,    4.,    4.,    3.,   -0.25], [ 3.,    5.,    5.,    3.,    0.25],
                   [ 4.,    0.,    0.,    4.,    0.25], [ 4.,    1.,    1.,    4.,   -0.25],
                   [ 4.,    2.,    2.,    4.,    0.25], [ 4.,    3.,    3.,    4.,   -0.25],
                   [ 4.,    4.,    4.,    4.,    0.25], [ 4.,    5.,    5.,    4.,   -0.25],
                   [ 5.,    0.,    0.,    5.,   -0.25], [ 5.,    1.,    1.,    5.,    0.25],
                   [ 5.,    2.,    2.,    5.,   -0.25], [ 5.,    3.,    3.,    5.,    0.25],
                   [ 5.,    4.,    4.,    5.,   -0.25], [ 5.,    5.,    5.,    5.,    0.25],
                   [ 0.,    1.,    0.,    1.,    0.5 ], [ 0.,    3.,    2.,    1.,    0.5 ],
                   [ 0.,    5.,    4.,    1.,    0.5 ], [ 1.,    0.,    1.,    0.,    0.5 ],
                   [ 1.,    2.,    3.,    0.,    0.5 ], [ 1.,    4.,    5.,    0.,    0.5 ],
                   [ 2.,    1.,    0.,    3.,    0.5 ], [ 2.,    3.,    2.,    3.,    0.5 ],
                   [ 2.,    5.,    4.,    3.,    0.5 ], [ 3.,    0.,    1.,    2.,    0.5 ],
                   [ 3.,    2.,    3.,    2.,    0.5 ], [ 3.,    4.,    5.,    2.,    0.5 ],
                   [ 4.,    1.,    0.,    5.,    0.5 ], [ 4.,    3.,    2.,    5.,    0.5 ],
                   [ 4.,    5.,    4.,    5.,    0.5 ], [ 5.,    0.,    1.,    4.,    0.5 ],
                   [ 5.,    2.,    3.,    4.,    0.5 ], [ 5.,    4.,    5.,    4.,    0.5 ]])
@pytest.mark.parametrize(
    ("mol_name", "n_act_elect", "n_act_orb", "s2_me_exp", "init_term_exp"),
    [
        ('h2_pyscf', None, None, me,     1.5),
        ('h2_pyscf', 2,    None, me,     1.5),
        ('lih',      2,    2,    me,     1.5),
        ('lih',      None, 3,    me_lih, 3.0),
    ],
)
def test_get_s2_matrix_elements(
    mol_name,
    n_act_elect,
    n_act_orb,
    s2_me_exp,
    init_term_exp,
    tol
):
    r"""Test that the table of matrix elements and the term use to initialize the
    FermionOperator are computed corrrectly for different active spaces."""

    s2_me_res, init_term_res = obs.get_s2_me(
        mol_name,
        ref_dir,
        n_active_electrons=n_act_elect,
        n_active_orbitals=n_act_orb
    )

    assert np.allclose(s2_me_res, s2_me_exp, **tol)
    assert init_term_res == init_term_exp

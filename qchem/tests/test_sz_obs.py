import os

import numpy as np
import pytest

from pennylane import qchem

from openfermion.ops._qubit_operator import QubitOperator

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")

me = np.array([[ 0.,   0.,   0.5],
               [ 1.,   1.,  -0.5],
               [ 2.,   2.,   0.5],
               [ 3.,   3.,  -0.5]])

me_lih = np.array([[ 0.,   0.,   0.5],
                   [ 1.,   1.,  -0.5],
                   [ 2.,   2.,   0.5],
                   [ 3.,   3.,  -0.5],
                   [ 4.,   4.,   0.5],
                   [ 5.,   5.,  -0.5]])


@pytest.mark.parametrize(
    ("mol_name", "n_act_elect", "n_act_orb", "sz_me_exp"),
    [
        ("h2_pyscf", None, None, me),
        ("h2_pyscf", 2, None, me),
        ("lih", 2, 2, me),
        ("lih", None, 3, me_lih),
    ],
)
def test_get_spinZ_matrix_elements(
    mol_name, n_act_elect, n_act_orb, sz_me_exp, tol
):
    r"""Test that the table of matrix elements
    :math:`\langle \alpha \vert \hat{s}_z \vert \beta \rangle` are computed correctly
    for different choices of the active space."""

    sz_me_res = qchem.get_spinZ_matrix_elements(
        mol_name, ref_dir, n_active_electrons=n_act_elect, n_active_orbitals=n_act_orb
    )

    assert np.allclose(sz_me_res, sz_me_exp, **tol)

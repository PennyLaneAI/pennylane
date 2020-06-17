import os

import numpy as np
import pytest

from pennylane import qchem
from pennylane_qchem import obs

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

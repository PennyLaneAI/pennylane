import os

import numpy as np
import pytest

from pennylane import qchem
from openfermion.hamiltonians import MolecularData

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")


table_1 = np.array(
    [
        [0.0, 0.0, -32.70260436],
        [1.0, 1.0, -32.70260436],
        [0.0, 2.0, -0.5581082],
        [1.0, 3.0, -0.5581082],
        [0.0, 6.0, 0.23519027],
        [1.0, 7.0, 0.23519027],
        [0.0, 10.0, 0.30460521],
        [1.0, 11.0, 0.30460521],
        [2.0, 0.0, -0.5581082],
        [3.0, 1.0, -0.5581082],
        [2.0, 2.0, -7.6707491],
        [3.0, 3.0, -7.6707491],
        [2.0, 6.0, 0.43168603],
        [3.0, 7.0, 0.43168603],
        [2.0, 10.0, 1.38140486],
        [3.0, 11.0, 1.38140486],
        [4.0, 4.0, -6.36396432],
        [5.0, 5.0, -6.36396432],
        [4.0, 12.0, 1.70992104],
        [5.0, 13.0, 1.70992104],
        [6.0, 0.0, 0.23519027],
        [7.0, 1.0, 0.23519027],
        [6.0, 2.0, 0.43168603],
        [7.0, 3.0, 0.43168603],
        [6.0, 6.0, -6.98622104],
        [7.0, 7.0, -6.98622104],
        [6.0, 10.0, 1.08020193],
        [7.0, 11.0, 1.08020193],
        [8.0, 8.0, -7.4571701],
        [9.0, 9.0, -7.4571701],
        [10.0, 0.0, 0.30460521],
        [11.0, 1.0, 0.30460521],
        [10.0, 2.0, 1.38140486],
        [11.0, 3.0, 1.38140486],
        [10.0, 6.0, 1.08020193],
        [11.0, 7.0, 1.08020193],
        [10.0, 10.0, -5.33601654],
        [11.0, 11.0, -5.33601654],
        [12.0, 4.0, 1.70992104],
        [13.0, 5.0, 1.70992104],
        [12.0, 12.0, -5.60348511],
        [13.0, 13.0, -5.60348511],
    ]
)

table_2 = np.array(
    [
        [0.0, 0.0, -7.4571701],
        [1.0, 1.0, -7.4571701],
        [2.0, 2.0, -5.33601654],
        [3.0, 3.0, -5.33601654],
    ]
)

table_3 = np.array(
    [
        [0.0, 0.0, -7.4571701],
        [1.0, 1.0, -7.4571701],
        [2.0, 2.0, -5.33601654],
        [3.0, 3.0, -5.33601654],
        [4.0, 4.0, -5.60348511],
        [5.0, 5.0, -5.60348511],
    ]
)

table_4 = np.array(
    [
        [0.0, 0.0, -32.70260436],
        [1.0, 1.0, -32.70260436],
        [0.0, 2.0, -0.5581082],
        [1.0, 3.0, -0.5581082],
        [0.0, 6.0, 0.23519027],
        [1.0, 7.0, 0.23519027],
        [0.0, 10.0, 0.30460521],
        [1.0, 11.0, 0.30460521],
        [2.0, 0.0, -0.5581082],
        [3.0, 1.0, -0.5581082],
        [2.0, 2.0, -7.6707491],
        [3.0, 3.0, -7.6707491],
        [2.0, 6.0, 0.43168603],
        [3.0, 7.0, 0.43168603],
        [2.0, 10.0, 1.38140486],
        [3.0, 11.0, 1.38140486],
        [4.0, 4.0, -6.36396432],
        [5.0, 5.0, -6.36396432],
        [6.0, 0.0, 0.23519027],
        [7.0, 1.0, 0.23519027],
        [6.0, 2.0, 0.43168603],
        [7.0, 3.0, 0.43168603],
        [6.0, 6.0, -6.98622104],
        [7.0, 7.0, -6.98622104],
        [6.0, 10.0, 1.08020193],
        [7.0, 11.0, 1.08020193],
        [8.0, 8.0, -7.4571701],
        [9.0, 9.0, -7.4571701],
        [10.0, 0.0, 0.30460521],
        [11.0, 1.0, 0.30460521],
        [10.0, 2.0, 1.38140486],
        [11.0, 3.0, 1.38140486],
        [10.0, 6.0, 1.08020193],
        [11.0, 7.0, 1.08020193],
        [10.0, 10.0, -5.33601654],
        [11.0, 11.0, -5.33601654],
    ]
)


@pytest.mark.parametrize(
    ("core", "active", "table_exp", "t_core_exp"),
    [
        (None, None, table_1, 0),
        ([0, 1, 2, 3], [4, 5], table_2, -107.4470776470725),
        ([0, 1, 2, 3], None, table_3, -107.4470776470725),
        (None, [0, 1, 2, 3, 4, 5], table_4, 0),
    ],
)
def test_table_one_particle(core, active, table_exp, t_core_exp, tol):
    r"""Test the table of one-particle matrix elements and the contribution of core orbitals
    as implemented in the `one_particle` function of the `obs` module"""

    hf_data = MolecularData(filename=os.path.join(ref_dir, "h2o_psi4"))

    table, t_core = qchem.one_particle(hf_data.one_body_integrals, core=core, active=active)

    assert np.allclose(table, table_exp, **tol)
    assert np.allclose(t_core, t_core_exp, **tol)


table_1D = np.array([1, 2, 3])
table_2D = np.array([[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize(
    ("t_me", "core", "active", "msg_match"),
    [
        (table_1D, [0], None, "'matrix_elements' must be a 2D array"),
        (table_2D, [-1, 0, 1, 2], None, "Indices of core orbitals must be between 0 and"),
        (table_2D, [0, 1, 2, 3], None, "Indices of core orbitals must be between 0 and"),
        (table_2D, None, [-1, 0], "Indices of active orbitals must be between 0 and"),
        (table_2D, None, [2, 6], "Indices of active orbitals must be between 0 and"),
    ],
)
def test_exceptions_one_particle(t_me, core, active, msg_match):
    """Test that the function `'one_particle'` throws an exception
    if the dimension of the matrix elements array is not a 2D array or
    if the indices of core and/or active orbitals are out of range."""

    with pytest.raises(ValueError, match=msg_match):
        qchem.one_particle(t_me, core=core, active=active)

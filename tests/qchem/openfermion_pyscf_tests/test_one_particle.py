# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the ``one_particle`` function of FermionOperator
in openfermion.
"""
import os

import numpy as np
import pytest

from pennylane import qchem

ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")

t_op_1 = {
    (): 0.0,
    ((0, 1), (0, 0)): -32.7026043574631,
    ((1, 1), (1, 0)): -32.7026043574631,
    ((0, 1), (2, 0)): -0.5581081999989266,
    ((1, 1), (3, 0)): -0.5581081999989266,
    ((0, 1), (6, 0)): 0.23519027195177022,
    ((1, 1), (7, 0)): 0.23519027195177022,
    ((0, 1), (10, 0)): 0.30460521200741786,
    ((1, 1), (11, 0)): 0.30460521200741786,
    ((2, 1), (0, 0)): -0.5581081999989275,
    ((3, 1), (1, 0)): -0.5581081999989275,
    ((2, 1), (2, 0)): -7.670749097654825,
    ((3, 1), (3, 0)): -7.670749097654825,
    ((2, 1), (6, 0)): 0.43168602920745835,
    ((3, 1), (7, 0)): 0.43168602920745835,
    ((2, 1), (10, 0)): 1.3814048618806472,
    ((3, 1), (11, 0)): 1.3814048618806472,
    ((4, 1), (4, 0)): -6.363964323744818,
    ((5, 1), (5, 0)): -6.363964323744818,
    ((4, 1), (12, 0)): 1.709921037759953,
    ((5, 1), (13, 0)): 1.709921037759953,
    ((6, 1), (0, 0)): 0.23519027195177022,
    ((7, 1), (1, 0)): 0.23519027195177022,
    ((6, 1), (2, 0)): 0.4316860292074589,
    ((7, 1), (3, 0)): 0.4316860292074589,
    ((6, 1), (6, 0)): -6.986221044673506,
    ((7, 1), (7, 0)): -6.986221044673506,
    ((6, 1), (10, 0)): 1.0802019338400384,
    ((7, 1), (11, 0)): 1.0802019338400384,
    ((8, 1), (8, 0)): -7.457170098157133,
    ((9, 1), (9, 0)): -7.457170098157133,
    ((10, 1), (0, 0)): 0.3046052120074161,
    ((11, 1), (1, 0)): 0.3046052120074161,
    ((10, 1), (2, 0)): 1.3814048618806474,
    ((11, 1), (3, 0)): 1.3814048618806474,
    ((10, 1), (6, 0)): 1.0802019338400384,
    ((11, 1), (7, 0)): 1.0802019338400384,
    ((10, 1), (10, 0)): -5.336016542015546,
    ((11, 1), (11, 0)): -5.336016542015546,
    ((12, 1), (4, 0)): 1.7099210377599523,
    ((13, 1), (5, 0)): 1.7099210377599523,
    ((12, 1), (12, 0)): -5.60348510681844,
    ((13, 1), (13, 0)): -5.60348510681844,
}

t_op_2 = {
    (): -107.4470776470725,
    ((0, 1), (0, 0)): -7.457170098157133,
    ((1, 1), (1, 0)): -7.457170098157133,
    ((2, 1), (2, 0)): -5.336016542015546,
    ((3, 1), (3, 0)): -5.336016542015546,
}

t_op_3 = {
    (): -107.4470776470725,
    ((0, 1), (0, 0)): -7.457170098157133,
    ((1, 1), (1, 0)): -7.457170098157133,
    ((2, 1), (2, 0)): -5.336016542015546,
    ((3, 1), (3, 0)): -5.336016542015546,
    ((4, 1), (4, 0)): -5.60348510681844,
    ((5, 1), (5, 0)): -5.60348510681844,
}

t_op_4 = {
    (): 0.0,
    ((0, 1), (0, 0)): -32.7026043574631,
    ((1, 1), (1, 0)): -32.7026043574631,
    ((0, 1), (2, 0)): -0.5581081999989266,
    ((1, 1), (3, 0)): -0.5581081999989266,
    ((0, 1), (6, 0)): 0.23519027195177022,
    ((1, 1), (7, 0)): 0.23519027195177022,
    ((0, 1), (10, 0)): 0.30460521200741786,
    ((1, 1), (11, 0)): 0.30460521200741786,
    ((2, 1), (0, 0)): -0.5581081999989275,
    ((3, 1), (1, 0)): -0.5581081999989275,
    ((2, 1), (2, 0)): -7.670749097654825,
    ((3, 1), (3, 0)): -7.670749097654825,
    ((2, 1), (6, 0)): 0.43168602920745835,
    ((3, 1), (7, 0)): 0.43168602920745835,
    ((2, 1), (10, 0)): 1.3814048618806472,
    ((3, 1), (11, 0)): 1.3814048618806472,
    ((4, 1), (4, 0)): -6.363964323744818,
    ((5, 1), (5, 0)): -6.363964323744818,
    ((6, 1), (0, 0)): 0.23519027195177022,
    ((7, 1), (1, 0)): 0.23519027195177022,
    ((6, 1), (2, 0)): 0.4316860292074589,
    ((7, 1), (3, 0)): 0.4316860292074589,
    ((6, 1), (6, 0)): -6.986221044673506,
    ((7, 1), (7, 0)): -6.986221044673506,
    ((6, 1), (10, 0)): 1.0802019338400384,
    ((7, 1), (11, 0)): 1.0802019338400384,
    ((8, 1), (8, 0)): -7.457170098157133,
    ((9, 1), (9, 0)): -7.457170098157133,
    ((10, 1), (0, 0)): 0.3046052120074161,
    ((11, 1), (1, 0)): 0.3046052120074161,
    ((10, 1), (2, 0)): 1.3814048618806474,
    ((11, 1), (3, 0)): 1.3814048618806474,
    ((10, 1), (6, 0)): 1.0802019338400384,
    ((11, 1), (7, 0)): 1.0802019338400384,
    ((10, 1), (10, 0)): -5.336016542015546,
    ((11, 1), (11, 0)): -5.336016542015546,
}


@pytest.mark.parametrize(
    ("core", "active", "t_op_exp"),
    [
        (None, None, t_op_1),
        ([0, 1, 2, 3], [4, 5], t_op_2),
        ([0, 1, 2, 3], None, t_op_3),
        (None, [0, 1, 2, 3, 4, 5], t_op_4),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support")
def test_table_one_particle(core, active, t_op_exp):
    r"""Test the correctness of the FermionOperator built by the `'one_particle'` function
    of the `obs` module"""
    import openfermion

    hf = openfermion.MolecularData(filename=os.path.join(ref_dir, "h2o_psi4"))

    t_op = qchem.one_particle(hf.one_body_integrals, core=core, active=active)

    assert t_op.terms == t_op_exp


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
@pytest.mark.usefixtures("skip_if_no_openfermion_support")
def test_exceptions_one_particle(t_me, core, active, msg_match):
    """Test that the function `'one_particle'` throws an exception
    if the matrix elements array is not a 2D array or if the indices
    of core and/or active orbitals are out of range."""

    with pytest.raises(ValueError, match=msg_match):
        qchem.one_particle(t_me, core=core, active=active)

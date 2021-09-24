# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for functions needed for computing the Hamiltonian.
"""
import autograd
import pytest
from pennylane import numpy as np
from pennylane.hf.hamiltonian import generate_electron_integrals
from pennylane.hf.molecule import Molecule


@pytest.mark.parametrize(
    ("symbols", "geometry", "core", "active", "e_core", "one", "two"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            None,
            None,
            np.array([0.0]),
            # computed with OpenFermion using molecule.one_body_integrals.flatten()
            np.array([-1.39021927e00, -6.33695658e-17, 7.33430905e-17, -2.91653305e-01]),
            # computed with OpenFermion using molecule.two_body_integrals.flatten()
            np.array(
                [
                    7.14439079e-01,
                    2.61993266e-17,
                    -4.37704222e-17,
                    1.70241444e-01,
                    -4.37704222e-17,
                    1.70241444e-01,
                    7.01853156e-01,
                    -5.47923657e-16,
                    2.61993266e-17,
                    7.01853156e-01,
                    1.70241444e-01,
                    1.48039114e-16,
                    1.70241444e-01,
                    1.48039114e-16,
                    -5.47923657e-16,
                    7.38836693e-01,
                ]
            ),
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            [],
            [0, 1],
            np.array([0.0]),
            # computed with OpenFermion using molecule.one_body_integrals.flatten()
            np.array([-1.39021927e00, -6.33695658e-17, 7.33430905e-17, -2.91653305e-01]),
            # computed with OpenFermion using molecule.two_body_integrals.flatten()
            np.array(
                [
                    7.14439079e-01,
                    2.61993266e-17,
                    -4.37704222e-17,
                    1.70241444e-01,
                    -4.37704222e-17,
                    1.70241444e-01,
                    7.01853156e-01,
                    -5.47923657e-16,
                    2.61993266e-17,
                    7.01853156e-01,
                    1.70241444e-01,
                    1.48039114e-16,
                    1.70241444e-01,
                    1.48039114e-16,
                    -5.47923657e-16,
                    7.38836693e-01,
                ]
            ),
        ),
    ],
)
def test_generate_electron_integrals(symbols, geometry, core, active, e_core, one, two):
    r"""Test that generate_electron_integrals returns the correct values."""
    mol = Molecule(symbols, geometry)
    args = []
    result = generate_electron_integrals(mol, core=core, active=active)(*args)

    assert np.allclose(result, np.concatenate((e_core, one, two)))

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
import pytest
from pennylane import numpy as np
from pennylane.hf.hamiltonian import generate_electron_integrals
from pennylane.hf.molecule import Molecule


@pytest.mark.parametrize(
    ("symbols", "geometry", "core", "active", "e_core", "one_ref", "two_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            None,
            None,
            np.array([1.0000000000321256]),
            # computed with OpenFermion using molecule.one_body_integrals
            np.array([[-1.39021927e00, -1.28555566e-16], [-3.52805508e-16, -2.91653305e-01]]),
            # computed with OpenFermion using molecule.two_body_integrals
            np.array(
                [
                    [
                        [[7.14439079e-01, 6.62555256e-17], [2.45552260e-16, 1.70241444e-01]],
                        [[2.45552260e-16, 1.70241444e-01], [7.01853156e-01, 6.51416091e-16]],
                    ],
                    [
                        [[6.62555256e-17, 7.01853156e-01], [1.70241444e-01, 2.72068603e-16]],
                        [[1.70241444e-01, 2.72068603e-16], [6.51416091e-16, 7.38836693e-01]],
                    ],
                ]
            ),
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            [],
            [0, 1],
            np.array([1.0000000000321256]),
            # computed with OpenFermion using molecule.one_body_integrals
            np.array([[-1.39021927e00, -1.28555566e-16], [-3.52805508e-16, -2.91653305e-01]]),
            # computed with OpenFermion using molecule.two_body_integrals
            np.array(
                [
                    [
                        [[7.14439079e-01, 6.62555256e-17], [2.45552260e-16, 1.70241444e-01]],
                        [[2.45552260e-16, 1.70241444e-01], [7.01853156e-01, 6.51416091e-16]],
                    ],
                    [
                        [[6.62555256e-17, 7.01853156e-01], [1.70241444e-01, 2.72068603e-16]],
                        [[1.70241444e-01, 2.72068603e-16], [6.51416091e-16, 7.38836693e-01]],
                    ],
                ]
            ),
        ),
        (
            ["Li", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            [0, 1, 2, 3],
            [4, 5],
            # reference values of e_core, one and two are computed with our initial prototype code
            np.array([-5.141222763432437]),
            np.array([[1.17563204e00, -5.75186616e-18], [-5.75186616e-18, 1.78830226e00]]),
            np.array(
                [
                    [
                        [[3.12945511e-01, 4.79898448e-19], [4.79898448e-19, 9.78191587e-03]],
                        [[4.79898448e-19, 9.78191587e-03], [3.00580620e-01, 4.28570365e-18]],
                    ],
                    [
                        [[4.79898448e-19, 3.00580620e-01], [9.78191587e-03, 4.28570365e-18]],
                        [[9.78191587e-03, 4.28570365e-18], [4.28570365e-18, 5.10996835e-01]],
                    ],
                ]
            ),
        ),
    ],
)
def test_generate_electron_integrals(symbols, geometry, core, active, e_core, one_ref, two_ref):
    r"""Test that generate_electron_integrals returns the correct values."""
    mol = Molecule(symbols, geometry)
    args = []

    e, one, two = generate_electron_integrals(mol, core=core, active=active)(*args)

    assert np.allclose(e, e_core)
    assert np.allclose(one, one_ref)
    assert np.allclose(two, two_ref)

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
from pennylane.hf.hamiltonian import (
    generate_electron_integrals,
    generate_fermionic_hamiltonian,
    _generate_qubit_operator,
    _pauli_mult,
)
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


@pytest.mark.parametrize(
    ("symbols", "geometry", "alpha", "coeffs_h_ref", "ops_h_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            np.array(
                [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
                requires_grad=True,
            ),
            # Hamiltonian coefficients and operators computed with OpenFermion using
            # openfermion.transforms.get_fermion_operator(molecule.get_molecular_hamiltonian())
            # The "^" symbols in the operators are removed and "," is added for consistency
            np.array(
                [
                    1.0000000000321256,
                    -1.3902192706002598,
                    0.35721953951840535,
                    0.08512072192002007,
                    0.35721953951840535,
                    0.08512072192002007,
                    0.08512072192002007,
                    0.35092657803574406,
                    0.08512072192002007,
                    0.35092657803574406,
                    0.35721953951840535,
                    0.08512072192002007,
                    -1.3902192706002598,
                    0.35721953951840535,
                    0.08512072192002007,
                    0.08512072192002007,
                    0.35092657803574406,
                    0.08512072192002007,
                    0.35092657803574406,
                    0.35092657803574495,
                    0.08512072192002007,
                    0.35092657803574495,
                    0.08512072192002007,
                    -0.2916533049477536,
                    0.08512072192002007,
                    0.3694183466586136,
                    0.08512072192002007,
                    0.3694183466586136,
                    0.35092657803574495,
                    0.08512072192002007,
                    0.35092657803574495,
                    0.08512072192002007,
                    0.08512072192002007,
                    0.3694183466586136,
                    -0.2916533049477536,
                    0.08512072192002007,
                    0.3694183466586136,
                ]
            ),
            [
                [],
                [0, 0],
                [0, 0, 0, 0],
                [0, 0, 2, 2],
                [0, 1, 1, 0],
                [0, 1, 3, 2],
                [0, 2, 0, 2],
                [0, 2, 2, 0],
                [0, 3, 1, 2],
                [0, 3, 3, 0],
                [1, 0, 0, 1],
                [1, 0, 2, 3],
                [1, 1],
                [1, 1, 1, 1],
                [1, 1, 3, 3],
                [1, 2, 0, 3],
                [1, 2, 2, 1],
                [1, 3, 1, 3],
                [1, 3, 3, 1],
                [2, 0, 0, 2],
                [2, 0, 2, 0],
                [2, 1, 1, 2],
                [2, 1, 3, 0],
                [2, 2],
                [2, 2, 0, 0],
                [2, 2, 2, 2],
                [2, 3, 1, 0],
                [2, 3, 3, 2],
                [3, 0, 0, 3],
                [3, 0, 2, 1],
                [3, 1, 1, 3],
                [3, 1, 3, 1],
                [3, 2, 0, 1],
                [3, 2, 2, 3],
                [3, 3],
                [3, 3, 1, 1],
                [3, 3, 3, 3],
            ],
        )
    ],
)
def test_generate_fermionic_hamiltonian(symbols, geometry, alpha, coeffs_h_ref, ops_h_ref):
    r"""Test that fermionic_hamiltonian returns the correct Hamiltonian."""
    mol = Molecule(symbols, geometry, alpha=alpha)
    args = [alpha]
    h = generate_fermionic_hamiltonian(mol)(*args)

    assert np.allclose(h[0], coeffs_h_ref)
    assert h[1] == ops_h_ref


@pytest.mark.parametrize(
    ("f_operator", "q_operator"),
    [
        (
            [0, 0],
            # obtained with openfermion using jordan_wigner(FermionOperator('0^ 0', 1)),
            # reformatted the original openfermion output: (0.5+0j) [] + (-0.5+0j) [Z0]
            ([(0.5 + 0j), (-0.5 + 0j)], [[], [(0, "Z")]]),
        ),
    ],
)
def test_generate_qubit_operator(f_operator, q_operator):
    r"""Test that _generate_qubit_operator returns the correct operator."""
    result = _generate_qubit_operator(f_operator)

    assert result == q_operator


@pytest.mark.parametrize(
    ("p1", "p2", "c1", "c2", "p_ref"),
    [
        (
            [(0, "X"), (1, "Y")],  # X_0 @ Y_1
            [(0, "X"), (2, "Y")],  # X_0 @ Y_2
            0.2,
            0.6,
            ([(2, "Y"), (1, "Y")], 0.12),  # 0.2 * 0.6 * X_0 @ Y_1 @ X_0 @ Y_2
        ),
    ],
)
def test_pauli_mult(p1, p2, c1, c2, p_ref):
    r"""Test that _generate_qubit_operator returns the correct operator."""
    result = _pauli_mult(p1, p2, c1, c2)

    assert result == p_ref

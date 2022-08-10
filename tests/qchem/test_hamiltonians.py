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

import pennylane as qml
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import numpy as np
from pennylane import qchem


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
def test_electron_integrals(symbols, geometry, core, active, e_core, one_ref, two_ref):
    r"""Test that electron_integrals returns the correct values."""
    mol = qchem.Molecule(symbols, geometry)
    args = []

    e, one, two = qchem.electron_integrals(mol, core=core, active=active)(*args)

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
def test_fermionic_hamiltonian(symbols, geometry, alpha, coeffs_h_ref, ops_h_ref):
    r"""Test that fermionic_hamiltonian returns the correct Hamiltonian."""
    mol = qchem.Molecule(symbols, geometry, alpha=alpha)
    args = [alpha]
    h = qchem.fermionic_hamiltonian(mol)(*args)

    assert np.allclose(h[0], coeffs_h_ref)
    assert h[1] == ops_h_ref


@pytest.mark.parametrize(
    ("symbols", "geometry", "h_ref_data"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=False),
            # computed with qchem.convert_observable and an OpenFermion Hamiltonian; data reordered
            # h_mol = molecule.get_molecular_hamiltonian()
            # h_f = openfermion.transforms.get_fermion_operator(h_mol)
            # h_q = openfermion.transforms.jordan_wigner(h_f)
            # h_pl = qchem.convert_observable(h_q, wires=[0, 1, 2, 3], tol=(5e-5))
            (
                np.array(
                    [
                        0.2981788017,
                        0.2081336485,
                        0.2081336485,
                        0.1786097698,
                        0.042560361,
                        -0.042560361,
                        -0.042560361,
                        0.042560361,
                        -0.3472487379,
                        0.1329029281,
                        -0.3472487379,
                        0.175463289,
                        0.175463289,
                        0.1329029281,
                        0.1847091733,
                    ]
                ),
                [
                    Identity(wires=[0]),
                    PauliZ(wires=[0]),
                    PauliZ(wires=[1]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                    PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
                    PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
                    PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
                    PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
                    PauliZ(wires=[2]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[3]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[2]) @ PauliZ(wires=[3]),
                ],
            ),
        )
    ],
)
def test_diff_hamiltonian(symbols, geometry, h_ref_data):
    r"""Test that diff_hamiltonian returns the correct Hamiltonian."""

    mol = qchem.Molecule(symbols, geometry)
    args = []
    h = qchem.diff_hamiltonian(mol)(*args)
    h_ref = qml.Hamiltonian(h_ref_data[0], h_ref_data[1])

    assert np.allclose(h.terms()[0], h_ref.terms()[0])
    assert qml.Hamiltonian(np.ones(len(h.terms()[0])), h.terms()[1]).compare(
        qml.Hamiltonian(np.ones(len(h_ref.terms()[0])), h_ref.terms()[1])
    )


def test_gradient_expvalH():
    r"""Test that the gradient of expval(H) computed with ``autograd.grad`` is equal to the value
    obtained with the finite difference method."""
    symbols = ["H", "H"]
    geometry = (
        np.array([[0.0, 0.0, -0.3674625962], [0.0, 0.0, 0.3674625962]], requires_grad=False)
        / 0.529177210903
    )
    alpha = np.array(
        [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
        requires_grad=True,
    )

    mol = qchem.Molecule(symbols, geometry, alpha=alpha)
    args = [alpha]
    dev = qml.device("default.qubit", wires=4)

    def energy(mol):
        @qml.qnode(dev)
        def circuit(*args):
            qml.PauliX(0)
            qml.PauliX(1)
            qml.DoubleExcitation(0.22350048111151138, wires=[0, 1, 2, 3])
            h_qubit = qchem.diff_hamiltonian(mol)(*args)
            return qml.expval(h_qubit)

        return circuit

    grad_autograd = autograd.grad(energy(mol), argnum=0)(*args)

    alpha_1 = np.array(
        [[3.42515091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
        requires_grad=False,
    )  # alpha[0][0] -= 0.0001

    alpha_2 = np.array(
        [[3.42535091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]],
        requires_grad=False,
    )  # alpha[0][0] += 0.0001

    e_1 = energy(mol)(*[alpha_1])
    e_2 = energy(mol)(*[alpha_2])

    grad_finitediff = (e_2 - e_1) / 0.0002

    assert np.allclose(grad_autograd[0][0], grad_finitediff)

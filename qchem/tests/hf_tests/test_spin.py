# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for functions needed for computing the spin observables.
"""
import pytest

import pennylane as qml
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import numpy as np
from pennylane import qchem


@pytest.mark.parametrize(
    ("n_spin_orbs", "matrix_ref"),
    [
        (1, [0.0, 0.0, 0.0, 0.0, 0.25]),
        (
            3,
            [
                [0.0, 0.0, 0.0, 0.0, 0.25],
                [0.0, 1.0, 1.0, 0.0, -0.25],
                [0.0, 2.0, 2.0, 0.0, 0.25],
                [1.0, 0.0, 0.0, 1.0, -0.25],
                [1.0, 1.0, 1.0, 1.0, 0.25],
                [1.0, 2.0, 2.0, 1.0, -0.25],
                [2.0, 0.0, 0.0, 2.0, 0.25],
                [2.0, 1.0, 1.0, 2.0, -0.25],
                [2.0, 2.0, 2.0, 2.0, 0.25],
                [0.0, 1.0, 0.0, 1.0, 0.5],
                [1.0, 0.0, 1.0, 0.0, 0.5],
            ],
        ),
        (
            6,
            [
                [0.0, 0.0, 0.0, 0.0, 0.25],
                [0.0, 1.0, 1.0, 0.0, -0.25],
                [0.0, 2.0, 2.0, 0.0, 0.25],
                [0.0, 3.0, 3.0, 0.0, -0.25],
                [0.0, 4.0, 4.0, 0.0, 0.25],
                [0.0, 5.0, 5.0, 0.0, -0.25],
                [1.0, 0.0, 0.0, 1.0, -0.25],
                [1.0, 1.0, 1.0, 1.0, 0.25],
                [1.0, 2.0, 2.0, 1.0, -0.25],
                [1.0, 3.0, 3.0, 1.0, 0.25],
                [1.0, 4.0, 4.0, 1.0, -0.25],
                [1.0, 5.0, 5.0, 1.0, 0.25],
                [2.0, 0.0, 0.0, 2.0, 0.25],
                [2.0, 1.0, 1.0, 2.0, -0.25],
                [2.0, 2.0, 2.0, 2.0, 0.25],
                [2.0, 3.0, 3.0, 2.0, -0.25],
                [2.0, 4.0, 4.0, 2.0, 0.25],
                [2.0, 5.0, 5.0, 2.0, -0.25],
                [3.0, 0.0, 0.0, 3.0, -0.25],
                [3.0, 1.0, 1.0, 3.0, 0.25],
                [3.0, 2.0, 2.0, 3.0, -0.25],
                [3.0, 3.0, 3.0, 3.0, 0.25],
                [3.0, 4.0, 4.0, 3.0, -0.25],
                [3.0, 5.0, 5.0, 3.0, 0.25],
                [4.0, 0.0, 0.0, 4.0, 0.25],
                [4.0, 1.0, 1.0, 4.0, -0.25],
                [4.0, 2.0, 2.0, 4.0, 0.25],
                [4.0, 3.0, 3.0, 4.0, -0.25],
                [4.0, 4.0, 4.0, 4.0, 0.25],
                [4.0, 5.0, 5.0, 4.0, -0.25],
                [5.0, 0.0, 0.0, 5.0, -0.25],
                [5.0, 1.0, 1.0, 5.0, 0.25],
                [5.0, 2.0, 2.0, 5.0, -0.25],
                [5.0, 3.0, 3.0, 5.0, 0.25],
                [5.0, 4.0, 4.0, 5.0, -0.25],
                [5.0, 5.0, 5.0, 5.0, 0.25],
                [0.0, 1.0, 0.0, 1.0, 0.5],
                [0.0, 3.0, 2.0, 1.0, 0.5],
                [0.0, 5.0, 4.0, 1.0, 0.5],
                [1.0, 0.0, 1.0, 0.0, 0.5],
                [1.0, 2.0, 3.0, 0.0, 0.5],
                [1.0, 4.0, 5.0, 0.0, 0.5],
                [2.0, 1.0, 0.0, 3.0, 0.5],
                [2.0, 3.0, 2.0, 3.0, 0.5],
                [2.0, 5.0, 4.0, 3.0, 0.5],
                [3.0, 0.0, 1.0, 2.0, 0.5],
                [3.0, 2.0, 3.0, 2.0, 0.5],
                [3.0, 4.0, 5.0, 2.0, 0.5],
                [4.0, 1.0, 0.0, 5.0, 0.5],
                [4.0, 3.0, 2.0, 5.0, 0.5],
                [4.0, 5.0, 4.0, 5.0, 0.5],
                [5.0, 0.0, 1.0, 4.0, 0.5],
                [5.0, 2.0, 3.0, 4.0, 0.5],
                [5.0, 4.0, 5.0, 4.0, 0.5],
            ],
        ),
    ],
)
def test_spin2_matrix_elements(n_spin_orbs, matrix_ref):
    r"""Test the calculation of the matrix elements of the two-particle spin operator
    :math:`\hat{s}_1 \cdot \hat{s}_2` implemented by the function `'_spin2_matrix_elements'`"""

    sz = np.where(np.arange(n_spin_orbs) % 2 == 0, 0.5, -0.5)

    s2_me_result = qchem.spin._spin2_matrix_elements(sz)

    assert np.allclose(s2_me_result, matrix_ref)


@pytest.mark.parametrize(
    ("electrons", "orbitals", "coeffs_ref", "ops_ref"),
    [
        (  # computed with PL-QChem using OpenFermion
            2,
            4,
            np.array(
                [
                    0.75,
                    0.375,
                    -0.375,
                    0.125,
                    0.375,
                    -0.125,
                    -0.125,
                    0.125,
                    0.375,
                    0.375,
                    -0.375,
                    0.125,
                    0.125,
                    0.125,
                    -0.125,
                    -0.125,
                    0.125,
                    0.125,
                    0.125,
                ]
            ),
            [
                Identity(wires=[0]),
                PauliZ(wires=[1]),
                PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                PauliZ(wires=[0]) @ PauliZ(wires=[2]),
                PauliZ(wires=[0]),
                PauliZ(wires=[0]) @ PauliZ(wires=[3]),
                PauliZ(wires=[1]) @ PauliZ(wires=[2]),
                PauliZ(wires=[1]) @ PauliZ(wires=[3]),
                PauliZ(wires=[2]),
                PauliZ(wires=[3]),
                PauliZ(wires=[2]) @ PauliZ(wires=[3]),
                PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
                PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
                PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
                PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
                PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
                PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
                PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
                PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
            ],
        ),
    ],
)
def test_spin2(electrons, orbitals, coeffs_ref, ops_ref):
    r"""Tests the correctness of the total spin observable :math:`\hat{S}^2`
    built by the function `'spin2'`.
    """
    s2 = qchem.spin.spin2(electrons, orbitals)
    s2_ref = qml.Hamiltonian(coeffs_ref, ops_ref)

    assert s2.compare(s2_ref)


@pytest.mark.parametrize(
    ("electrons", "orbitals", "msg_match"),
    [
        (-2, 4, "'electrons' must be greater than 0"),
        (0, 4, "'electrons' must be greater than 0"),
        (3, -6, "'orbitals' must be greater than 0"),
        (3, 0, "'orbitals' must be greater than 0"),
    ],
)
def test_exception_spin2(electrons, orbitals, msg_match):
    """Test that the function `'spin2'` throws an exception if the
    number of electrons or the number of orbitals is less than zero."""

    with pytest.raises(ValueError, match=msg_match):
        qchem.spin.spin2(electrons, orbitals)


@pytest.mark.parametrize(
    ("orbitals", "coeffs_ref", "ops_ref"),
    [
        (  # computed with PL-QChem using OpenFermion
            4,
            np.array([-0.25, 0.25, -0.25, 0.25]),
            [PauliZ(wires=[0]), PauliZ(wires=[1]), PauliZ(wires=[2]), PauliZ(wires=[3])],
        ),
        (
            6,
            np.array([-0.25, 0.25, -0.25, 0.25, -0.25, 0.25]),
            [
                PauliZ(wires=[0]),
                PauliZ(wires=[1]),
                PauliZ(wires=[2]),
                PauliZ(wires=[3]),
                PauliZ(wires=[4]),
                PauliZ(wires=[5]),
            ],
        ),
    ],
)
def test_spinz(orbitals, coeffs_ref, ops_ref):
    r"""Tests the correctness of the :math:`\hat{S}_z` observable built by the
    function `'spin_z'`.
    """
    sz = qchem.spin.spinz(orbitals)
    sz_ref = qml.Hamiltonian(coeffs_ref, ops_ref)

    assert sz.compare(sz_ref)


@pytest.mark.parametrize(
    ("orbitals", "msg_match"),
    [
        (-3, "'orbitals' must be greater than 0"),
        (0, "'orbitals' must be greater than 0"),
    ],
)
def test_exception_spinz(orbitals, msg_match):
    """Test that the function `'spin_z'` throws an exception if the
    number of orbitals is less than zero."""

    with pytest.raises(ValueError, match=msg_match):
        qchem.spin.spinz(orbitals)

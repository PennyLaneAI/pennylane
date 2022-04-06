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
Unit tests for functions needed for computing the particle number observable.
"""
import pytest

import pennylane as qml
from pennylane import Identity, PauliZ
from pennylane import numpy as np
from pennylane import qchem


@pytest.mark.parametrize(
    ("orbitals", "coeffs_ref", "ops_ref"),
    [
        (  # computed with PL-QChem using OpenFermion
            4,
            np.array([2.0, -0.5, -0.5, -0.5, -0.5]),
            [
                Identity(wires=[0]),
                PauliZ(wires=[0]),
                PauliZ(wires=[1]),
                PauliZ(wires=[2]),
                PauliZ(wires=[3]),
            ],
        ),
        (
            6,
            np.array([3.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]),
            [
                Identity(wires=[0]),
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
def test_particle_number(orbitals, coeffs_ref, ops_ref):
    r"""Tests the correctness of the :math:`\hat{S}_z` observable built by the
    function `'spin_z'`.
    """
    n = qchem.particle_number(orbitals)
    n_ref = qml.Hamiltonian(coeffs_ref, ops_ref)

    assert n.compare(n_ref)


@pytest.mark.parametrize(
    ("orbitals", "msg_match"),
    [
        (-3, "'orbitals' must be greater than 0"),
        (0, "'orbitals' must be greater than 0"),
    ],
)
def test_exception_particle_number(orbitals, msg_match):
    """Test that the function `'particle_number'` throws an exception if the
    number of orbitals is less than zero."""

    with pytest.raises(ValueError, match=msg_match):
        qchem.particle_number(orbitals)

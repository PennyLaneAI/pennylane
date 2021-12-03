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
Unit tests for functions needed for qubit tapering.
"""
import pennylane as qml
import pytest
from pennylane import numpy as np
from pennylane.hf.tapering import clifford, observable_mult, simplify, transform_hamiltonian


@pytest.mark.parametrize(
    ("obs_a", "obs_b", "result"),
    [
        (
            qml.Hamiltonian(np.array([-1.0]), [qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2)]),
            qml.Hamiltonian(np.array([-1.0]), [qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliX(2)]),
            qml.Hamiltonian(np.array([1.0]), [qml.Identity(0)]),
        ),
        (
            qml.Hamiltonian(
                np.array([0.5, 0.5]), [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliZ(1)]
            ),
            qml.Hamiltonian(
                np.array([0.5, 0.5]), [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(1)]
            ),
            qml.Hamiltonian(
                np.array([-0.25j, 0.25j, -0.25j, 0.25]),
                [qml.PauliY(0), qml.PauliY(1), qml.PauliZ(1), qml.PauliY(0) @ qml.PauliX(1)],
            ),
        ),
    ],
)
def test_observable_mult(obs_a, obs_b, result):
    r"""Test that observable_mult returns the correct result."""
    o = observable_mult(obs_a, obs_b)
    assert o.compare(result)


@pytest.mark.parametrize(
    ("generator", "paulix_wires", "result"),
    [
        (
            [
                qml.Hamiltonian(np.array([1.0]), [qml.PauliZ(0)]),
                qml.Hamiltonian(np.array([1.0]), [qml.PauliZ(1)]),
            ],
            [0, 1],
            qml.Hamiltonian(
                np.array(
                    [
                        (1 / np.sqrt(2)) ** 2,
                        (1 / np.sqrt(2)) ** 2,
                        (1 / np.sqrt(2)) ** 2,
                        (1 / np.sqrt(2)) ** 2,
                    ]
                ),
                [
                    qml.PauliZ(0) @ qml.PauliZ(1),
                    qml.PauliZ(0) @ qml.PauliX(1),
                    qml.PauliX(0) @ qml.PauliZ(1),
                    qml.PauliX(0) @ qml.PauliX(1),
                ],
            ),
        ),
    ],
)
def test_cliford(generator, paulix_wires, result):
    r"""Test that clifford returns the correct operator."""
    u = clifford(generator, paulix_wires)
    assert u.compare(result)


@pytest.mark.parametrize(
    ("hamiltonian", "result"),
    [
        (
            qml.Hamiltonian(
                np.array([0.5, 0.5]), [qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(0) @ qml.PauliY(1)]
            ),
            qml.Hamiltonian(np.array([1.0]), [qml.PauliX(0) @ qml.PauliY(1)]),
        ),
    ],
)
def test_simplify(hamiltonian, result):
    r"""Test that simplify returns the correct hamiltonian."""
    h = simplify(hamiltonian)
    assert h.compare(result)


@pytest.mark.parametrize(
    ("symbols", "geometry", "generator", "paulix_wires", "paulix_sector", "ham_ref"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]], requires_grad=False),
            [
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(2)]),
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(3)]),
            ],
            [1, 2, 3],
            [[1, -1, -1]],
            qml.Hamiltonian(
                np.array([0.79596785, -0.3210344, 0.18092703]),
                [qml.PauliZ(0), qml.PauliX(0), qml.Identity(0)],
            ),
        ),
    ],
)
def test_transform_hamiltonian(symbols, geometry, generator, paulix_wires, paulix_sector, ham_ref):
    r"""Test that transform_hamiltonian returns the correct hamiltonian."""
    mol = qml.hf.Molecule(symbols, geometry)
    h = qml.hf.generate_hamiltonian(mol)()
    output_sector, h_trans = transform_hamiltonian(h, generator, paulix_wires, paulix_sector)[0]

    assert output_sector == paulix_sector[0]
    assert np.allclose(sorted(h_trans.terms[0]), sorted(ham_ref.terms[0]))

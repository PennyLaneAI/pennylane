# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for choi matrix in qml.math"""
import pytest

import pennylane as qml
from pennylane import math
from pennylane.math import choi_matrix

# tf = pytest.importorskip("tensorflow", minversion="2.1")
# torch = pytest.importorskip("torch")
# jax = pytest.importorskip("jax")
# jnp = pytest.importorskip("jax.numpy")

Ks1 = [qml.matrix(qml.CNOT((0, 1)))]  # a simple unitary channel
Ks2 = [
    math.sqrt(0.5) * qml.matrix(qml.CNOT((0, 1))),
    math.sqrt(0.5) * qml.matrix(qml.CZ((0, 1))),
]  # equal probability channel
coeffs = math.arange(1, 5)
coeffs = coeffs / math.linalg.norm(coeffs)
Us = [
    qml.CNOT((0, 1)),
    qml.exp(-1j * 0.5 * (qml.X(0) + qml.Y(1) + qml.Z(0) @ qml.Z(1))),
    qml.X(0),
    qml.Z(1),
]
Us = [qml.matrix(U, wire_order=range(2)) for U in Us]
Ks3 = [coeffs[j] * Us[j] for j in range(len(Us))]


@pytest.mark.parametrize("Ks", [Ks1, Ks2, Ks3])
def test_density_matrix(Ks):
    """Test that the resulting choi matrix is a density matrix"""
    choi = choi_matrix(Ks)
    assert math.isclose(math.trace(choi), 1.0), "not a density matrix, tr(choi) != 1"
    assert math.allclose(choi, choi.conj().T), "not a density matrix, not Hermitian"
    lambdas = math.linalg.eigvalsh(choi)
    assert math.all(math.round(lambdas, 8) >= 0), "not a density matrix, not positive"

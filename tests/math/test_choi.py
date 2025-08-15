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
import numpy as np
import pytest

import pennylane as qml
from pennylane import math
from pennylane.math import choi_matrix

Ks1 = np.array([qml.matrix(qml.CNOT((0, 1)))])  # a simple unitary channel
Ks2 = np.array(
    [
        math.sqrt(0.5) * qml.matrix(qml.CNOT((0, 1))),
        math.sqrt(0.5) * qml.matrix(qml.CZ((0, 1))),
    ]
)  # equal probability channel
coeffs = math.arange(1, 5)
coeffs = coeffs / math.linalg.norm(coeffs)
Us = [
    qml.CNOT((0, 1)),
    qml.exp(-1j * 0.5 * (qml.X(0) + qml.Y(1) + qml.Z(0) @ qml.Z(1))),
    qml.X(0),
    qml.Z(1),
]
Us = [qml.matrix(U, wire_order=range(2)) for U in Us]
Ks3 = np.array([coeffs[j] * Us[j] for j in range(len(Us))])


@pytest.mark.all_interfaces
@pytest.mark.parametrize("interface", [None, "autograd", "jax", "torch"])
@pytest.mark.parametrize("Ks", [Ks1, Ks2, Ks3])
def test_density_matrix(Ks, interface):
    """Test that the resulting choi matrix is valid, i.e. a density matrix"""

    if interface:
        Ks = qml.math.asarray(np.array(Ks), like=interface)

    choi = choi_matrix(Ks)
    val_tr = math.trace(choi)
    assert math.isclose(val_tr, math.ones_like(val_tr)), "not a density matrix, tr(choi) != 1"
    assert math.allclose(
        choi, math.transpose(math.conj(choi))
    ), "not a density matrix, not Hermitian"
    lambdas = math.linalg.eigvalsh(choi)
    assert math.all(
        math.asarray(lambdas, like="numpy") >= -1e-7
    ), "not a density matrix, not positive"


def test_error_message():
    """Test that an error is raised when input Kraus operators are not trace-preserving and check_Ks is set to True"""
    # easiest way to construct a non-trace-preserving channel is use non-normalized unitary operators
    Ks_non_trace_preserving = [qml.matrix(qml.CNOT((0, 1))), qml.matrix(qml.CZ((0, 1)))]
    with pytest.raises(ValueError, match="The provided Kraus operators are not trace-preserving"):
        _ = choi_matrix(Ks_non_trace_preserving, check_Ks=True)

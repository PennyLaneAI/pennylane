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
Unit tests for functions needed for estimating the number of logical qubits and non-Clifford gates
for quantum algorithms in first quantization using a plane-wave basis.
"""
import pytest
import pennylane as qml
from pennylane import numpy as np


@pytest.mark.parametrize(
    ("n_basis", "br", "prob_ref"),
    [
        # prob_ref computed with TFermion
        (1, 7, 1.0),
        (10000, 7, 0.9998814293823286),
    ],
)
def test_success_prob(n_basis, br, prob_ref):
    r"""Test that success_prob returns the correct value."""
    prob = qml.resources.success_prob(n_basis, br)

    assert prob == prob_ref


@pytest.mark.parametrize(
    ("eta", "n", "omega", "br", "charge", "norm_ref"),
    [
        (156, 10000, 1145.166, 7, 0,  1254382.0657073155),
    ],
)
def test_norm(eta, n, omega, br, charge, norm_ref):
    r"""Test that norm returns the correct value."""
    norm = qml.resources.norm(eta, n, omega, br, charge)

    assert norm == norm_ref

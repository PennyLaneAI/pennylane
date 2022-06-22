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
    ("n_basis", "br"),
    [
        (-1, 7),
        (1.2, 7),
        (10, 7.2),
        (10, -7),
    ],
)
def test_success_prob_error(n_basis, br):
    r"""Test that success_prob raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be a positive integer"):
        qml.resources.success_prob(n_basis, br)


@pytest.mark.parametrize(
    ("eta", "n", "omega", "error", "br", "charge", "norm_ref"),
    [
        (156, 10000, 1145.166, 0.001, 7, 0, 281053.7561247674),
    ],
)
def test_norm(eta, n, omega, error, br, charge, norm_ref):
    r"""Test that norm_fq returns the correct value."""
    norm = qml.resources.norm_fq(eta, n, omega, error, br, charge)

    assert np.allclose(norm, norm_ref)


@pytest.mark.parametrize(
    ("eta", "n", "omega", "error", "br", "charge"),
    [
        (156.2, 10000, 1145.166, 0.001, 7, 0),
        (-156, 10000, 1145.166, 0.001, 7, 0),
        (156, 10000.5, 1145.166, 0.001, 7, 0),
        (156, -10000, 1145.166, 0.001, 7, 0),
        (156, 10000, -1145.166, 0.001, 7, 0),
        (156, 10000, 1145.166, -0.001, 7, 0),
        (156, 10000, 1145.166, 0.001, 7.5, 0),
        (156, 10000, 1145.166, 0.001, -7, 0),
        (156, 10000, 1145.166, 0.001, 7, 1.2),
    ],
)
def test_norm_error(eta, n, omega, error, br, charge):
    r"""Test that norm_fq raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resources.norm_fq(eta, n, omega, error, br, charge)

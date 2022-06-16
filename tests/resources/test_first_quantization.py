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
    ("k", "lz", "cost_ref"),
    [
        # the reference cost is computed manually
        (4, 100, 23),
    ],
)
def test_cost_qrom(k, lz, cost_ref):
    r"""Test that _cost_qrom returns the correct value."""
    cost = qml.resources._cost_qrom(k, lz)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("lz", "cost_ref"),
    [
        # the reference cost is obtained manually by computing the cost for a range of k values, as
        # powers of two and selecting the minimum cost.
        (20, 9),
        (100, 21),
        (300, 35),
    ],
)
def test_cost_qrom_min(lz, cost_ref):
    r"""Test that _cost_qrom_min returns the correct value."""
    cost = qml.resources._cost_qrom_min(lz)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "lamb", "br", "charge", "cost_ref"),
    [
        (100000, 156, 169.69608, 0.01, 5128920.595980267, 7, 0, 12819),
    ],
)
def test_unitary_cost(n, eta, omega, error, lamb, br, charge, cost_ref):
    r"""Test that unitary_cost returns the correct value."""
    cost = qml.resources.unitary_cost(n, eta, omega, error, lamb, br, charge)

    assert cost == cost_ref

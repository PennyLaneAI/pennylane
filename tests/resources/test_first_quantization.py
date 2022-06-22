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
    ("lz", "cost_ref"),
    [
        # the reference cost is obtained manually by computing the cost for a range of k values, as
        # powers of two and selecting the minimum cost.
        (20, 9),
        (100, 21),
        (300, 35),
    ],
)
def test_cost_qrom(lz, cost_ref):
    r"""Test that _cost_qrom returns the correct value."""
    cost = qml.resources.first_quantization._cost_qrom_fq(lz)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "lamb", "br", "charge", "cost_ref"),
    [
        (10000, 156, 1145.166, 0.001, 5128920.595980267, 7, 0, 12333),
    ],
)
def test_unitary_cost(n, eta, omega, error, lamb, br, charge, cost_ref):
    r"""Test that unitary_cost returns the correct value."""
    cost = qml.resources.first_quantization.unitary_cost_fq(n, eta, omega, error, lamb, br, charge)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "lamb", "br", "charge"),
    [
        (10000.5, 156, 1145.166, 0.001, 281053.7, 7, 0),
        (-10000, 156, 1145.166, 0.001, 281053.7, 7, 0),
        (10000, 156.5, 1145.166, 0.001, 281053.7, 7, 0),
        (10000, -156, 1145.166, 0.001, 281053.7, 7, 0),
        (10000, 156, -1145.166, 0.001, 281053.7, 7, 0),
        (10000, 156, 1145.166, -0.001, 281053.7, 7, 0),
        (10000, 156, 1145.166, 0.001, -281053.7, 7, 0),
        (10000, 156, 1145.166, 0.001, 281053.7, 7.5, 0),
        (10000, 156, 1145.166, 0.001, 281053.7, -7, 0),
        (10000, 156, 1145.166, 0.001, 281053.7, 7, 1.2),
    ],
)
def test_unitary_cost_error(n, eta, omega, error, lamb, br, charge):
    r"""Test that unitary_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resources.first_quantization.unitary_cost_fq(n, eta, omega, error, lamb, br, charge)


@pytest.mark.parametrize(
    ("norm", "error", "cost_ref"),
    [
        # the reference cost is computed manually
        (5128920.595980267, 0.001, 8060117502),
    ],
)
def test_estimation_cost(norm, error, cost_ref):
    r"""Test that estimation_cost returns the correct values."""
    cost = qml.resources.first_quantization.estimation_cost_fq(norm, error)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("norm", "error"),
    [
        (5.28, 0.0),
        (5.28, -1.0),
        (-5.28, 0.01),
        (0.0, 0.01),
    ],
)
def test_estimation_cost_error(norm, error):
    r"""Test that estimation_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        qml.resources.first_quantization.estimation_cost_fq(norm, error)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "lamb", "br", "charge", "cost_ref"),
    [
        (10000, 156, 1145.166, 0.001, 5128920.595980267, 7, 0, 99405429152166),
    ],
)
def test_gate_cost(n, eta, omega, error, lamb, br, charge, cost_ref):
    r"""Test that gate_cost returns the correct value."""
    cost = qml.resources.first_quantization.gate_cost_fq(n, eta, omega, error, lamb, br, charge)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "lamb", "br", "charge"),
    [
        (10000.5, 156, 1145.166, 0.001, 281053.7, 7, 0),
        (-10000, 156, 1145.166, 0.001, 281053.7, 7, 0),
        (10000, 156.5, 1145.166, 0.001, 281053.7, 7, 0),
        (10000, -156, 1145.166, 0.001, 281053.7, 7, 0),
        (10000, 156, -1145.166, 0.001, 281053.7, 7, 0),
        (10000, 156, 1145.166, -0.001, 281053.7, 7, 0),
        (10000, 156, 1145.166, 0.001, -281053.7, 7, 0),
        (10000, 156, 1145.166, 0.001, 281053.7, 7.5, 0),
        (10000, 156, 1145.166, 0.001, 281053.7, -7, 0),
        (10000, 156, 1145.166, 0.001, 281053.7, 7, 1.2),
    ],
)
def test_gate_cost_error(n, eta, omega, error, lamb, br, charge):
    r"""Test that gate_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resources.first_quantization.gate_cost_fq(n, eta, omega, error, lamb, br, charge)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "lamb", "charge", "cost_ref"),
    [
        (10000, 156, 1145.166, 0.001, 281345.0354393263, 0, 3747),
    ],
)
def test_qubit_cost(n, eta, omega, error, lamb, charge, cost_ref):
    r"""Test that qubit_cost returns the correct value."""
    cost = qml.resources.first_quantization.qubit_cost_fq(n, eta, omega, error, lamb, charge)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "lamb", "charge"),
    [
        (10000.5, 156, 1145.166, 0.001, 281053.7, 0),
        (-10000, 156, 1145.166, 0.001, 281053.7, 0),
        (10000, 156.5, 1145.166, 0.001, 281053.7, 0),
        (10000, -156, 1145.166, 0.001, 281053.7, 0),
        (10000, 156, -1145.166, 0.001, 281053.7, 0),
        (10000, 156, 1145.166, -0.001, 281053.7, 0),
        (10000, 156, 1145.166, 0.001, -281053.7, 0),
        (10000, 156, 1145.166, 0.001, 281053.7, 1.2),
    ],
)
def test_qubit_cost_error(n, eta, omega, error, lamb, charge):
    r"""Test that qubit_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resources.first_quantization.qubit_cost_fq(n, eta, omega, error, lamb, charge)

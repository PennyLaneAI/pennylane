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
    ("n", "eta", "omega", "error", "charge", "br"),
    [
        (10000, 156, 1145.166, 0.0016, 0, 7),
    ],
)
def test_fq_params(n, eta, omega, error, charge, br):
    r"""Test that the FirstQuantization class initiates correct parameters."""
    est = qml.resource.FirstQuantization(n, eta, omega)

    assert np.allclose(est.n, n)
    assert np.allclose(est.eta, eta)
    assert np.allclose(est.omega, omega)
    assert np.allclose(est.error, error)
    assert np.allclose(est.charge, charge)
    assert np.allclose(est.br, br)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "lamb", "g_cost", "q_cost"),
    [
        (10000, 156, 1145.166, 281053.7561247674, 3942519392660, 3716),
    ],
)
def test_fq_vals(n, eta, omega, lamb, g_cost, q_cost):
    r"""Test that the FirstQuantization class computes correct attributes."""
    est = qml.resource.FirstQuantization(n, eta, omega)

    assert np.allclose(est.lamb, lamb)
    assert np.allclose(est.gates, g_cost)
    assert np.allclose(est.qubits, q_cost)


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
    cost = qml.resource.FirstQuantization._cost_qrom(lz)

    assert cost == cost_ref


@pytest.mark.parametrize(
    "lz",
    [
        5.7,
        -6,
    ],
)
def test_cost_qrom_error(lz):
    r"""Test that _cost_qrom raises an error with incorrect input."""
    with pytest.raises(ValueError, match="sum of the atomic numbers must be a positive integer"):
        qml.resource.FirstQuantization._cost_qrom(lz)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "br", "charge", "cost_ref"),
    [
        (10000, 156, 1145.166, 0.001, 7, 0, 14387),
    ],
)
def test_unitary_cost(n, eta, omega, error, br, charge, cost_ref):
    r"""Test that unitary_cost returns the correct value."""
    cost = qml.resource.FirstQuantization.unitary_cost(n, eta, omega, error, br, charge)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "br", "charge"),
    [
        (-10000, 156, 1145.166, 0.001, 7, 0),
        (10000, 156.5, 1145.166, 0.001, 7, 0),
        (10000, -156, 1145.166, 0.001, 7, 0),
        (10000, 156, -1145.166, 0.001, 7, 0),
        (10000, 156, 1145.166, -0.001, 7, 0),
        (10000, 156, 1145.166, 0.001, 7.5, 0),
        (10000, 156, 1145.166, 0.001, -7, 0),
        (10000, 156, 1145.166, 0.001, 7, 1.2),
    ],
)
def test_unitary_cost_error(n, eta, omega, error, br, charge):
    r"""Test that unitary_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resource.FirstQuantization.unitary_cost(n, eta, omega, error, br, charge)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "cost_ref"),
    [
        # the reference cost is computed manually
        (10000, 156, 1145.166, 0.001, 441677008),
    ],
)
def test_estimation_cost(n, eta, omega, error, cost_ref):
    r"""Test that estimation_cost returns the correct values."""
    cost = qml.resource.FirstQuantization.estimation_cost(n, eta, omega, error)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error"),
    [
        (10000, 156, 1145.166, 0.0),
        (10000, 156, 1145.166, -1.0),
    ],
)
def test_estimation_cost_error(n, eta, omega, error):
    r"""Test that estimation_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        qml.resource.FirstQuantization.estimation_cost(n, eta, omega, error)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "br", "charge", "cost_ref"),
    [
        (10000, 156, 1145.166, 0.001, 7, 0, 6354407114096),
    ],
)
def test_gate_cost(n, eta, omega, error, br, charge, cost_ref):
    r"""Test that gate_cost returns the correct value."""
    cost = qml.resource.FirstQuantization.gate_cost(n, eta, omega, error, br, charge)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "br", "charge"),
    [
        (-10000, 156, 1145.166, 0.001, 7, 0),
        (10000, 156.5, 1145.166, 0.001, 7, 0),
        (10000, -156, 1145.166, 0.001, 7, 0),
        (10000, 156, -1145.166, 0.001, 7, 0),
        (10000, 156, 1145.166, -0.001, 7, 0),
        (10000, 156, 1145.166, 0.001, 7.5, 0),
        (10000, 156, 1145.166, 0.001, -7, 0),
        (10000, 156, 1145.166, 0.001, 7, 1.2),
    ],
)
def test_gate_cost_error(n, eta, omega, error, br, charge):
    r"""Test that gate_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resource.FirstQuantization.gate_cost(n, eta, omega, error, br, charge)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "br", "charge", "cost_ref"),
    [
        (10000, 156, 1145.166, 0.001, 7, 0, 3747),
    ],
)
def test_qubit_cost(n, eta, omega, error, br, charge, cost_ref):
    r"""Test that qubit_cost returns the correct value."""
    cost = qml.resource.FirstQuantization.qubit_cost(n, eta, omega, error, br, charge)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "charge"),
    [
        (-10000, 156, 1145.166, 0.001, 0),
        (10000, 156, 1145.166, 0.001, 0.35),
        (10000, 156.5, 1145.166, 0.001, 0),
        (10000, -156, 1145.166, 0.001, 0),
        (10000, 156, -1145.166, 0.001, 0),
        (10000, 156, 1145.166, -0.001, 0),
        (10000, 156, 1145.166, 0.001, 1.2),
    ],
)
def test_qubit_cost_error(n, eta, omega, error, charge):
    r"""Test that qubit_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resource.FirstQuantization.qubit_cost(n, eta, omega, error, charge)


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
    prob = qml.resource.FirstQuantization.success_prob(n_basis, br)

    assert prob == prob_ref


@pytest.mark.parametrize(
    ("n_basis", "br"),
    [
        (-10000, 7),
        (10000, 7.2),
        (10000, -7),
    ],
)
def test_success_prob_error(n_basis, br):
    r"""Test that success_prob raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be a positive"):
        qml.resource.FirstQuantization.success_prob(n_basis, br)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "br", "charge", "norm_ref"),
    [
        (10000, 156, 1145.166, 0.001, 7, 0, 281053.7561247674),
    ],
)
def test_norm(n, eta, omega, error, br, charge, norm_ref):
    r"""Test that norm returns the correct value."""
    norm = qml.resource.FirstQuantization.norm(n, eta, omega, error, br, charge)

    assert np.allclose(norm, norm_ref)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "br", "charge"),
    [
        (-10000, -156, 1145.166, 0.001, 7, 0),
        (10000, 156.2, 1145.166, 0.001, 7, 0),
        (10000, -156, 1145.166, 0.001, 7, 0),
        (10000, 156, -1145.166, 0.001, 7, 0),
        (10000, 156, 1145.166, -0.001, 7, 0),
        (10000, 156, 1145.166, 0.001, 7.5, 0),
        (10000, 156, 1145.166, 0.001, -7, 0),
        (10000, 156, 1145.166, 0.001, 7, 1.2),
    ],
)
def test_norm_error(n, eta, omega, error, br, charge):
    r"""Test that norm raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resource.FirstQuantization.norm(n, eta, omega, error, br, charge)

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
# pylint: disable=too-many-arguments,protected-access
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


@pytest.mark.parametrize(
    ("n", "eta", "omega", "vectors", "lamb", "g_cost", "q_cost"),
    [
        (
            100000,
            156,
            None,
            np.array(
                [
                    [9.44862994, 0.0, 0.0],
                    [0.0, 10.39349294, 0.0],
                    [0.94486299, 0.94486299, 11.33835593],
                ]
            ),
            817051.632523202,
            151664625909497,
            3331,
        ),
        (
            10000,
            312,
            None,
            np.array(
                [
                    [18.89725988, 0.0, 0.0],
                    [0.0, 10.39349294, 0.0],
                    [0.94486299, 0.94486299, 11.33835593],
                ]
            ),
            1793399.3143809892,
            64986483274430,
            5193,
        ),
        (
            100000,
            156,
            None,
            np.array(
                [
                    [9.44862994, 0.0, 0.0],
                    [0.0, 10.39349294, 0.0],
                    [0.0, 0.0, 11.33835593],
                ]
            ),
            725147.0916537816,
            134604911168852,
            3331,
        ),
    ],
)
def test_fq_vals_non_qubic(n, eta, omega, vectors, lamb, g_cost, q_cost):
    r"""Test that the FirstQuantization class computes correct attributes."""
    est = qml.resource.FirstQuantization(n, eta, omega, vectors=vectors)

    assert np.allclose(est.lamb, lamb)
    assert np.allclose(est.gates, g_cost)
    assert np.allclose(est.qubits, q_cost)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "br", "charge", "vectors"),
    [
        (10000, 156, None, 0.001, 7, 0, None),
    ],
)
def test_init_error_1(n, eta, omega, error, br, charge, vectors):
    r"""Test that init raises an error when volume and vectors are None."""
    with pytest.raises(ValueError, match="The lattice vectors must be provided"):
        qml.resource.FirstQuantization(n, eta, omega, error, charge, br, vectors)


@pytest.mark.parametrize(
    ("n", "eta", "omega", "error", "br", "charge", "vectors"),
    [
        (
            10000,
            156,
            1113.47,
            0.001,
            7,
            0,
            np.array(
                [
                    [9.0, 0.0, 0.0],
                    [0.0, 9.0, 0.0],
                    [9.0, 9.0, 9.0],
                ]
            ),
        ),
    ],
)
def test_init_error_2(n, eta, omega, error, br, charge, vectors):
    r"""Test that init raises an error when volume and vectors are None."""
    with pytest.raises(ValueError, match="lattice vectors and the unit cell volume should not be"):
        qml.resource.FirstQuantization(n, eta, omega, error, charge, br, vectors)


@pytest.mark.parametrize(
    ("n", "eta", "error", "br", "charge", "vectors"),
    [
        (
            1e10,
            100,
            0.0016,
            7,
            0,
            np.array(
                [
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [1.0, 1.0, 10.0],
                ]
            ),
        ),
    ],
)
def test_norm_error_noncubic(n, eta, error, br, charge, vectors):
    r"""Test that _norm_noncubic raises an error when the computed norm is zero."""
    print(n, eta, error, br, charge, vectors)
    with pytest.raises(ValueError, match="The computed 1-norm is zero"):
        qml.resource.FirstQuantization._norm_noncubic(n, eta, error, br, charge, vectors)


@pytest.mark.parametrize(
    ("n_p, n_m, n_dirty, n_tof, kappa, ms_cost_ref, beta_ref"),
    [(6, 37, 3331, 500, 1, 13372.0, 90.0), (6, 37, 3331, 1, 1, 30686.0, 68.0)],
)
def test_momentum_state_qrom(n_p, n_m, n_dirty, n_tof, kappa, ms_cost_ref, beta_ref):
    r"""Test that the _momentum_state_qrom function returns correct values."""
    ms_cost, beta = qml.resource.FirstQuantization._momentum_state_qrom(
        n_p, n_m, n_dirty, n_tof, kappa
    )

    assert ms_cost == ms_cost_ref
    assert beta == beta_ref

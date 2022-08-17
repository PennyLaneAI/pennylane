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
Unit tests for functions needed for resource estimation with the double factorization method.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np


one_h2 = np.array([[-1.25330961e00, 3.46833673e-13], [3.46944695e-13, -4.75069041e-01]])

two_h2 = np.array(  # in chemist notation
    [
        [
            [[6.74755872e-01, -4.00346423e-13], [-4.00290912e-13, 6.63711349e-01]],
            [[-4.00207645e-13, 1.81210478e-01], [1.81210478e-01, -3.69482223e-13]],
        ],
        [
            [[-4.00263156e-13, 1.81210478e-01], [1.81210478e-01, -3.69482223e-13]],
            [[6.63711349e-01, -3.69482223e-13], [-3.69260178e-13, 6.97651447e-01]],
        ],
    ]
)

two_h2_ph = np.array(  # in physicist notation
    [
        [
            [[6.74755872e-01, 8.45989945e-14], [8.47655279e-14, 1.81210478e-01]],
            [[8.48210391e-14, 1.81210478e-01], [6.63711349e-01, 7.84927678e-14]],
        ],
        [
            [[8.46545056e-14, 6.63711349e-01], [1.81210478e-01, 7.82707232e-14]],
            [[1.81210478e-01, 7.82707232e-14], [7.87148124e-14, 6.97651447e-01]],
        ],
    ]
)


@pytest.mark.parametrize(
    ("one", "two", "error", "tol_factor", "tol_eigval", "br", "alpha", "beta"),
    [
        (one_h2, two_h2, 0.0016, 1.0e-5, 1.0e-5, 7, 10, 20),
    ],
)
def test_df_params(one, two, error, tol_factor, tol_eigval, br, alpha, beta):
    r"""Test that the DoubleFactorization class initiates correct parameters."""
    est = qml.resource.DoubleFactorization(one, two, chemist_notation=True)
    assert np.allclose(est.one_electron, one)
    assert np.allclose(est.two_electron, two)
    assert np.allclose(est.error, error)
    assert np.allclose(est.tol_factor, tol_factor)
    assert np.allclose(est.tol_eigval, tol_eigval)
    assert np.allclose(est.br, br)
    assert np.allclose(est.alpha, alpha)
    assert np.allclose(est.beta, beta)


@pytest.mark.parametrize(
    ("one", "two_phys", "two_chem"),
    [
        (one_h2, two_h2_ph, two_h2),
    ],
)
def test_df_notation_conversion(one, two_phys, two_chem):
    r"""Test that the DoubleFactorization class initiates correct two-electron integrals."""
    est = qml.resource.DoubleFactorization(one, two_phys, chemist_notation=False)
    assert np.allclose(est.two_electron, two_chem)


@pytest.mark.parametrize(
    ("one", "two", "n", "factors", "eigvals", "eigvecs", "rank_r", "rank_m", "rank_max"),
    [
        (  # factors computed with openfermion (rearranged)
            one_h2,
            two_h2,
            4,
            np.array(
                [
                    [[1.06723431e-01, 3.28955607e-15], [3.34805476e-15, -1.04898524e-01]],
                    [[-7.89837537e-14, -4.25688240e-01], [-4.25688240e-01, -1.07150807e-13]],
                    [[-8.14472824e-01, 1.80079693e-13], [1.79803867e-13, -8.28642110e-01]],
                ]
            ),
            [
                np.array([-0.10489852, 0.10672343]),
                np.array([-0.42568824, 0.42568824]),
                np.array([-0.82864211, -0.81447282]),
            ],
            [
                np.array([[1.58209235e-14, -1.00000000e00], [-1.00000000e00, -1.58209235e-14]]),
                np.array([[0.70710678, -0.70710678], [0.70710678, 0.70710678]]),
                np.array([[-1.26896915e-11, -1.00000000e00], [1.00000000e00, -1.26896915e-11]]),
            ],
            3,
            2,
            2,
        ),
    ],
)
def test_df_factorization(one, two, n, factors, eigvals, eigvecs, rank_r, rank_m, rank_max):
    r"""Test that DoubleFactorization class returns correct factorization values."""
    est = qml.resource.DoubleFactorization(one, two, chemist_notation=True)

    assert np.allclose(est.n, n)
    assert np.allclose(est.factors, factors)
    assert np.allclose(np.array(est.eigvals), np.array(eigvals))
    assert np.allclose(np.array(est.eigvecs), np.array(eigvecs))
    assert np.allclose(est.rank_r, rank_r)
    assert np.allclose(est.rank_m, rank_m)
    assert np.allclose(est.rank_max, rank_max)


@pytest.mark.parametrize(
    ("one", "two", "lamb"),
    [
        (one_h2, two_h2, 1.6570514682587973),
    ],
)
def test_df_norm(one, two, lamb):
    r"""Test that DoubleFactorization class returns a correct norm."""
    est = qml.resource.DoubleFactorization(one, two)

    assert np.allclose(est.lamb, lamb)


@pytest.mark.parametrize(
    ("one", "two", "g_cost", "q_cost"),
    [
        (one_h2, two_h2, 876953, 113),
    ],
)
def test_df_costs(one, two, g_cost, q_cost):
    r"""Test that DoubleFactorization class returns correct costs."""
    est = qml.resource.DoubleFactorization(one, two, chemist_notation=True)

    assert np.allclose(est.gates, g_cost)
    assert np.allclose(est.qubits, q_cost)


@pytest.mark.parametrize(
    ("norm", "error", "cost_ref"),
    [  # cost_ref is computed manually
        (72.49779513025341, 0.001, 113880),
    ],
)
def test_estimation_cost(norm, error, cost_ref):
    r"""Test that estimation_cost returns the correct values."""
    cost = qml.resource.DoubleFactorization.estimation_cost(norm, error)

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
        qml.resource.DoubleFactorization.estimation_cost(norm, error)


@pytest.mark.parametrize(
    ("constants", "cost_ref", "k_ref"),
    [  # The reference costs and k values are obtained manually, by computing the cost for a range
        # of k values and selecting the k that gives the minimum cost.
        (
            (26, 1, 0, 15.0, -1),
            27,
            1,
        ),
        (
            (26, 1, 0, 1, 0),
            11,
            4,
        ),
        (
            (151.0, 7.0, 151.0, 280, 0),
            589,
            1,
        ),
        (
            (151.0, 7.0, 151.0, 2, 0),
            52,
            16,
        ),
        (
            (151.0, 7.0, 151.0, 30.0, -1),
            168,
            4,
        ),
    ],
)
def test_qrom_cost(constants, cost_ref, k_ref):
    r"""Test that _qrom_cost returns the correct values."""
    cost, k = qml.resource.DoubleFactorization._qrom_cost(constants)

    assert cost == cost_ref
    assert k == k_ref


@pytest.mark.parametrize(
    ("n", "rank_r", "rank_m", "rank_max", "br", "alpha", "beta", "cost_ref"),
    [
        (14, 26, 5.5, 7, 7, 10, 20, 2007),
    ],
)
def test_unitary_cost(n, rank_r, rank_m, rank_max, br, alpha, beta, cost_ref):
    r"""Test that unitary_cost returns the correct value."""
    cost = qml.resource.DoubleFactorization.unitary_cost(
        n, rank_r, rank_m, rank_max, br, alpha, beta
    )

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "rank_r", "rank_m", "rank_max", "br", "alpha", "beta"),
    [
        (14.5, 26, 5.5, 7, 7, 10, 20),
        (-14, 26, 5.5, 7, 7, 10, 20),
        (11, 26, 5.5, 7, 7, 10, 20),
        (14, -26, 5.5, 7, 7, 10, 20),
        (14, 26.1, 5.5, 7, 7, 10, 20),
        (14, 26, -5.5, 7, 7, 10, 20),
        (14, 26, 5.5, 7.5, 7, 10, 20),
        (14, 26, 5.5, -7, 7, 10, 20),
        (14, 26, 5.5, 7, -7, 10, 20),
        (14, 26, 5.5, 7, 7.5, 10, 20),
        (14, 26, 5.5, 7, 7, -10, 20),
        (14, 26, 5.5, 7, 7, 10.2, 20),
        (14, 26, 5.5, 7, 7, 10, -20),
        (14, 26, 5.5, 7, 7, 10, 20.9),
    ],
)
def test_unitary_cost_error(n, rank_r, rank_m, rank_max, br, alpha, beta):
    r"""Test that unitary_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be a positive"):
        qml.resource.DoubleFactorization.unitary_cost(n, rank_r, rank_m, rank_max, br, alpha, beta)


@pytest.mark.parametrize(
    ("n", "norm", "error", "rank_r", "rank_m", "rank_max", "br", "alpha", "beta", "cost_ref"),
    [
        (14, 52.98761457453095, 0.001, 26, 5.5, 7, 7, 10, 20, 167048631),
    ],
)
def test_gate_cost(n, norm, error, rank_r, rank_m, rank_max, br, alpha, beta, cost_ref):
    r"""Test that gate_cost returns the correct value."""
    cost = qml.resource.DoubleFactorization.gate_cost(
        n, norm, error, rank_r, rank_m, rank_max, br, alpha, beta
    )

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "norm", "error", "rank_r", "rank_m", "rank_max", "br", "alpha", "beta"),
    [
        (14.5, 5.5, 0.01, 26, 5.5, 7, 7, 10, 20),
        (-14, 5.5, 0.01, 26, 5.5, 7, 7, 10, 20),
        (11, 5.5, 0.01, 26, 5.5, 7, 7, 10, 20),
        (14, 5.28, 0.0, 26, 5.5, 7, 7, 10, 20),
        (14, 5.28, -1.0, 26, 5.5, 7, 7, 10, 20),
        (14, -5.28, 0.01, 26, 5.5, 7, 7, 10, 20),
        (14, 0.0, 0.01, 26, 5.5, 7, 7, 10, 20),
        (14, 5.5, 0.01, -26, 5.5, 7, 7, 10, 20),
        (14, 5.5, 0.01, 26.1, 5.5, 7, 7, 10, 20),
        (14, 5.5, 0.01, 26, -5.5, 7, 7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7.5, 7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, -7, 7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, -7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 7.5, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 7, -10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 7, 10.2, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 7, 10, -20),
        (14, 5.5, 0.01, 26, 5.5, 7, 7, 10, 20.9),
    ],
)
def test_gate_cost_error(n, norm, error, rank_r, rank_m, rank_max, br, alpha, beta):
    r"""Test that gate_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resource.DoubleFactorization.gate_cost(
            n, norm, error, rank_r, rank_m, rank_max, br, alpha, beta
        )


@pytest.mark.parametrize(
    ("n", "norm", "error", "rank_r", "rank_m", "rank_max", "br", "alpha", "beta", "cost_ref"),
    [
        (14, 52.98761457453095, 0.001, 26, 5.5, 7, 7, 10, 20, 292),
    ],
)
def test_qubit_cost(n, norm, error, rank_r, rank_m, rank_max, br, alpha, beta, cost_ref):
    r"""Test that qubit_cost returns the correct value."""
    cost = qml.resource.DoubleFactorization.qubit_cost(
        n, norm, error, rank_r, rank_m, rank_max, br, alpha, beta
    )

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "norm", "error", "rank_r", "rank_m", "rank_max", "br", "alpha", "beta"),
    [
        (14.5, 5.5, 0.01, 26, 5.5, 7, 7, 10, 20),
        (-14, 5.5, 0.01, 26, 5.5, 7, 7, 10, 20),
        (11, 5.5, 0.01, 26, 5.5, 7, 7, 10, 20),
        (14, 5.28, 0.0, 26, 5.5, 7, 7, 10, 20),
        (14, 5.28, -1.0, 26, 5.5, 7, 7, 10, 20),
        (14, -5.28, 0.01, 26, 5.5, 7, 7, 10, 20),
        (14, 0.0, 0.01, 26, 5.5, 7, 7, 10, 20),
        (14, 5.5, 0.01, -26, 5.5, 7, 7, 10, 20),
        (14, 5.5, 0.01, 26.1, 5.5, 7, 7, 10, 20),
        (14, 5.5, 0.01, 26, -5.5, 7, 7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7.5, 7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, -7, 7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, -7, 7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 7.5, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 7, -10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 7, 10.2, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 7, 10, -20),
        (14, 5.5, 0.01, 26, 5.5, 7, 7, 10, 20.9),
    ],
)
def test_qubit_cost_error(n, norm, error, rank_r, rank_m, rank_max, br, alpha, beta):
    r"""Test that qubit_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resource.DoubleFactorization.qubit_cost(
            n, norm, error, rank_r, rank_m, rank_max, br, alpha, beta
        )


@pytest.mark.parametrize(
    ("one", "two", "eigvals", "lamb_ref"),
    [
        (
            np.array([[-1.25330961e00, 4.01900735e-14], [4.01900735e-14, -4.75069041e-01]]),
            # two-electron integral is arranged in chemist notation
            np.array(
                [
                    [
                        [[6.74755872e-01, -4.60742555e-14], [-4.60742555e-14, 6.63711349e-01]],
                        [[-4.61020111e-14, 1.81210478e-01], [1.81210478e-01, -4.26325641e-14]],
                    ],
                    [
                        [[-4.60464999e-14, 1.81210478e-01], [1.81210478e-01, -4.25215418e-14]],
                        [[6.63711349e-01, -4.28546088e-14], [-4.24105195e-14, 6.97651447e-01]],
                    ],
                ]
            ),
            np.tensor(
                [[-0.10489852, 0.10672343], [-0.42568824, 0.42568824], [-0.82864211, -0.81447282]]
            ),
            1.6570518796336895,  # lambda value obtained from openfermion
        )
    ],
)
def test_df_norm(one, two, eigvals, lamb_ref):
    r"""Test that the norm function returns the correct 1-norm."""
    lamb = qml.resource.DoubleFactorization.norm(one, two, eigvals)

    assert np.allclose(lamb, lamb_ref)

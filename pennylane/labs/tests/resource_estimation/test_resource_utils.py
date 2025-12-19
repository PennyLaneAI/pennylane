# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Test the utility functions for resource estimation.
"""
from collections.abc import Callable

import numpy as np
import pytest

from pennylane.labs.resource_estimation.resource_utils import (
    _chebyshev_gauss_lobatto,
    _legendre_gauss_lobatto,
    approx_poly_degree,
)


def test_approx_poly_degree():
    """Test the approx_poly_degree function"""

    def target_func(x: np.ndarray) -> np.ndarray:
        """Target function to be approximated"""
        return x**2

    x_vec = np.array([1, 2, 3, 4, 5])
    degree, poly, loss = approx_poly_degree(target_func, x_vec, poly_degs=4)
    assert degree == 2
    assert np.allclose(poly.coef, np.array([0, 0, 1]))
    assert np.allclose(loss, 0)


@pytest.mark.parametrize("basis", ["chebyshev", "legendre", "hermite"])
def test_approx_poly_degree_basis(basis):
    """Test the approx_poly_degree function with different bases"""

    def target_func(x: np.ndarray) -> np.ndarray:
        """Target function to be approximated"""
        return np.random.RandomState(863).rand(len(x))

    x_vec, e_tol = np.sort(np.random.RandomState(123).rand(10)), 1e-2
    degree, poly, loss = approx_poly_degree(
        target_func, x_vec, basis=basis, poly_degs=10, error_tol=e_tol
    )

    assert isinstance(degree, int)
    assert isinstance(poly, Callable)
    assert isinstance(loss, float)
    assert loss <= e_tol


def test_approx_poly_project_func():
    """Test the approx_poly_degree function with different project functions"""

    def morse(x: np.ndarray, x0=1.510, a=2.7968, D=6.610) -> np.ndarray:
        dx = x - x0
        return D * (np.exp(-2.0 * a * dx) - 2.0 * np.exp(-a * dx))

    x_vec = np.linspace(0.92, 1.93, 1000)

    def cheb_nodes(a: float, b: float, n: int) -> np.ndarray:
        k = np.arange(n)
        x = np.cos(np.pi * k / (n - 1))
        return 0.5 * (a + b) + 0.5 * (b - a) * x

    deg1, poly1, loss1 = approx_poly_degree(
        morse, x_vec, basis="chebyshev", domain_func=cheb_nodes, poly_degs=(3, 10)
    )
    deg2, poly2, loss2 = approx_poly_degree(
        morse, x_vec, basis="chebyshev", domain_func="gauss-lobatto", poly_degs=(3, 10)
    )

    assert deg1 == deg2
    assert np.allclose(poly1.coef, poly2.coef)
    assert np.allclose(loss1, loss2) and loss1 < 1e-4


@pytest.mark.parametrize("loss_func", [None, "mse", "mae", "rmse", "linf"])
def test_approx_poly_custom_func(loss_func):
    """Test the approx_poly_degree function with different fit-loss functions"""

    x_vec = np.linspace(0, 1, 1000)

    def target_func(x: np.ndarray) -> np.ndarray:
        return np.sin(x) * np.exp(-x)

    def fit_func(x: np.ndarray, y: np.ndarray, deg: int, **__) -> tuple[np.ndarray, float]:
        poly, stats = np.polynomial.laguerre.Laguerre.fit(x, y, deg, full=True)
        return poly, stats[0]

    deg, poly, loss = approx_poly_degree(
        target_func, x_vec, error_tol=1e-6, fit_func=fit_func, loss_func=loss_func, poly_degs=10
    )

    assert isinstance(deg, int) and deg < 10
    assert isinstance(poly, np.polynomial.laguerre.Laguerre)
    assert isinstance(loss, float) and loss < 1e-6


@pytest.mark.parametrize(
    "num_points, result",
    [
        (2, [-1, 1]),
        (3, [-1, 0, 1]),
        (4, [-1, -(1 / 5) * np.sqrt(5), (1 / 5) * np.sqrt(5), 1]),
        (5, [-1, -(1 / 7) * np.sqrt(21), 0, (1 / 7) * np.sqrt(21), 1]),
        (
            6,
            [
                -1,
                -np.sqrt((1 / 21) * (7 + 2 * np.sqrt(7))),
                -np.sqrt((1 / 21) * (7 - 2 * np.sqrt(7))),
                np.sqrt((1 / 21) * (7 - 2 * np.sqrt(7))),
                np.sqrt((1 / 21) * (7 + 2 * np.sqrt(7))),
                1,
            ],
        ),
    ],
)
def test_legendre_gauss_lobatto(num_points, result):
    """Test the Legendre-Gauss-Lobatto function

    Result populated from the table in https://mathworld.wolfram.com/LobattoQuadrature.html for the first 6 nodes.
    """

    x_nodes = _legendre_gauss_lobatto(-1, 1, num_points)
    assert len(x_nodes) == num_points
    assert np.allclose(x_nodes, result)


@pytest.mark.parametrize("num_points", range(2, 7))
def test_chebyshev_gauss_lobatto(num_points):
    """Test the Chebyshev-Gauss-Lobatto function"""

    x_nodes = _chebyshev_gauss_lobatto(-1, 1, num_points)
    assert len(x_nodes) == num_points
    assert np.allclose(x_nodes, np.polynomial.chebyshev.chebpts2(num_points)[::-1])

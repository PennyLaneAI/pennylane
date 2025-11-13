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
"""Utility functions for resource estimation."""
from collections.abc import Callable
from typing import Any

import numpy as np


# pylint: disable=unnecessary-lambda-assignment, too-many-arguments
def approx_poly_degree(
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    error_tol: float = 1e-6,
    degrees: tuple[int, int] | None = None,
    basis: str | None = None,
    approx_poly_func: Callable | None = None,
    **fit_kwargs: dict[str, Any],
):
    r"""Compute the minimum degree of a polynomial that fits the data with a given error tolerance.

    Args:
        x_vec (tensor_like): x-values for the sample points
        y_vec (tensor_like): y-values for the sample points
        error_tol (float): tolerance for the target fitting error. Defaults to ``1e-6``.
            Unless ``loss_func`` is provided, this is the least squares fit error.
        degrees (Tuple[int, int]): tuple of minimum and maximum degrees to consider.
            Defaults to ``None``, which means all degrees from ``2`` to ``len(x_vec) - 1``
            will be considered.
        basis (str): basis to use for the polynomial. Available options are ``"chebyshev"``,
            ``"legendre"``, and ``"hermite"``. Defaults to ``None``, which assumes fitting
            data to a polynomial in the monomial basis.
        loss_func (str | callable): loss function to use, where available options are
            `"mse"` (mean squared error), `"mae"` (mean absolute error), `"rmse"`
            (root mean squared error), or a custom loss function. Defaults to `"mse"`.
        approx_poly_func (callable): function to approximate the polynomial with signature
            ``(x_vec, y_vec, degree, **kwargs) -> tuple[np.array, float]``
            Defaults to ``None``, which will use the NumPy polynomial fit function corresponding
            to the ``basis`` keyword argument. Providing a custom function will override the
            ``basis`` and ``loss`` keyword arguments.
        projection_func (str | callable | None): function to project the dense interval
            based on `x_vec` to a sparse one with ``[x_vec[0], x_vec[-1]]`` as the domain.
            Defaults to ``None``, which means no projection is performed. When ``"uniform"``
            is used, the points are evenly spaced between ``x_vec[0]`` and ``x_vec[-1]``.
            Whereas, when ``"gauss-lobatto"`` is used, the "Chebyshev-Gauss-Lobatto" nodes are
            used for ``basis="chebyshev"``, and the "Legendre-Gauss-Lobatto" nodes are used for
            ``basis="legendre"``. When a callable is provided, it should have the signature
            ``(x_min: float, x_max: float, num_points: int) -> tensor_like``.
        **fit_kwargs: additional keyword arguments to pass to the `fitting` functions.
            See keyword arguments below for available arguments for the default ``NumPy`` fitting
            functions. Custom functions may support different keyword arguments.

    Keyword Arguments:
        rcond (float): the relative condition number of the fit.
        w (tensor_like): weights for the sample points. This is equivalent to using a custom
            ``interpolate_func`` to imitate unselected points with zero weights.

    Returns:
        tuple[np.ndarray, float]: the coefficients of the polynomial and the loss of the fit.
    """
    if degrees is None:
        degrees = (2, len(x_vec) - 1)

    min_degree, max_degree = degrees
    if min_degree > max_degree:
        raise ValueError("min_degree must be less than or equal to max_degree")

    if min_degree < 0:
        raise ValueError("min_degree must be non-negative")

    loss_func, approx_fit_class = lambda x: x, None
    if approx_poly_func is None:
        loss_func = lambda x: float("inf") if len(x[0]) == 0 else float(x[0][0])
        fit_kwargs["full"] = True

        match basis:
            case "chebyshev":
                approx_poly_func = np.polynomial.chebyshev.chebfit
                approx_fit_class = np.polynomial.chebyshev.Chebyshev
            case "legendre":
                approx_poly_func = np.polynomial.legendre.legfit
                approx_fit_class = np.polynomial.legendre.Legendre
            case "hermite":
                approx_poly_func = np.polynomial.hermite.hermfit
                approx_fit_class = np.polynomial.hermite.Hermite
            case _:
                approx_poly_func = np.polynomial.polynomial.polyfit
                approx_fit_class = np.polynomial.polynomial.Polynomial

    best_loss, best_poly = float("inf"), None
    for degree in range(min_degree, max_degree + 1):
        poly, stats = approx_poly_func(x_vec, y_vec, degree, **fit_kwargs)
        if (loss := loss_func(stats)) and loss < best_loss:
            best_loss, best_poly = loss, poly
            if loss <= error_tol:
                break

    return best_poly, best_loss


def _chebyshev_gauss_lobatto(x_min: float, x_max: float, num_points: int) -> np.ndarray:
    """Project the dense interval based on `x_vec` to a sparse one with `[x_vec[0], x_vec[-1]]` as the domain.

    Args:
        x_min (float): minimum x-value
        x_max (float): maximum x-value
        num_points (int): number of points to project to

    Returns:
        np.ndarray: projected x-values
    """
    x = np.cos(np.pi * np.arange(num_points) / (num_points - 1))
    return 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * x


def _legendre_gauss_lobatto(x_min: float, x_max: float, num_points: int) -> np.ndarray:
    """Project the dense interval based on `x_vec` to a sparse one with `[x_vec[0], x_vec[-1]]` as the domain.

    Args:
        x_min (float): minimum x-value
        x_max (float): maximum x-value
        num_points (int): number of points to project to

    Returns:
        np.ndarray: projected x-values
    """
    # Get the (num_points - 1)-th Legendre polynomial and the roots of its derivative
    internal_nodes = np.polynomial.legendre.Legendre.basis(num_points - 1).deriv().roots()
    x = np.sort(np.concatenate(([-1.0], internal_nodes, [1.0])))
    return 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * x

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
    max_degree: int | None = None,
    min_degree: int | None = None,
    basis: str | None = None,
    approx_poly_func: Callable | None = None,
    **fit_kwargs: dict[str, Any],
):
    r"""Compute the minimum degree of a polynomial that fits the data with a given error tolerance.

    Args:
        x_vec (tensor_like): x-values for the sample points
        y_vec (tensor_like): y-values for the sample points
        error_tol (float): tolerance for the least squares fit error. Defaults to ``1e-6``.
        max_degree (int): maximum degree of the polynomial. Defaults to ``0``.
        min_degree (int): minimum degree of the polynomial. Defaults to ``len(x_vec) - 1``.
        basis (str): basis to use for the polynomial. Available options are ``"chebyshev"``,
            ``"legendre"``, and ``"hermite"``. Defaults to ``None``, which assumes fitting
            data to a polynomial in the monomial basis.
        approx_poly_func (callable): function to approximate the polynomial with signature
            ``(x_vec, y_vec, degree, **kwargs) -> tuple[np.array, float]``
            Defaults to ``None``, which will use the NumPy polynomial fit function corresponding
            to the ``basis`` keyword argument. Providing a custom function will override the
            ``basis`` keyword argument.
        **fit_kwargs: additional keyword arguments to pass to the `fitting` functions.
            See keyword arguments below for available arguments for the default ``NumPy`` fitting
            functions. Custom functions may support different keyword arguments.

    Keyword Arguments:
        rcond (float): the relative condition number of the fit.
        w (tensor_like): weights for the sample points.

    Returns:
        tuple[np.ndarray, float]: the coefficients of the polynomial and the loss of the fit.
    """
    min_degree = 0 if min_degree is None else min_degree
    max_degree = len(x_vec) - 1 if max_degree is None else max_degree

    if min_degree > max_degree:
        raise ValueError("min_degree must be less than or equal to max_degree")

    if min_degree < 0:
        raise ValueError("min_degree must be non-negative")

    loss_func = lambda x: x
    if approx_poly_func is None:
        loss_func = lambda x: float("inf") if len(x[0]) == 0 else float(x[0][0])
        fit_kwargs["full"] = True

        match basis:
            case "chebyshev":
                approx_poly_func = np.polynomial.chebyshev.chebfit
            case "legendre":
                approx_poly_func = np.polynomial.legendre.legfit
            case "hermite":
                approx_poly_func = np.polynomial.hermite.hermfit
            case _:
                approx_poly_func = np.polynomial.polynomial.polyfit

    best_loss, best_poly = float("inf"), None
    for degree in range(min_degree, max_degree + 1):
        poly, stats = approx_poly_func(x_vec, y_vec, degree, **fit_kwargs)
        if (loss := loss_func(stats)) and loss < best_loss:
            best_loss, best_poly = loss, poly
            if loss <= error_tol:
                break

    return best_poly, best_loss

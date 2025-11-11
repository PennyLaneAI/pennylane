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
import numpy as np


def approx_poly_degree(x_vec, y_vec, error_tol=1e-6, loss="mse", max_degree=None, min_degree=None, basis=None, **fit_kwargs, **loss_kwargs):
    r"""Approximates a polynomial function of a given degree at value x

    Args:
        x_vec (tensor_like): values to approximate with a polynomial
        y_vec (tensor_like): values to approximate with a polynomial
        error_tol (float): tolerance for the error
        loss (str | callable): loss function to use, where available options are `"mse"` (mean squared error), `"mae"`
            (mean absolute error), `"rmse"` (root mean squared error), or a custom loss function. Defaults to `"mse"`.
        max_degree (int): maximum degree of the polynomial
        min_degree (int): minimum degree of the polynomial
        basis (str): basis to use for the polynomial
        **fit_kwargs: additional keyword arguments to pass to the Numpy's polynomial fit function.
            See these `docs <https://numpy.org/doc/stable/reference/generated/numpy.polynomial.Polynomial.fit.html>`_
            for more details.
        **loss_kwargs: additional keyword arguments to pass to the loss function
    """
    min_degree = 0 if min_degree is None else min_degree
    max_degree = len(x_vec) - 1 if max_degree is None else max_degree

    if min_degree > max_degree:
        raise ValueError("min_degree must be less than or equal to max_degree")
    
    if min_degree < 0:
        raise ValueError("min_degree must be non-negative")

    # pylint: disable=unnecessary-lambda-assignment
    if loss == "mse":
        loss_func = lambda y_pred, y_true: np.mean((y_pred - y_true) ** 2)
    elif loss == "mae":
        loss_func = lambda y_pred, y_true: np.mean(np.abs(y_pred - y_true))
    elif loss == "rmse":
        loss_func = lambda y_pred, y_true: np.sqrt(np.mean((y_pred - y_true) ** 2))
    else:
        loss_func = loss

    best_loss, best_poly = float('inf'), None
    for degree in range(min_degree, max_degree + 1):
        poly = approx_poly(x_vec, y_vec, degree, basis=basis, **fit_kwargs)
        loss = loss_func(poly(x_vec), y_vec, **loss_kwargs)
        if loss < best_loss:
            best_loss, best_poly = loss, poly
        if loss < error_tol:
            break

    return best_poly, best_loss

def approx_poly(x_vec, y_vec, degree, basis=None, **fit_kwargs):
    r"""Approximates a polynomial function of a given degree at value x

    Args:
        x_vec (tensor_like): values to approximate with a polynomial
        y_vec (tensor_like): values to approximate with a polynomial
        degree (int): degree of the polynomial
        basis (str): basis to use for the polynomial
        **fit_kwargs: additional keyword arguments to pass to the Numpy's polynomial fit function.
            See these `docs <https://numpy.org/doc/stable/reference/generated/numpy.polynomial.Polynomial.fit.html>`_
            for more details.
    """
    if basis == "chebyshev":
        return np.polynomial.Chebyshev.fit(x_vec, y_vec, degree, **fit_kwargs)
    return np.polynomial.Polynomial.fit(x_vec, y_vec, degree, **fit_kwargs)

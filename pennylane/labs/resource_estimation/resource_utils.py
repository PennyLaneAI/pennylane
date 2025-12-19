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


# pylint: disable=unnecessary-lambda-assignment, too-many-arguments, not-callable
def approx_poly_degree(
    target_func: Callable,
    x_vec: np.ndarray,
    error_tol: float = 1e-6,
    poly_degs: tuple[int, int] | int | None = None,
    basis: str = "monomial",
    loss_func: Callable | None = None,
    fit_func: Callable | None = None,
    domain_func: Callable | None = None,
    **fit_kwargs: dict[str, Any],
):
    r"""Compute the minimum degree of the polynomial that fits an input function within a given
    error tolerance.

    The polynomial is fit to the data using the least squares method in the monomial, Chebyshev,
    Legendre, or Hermite bases. The degree of the polynomial is incrementally increased within the
    range specified by the ``poly_degs`` keyword argument until the error tolerance (``error_tol``)
    is met, where it is defined as the least squares fit error unless a custom loss function
    (``loss_func``) is provided. Users can also specify a custom fitting function (``fit_func``)
    to use for the polynomial fitting and a custom domain function (``domain_func``) to transform
    the specified domain ``x_vec`` for the fitting. Please look at the examples below for more details.

    Args:
        target_func (Callable): function to be approximated with a polynomial. It must have the
            signature: ``f(x_vec: np.ndarray) -> np.ndarray``.
        x_vec (np.ndarray): the domain values for sampling the target function ``target_func``.
            It must be a sorted array with a minimum length of ``2``.
        error_tol (float | optional): tolerance for the target fitting error. Defaults to ``1e-6``.
            It is the least squares fit error unless a custom loss function (``loss_func``) is provided.
        poly_degs (Tuple[int, int] | int | None | optional): tuple containing the minimum and maximum
            degrees of the fitting polynomial. Defaults to ``None``, which means all degrees from
            ``1`` to ``len(x_vec) - 1`` will be considered. If an integer is provided, it
            will be used as the maximum degree permissible for the approximated polynomial.
        basis (str): the basis for the fitting polynomial. Available options are ``"monomial"``,
            ``"chebyshev"``, ``"legendre"``, and ``"hermite"``. Defaults to ``"monomial"``,
            which means fitting data to a polynomial in the monomial basis.
        loss_func (str | Callable | None | optional):  the loss function to be minimized during the fitting.
            Available options are ``"mse"`` for minimizing the mean squared error, ``"mae"`` for minimizing
            the mean absolute error, ``"rmse"`` for minimizing the root mean squared error, ``"linf"`` for
            minimizing the maximum absolute error, or a custom loss function with the signature:
            ``f(pred: np.ndarray, target: np.ndarray) -> float``. Defaults to ``None``, which means
            the least squares fit error is used.
        fit_func (Callable | None | optional): a function that fits a target data (``x_vec``, ``y_vec``) with
            a polynomial of degree ``deg`` and also computes the loss of the fit. It must have the signature:
            ``f(x_vec: np.ndarray, y_vec: np.ndarray, deg: int, **fit_kwargs) -> tuple[Callable, float]`` and
            should return a callable for the fit polynomial along with the least squares error of the fit.
            Defaults to ``None``, which means the NumPy polynomial fitting function corresponding to the
            ``basis`` keyword argument will be used. Providing a custom function will override
            the ``basis`` keyword argument.
        domain_func (str | Callable | None | optional): a function or a preset to generate the fitting
            points within the domain interval ``[x_vec[0], x_vec[-1]]`` for fitting the ``target_func``.
            Defaults to ``None``, which means all the points in ``x_vec`` are used directly without any
            transformation. When ``"uniform"`` is given, the points are sparsely spaced between ``x_vec[0]``
            and ``x_vec[-1]`` based on the degree of the polynomial being fit.
            Whereas, when ``"gauss-lobatto"`` is used, the "`Chebyshev/Legendre-Gauss-Lobatto
            <https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Lobatto_rules>`_\ " nodes
            are used for ``basis="chebyshev"/"legendre"``, respectively.
            When a user-defined function is provided, it should have the signature:
            ``f(x_min: float, x_max: float, num_points: int) -> np.ndarray``.
        **fit_kwargs: additional keyword arguments to pass to the `fitting` functions.
            See keyword arguments below for available arguments for the default ``NumPy``
            fitting functions that will be used with ``fit_func=None``. Custom functions
            may support different keyword arguments.

    Keyword Arguments:
        rcond (float): the relative condition number of the fit.
        w (np.ndarray): weights for the sample points. This is equivalent to using a custom
            ``domain_func`` to imitate unselected points with zero weights.

    Returns:
        tuple[int, Callable, float]: the degree of the polynomial, the fit polynomial function, and the loss of the fit.

    **Example:**

    This example fits the function :math:`f(x) = x^2` using the monomial basis in the default setting.

    .. code-block::

        >>> import numpy as np
        >>> from pennylane.labs.resource_estimation import approx_poly_degree
        >>> x_vec = np.linspace(-2, 2, 100)
        >>> target_func = lambda x: x**2
        >>> degree, poly, loss = approx_poly_degree(
        ...     target_func, x_vec, poly_degs=(2, 4), error_tol=1e-6
        ... )
        >>> print(degree)
        2
        >>> print(poly) # Monomial polynomial
        3.55271368e-16 + (3.4742413e-16)·x + 1.0·x²
        >>> print(loss) # Sum of squared errors
        2.40281579329135e-29

    This example fits the function :math:`f(x) = \sin(3x) + 0.3x^2 - 0.1x` using the Chebyshev basis with the Gauss-Lobatto nodes.

    .. code-block::

        >>> x_vec = np.linspace(-1, 1, 100)
        >>> target_func = lambda x: np.sin(3*x) + 0.3*x**2 - 0.1*x
        >>> degree, poly, loss = approx_poly_degree(
        ...     target_func, x_vec, poly_degs=(2, 10), error_tol=1e-3,
        ...     basis="chebyshev", loss_func="linf", domain_func="gauss-lobatto"
        ... )
        >>> print(degree)
        7
        >>> print(poly) # Chebyshev polynomial
        0.15 + 0.57811797·T₁(x) + 0.15·T₂(x) - 0.61812903·T₃(x) +
        (2.43236301e-18)·T₄(x) + 0.08622566·T₅(x) - (9.61838039e-18)·T₆(x) -
        0.00509459·T₇(x)
        >>> print(loss) # Maximum absolute error
        0.0003266957232994083

    This example fits the function :math:`f(x) = \sin(x) \exp(-x)` using a custom fit function for the Laguerre basis.

    .. code-block::

        >>> x_vec = np.linspace(0, 1, 100)
        >>> target_func = lambda x: np.sin(x) * np.exp(-x)
        >>> def fit_func(x, y, deg, **fit_kwargs):
        ...     poly, stats = np.polynomial.laguerre.Laguerre.fit(x, y, deg, full=True)
        ...     return poly, stats[0]
        >>> degree, poly, loss = approx_poly_degree(
        ...     target_func, x_vec, poly_degs=(3, 10), error_tol=1e-6,
        ...     fit_func=fit_func, loss_func="mse"
        ... )
        >>> print(degree)
        3
        >>> print(poly) # Laguerre polynomial
        0.71956319 - 2.0816571·L₁(x) + 2.99459653·L₂(x) - 1.63191581·L₃(x)
        >>> print(loss) # Mean squared error
        5.286491415090545e-08
    """
    x_vec = np.sort(x_vec)
    y_vec = target_func(x_vec)

    if poly_degs is None:
        poly_degs = (1, len(x_vec) - 1)
    elif isinstance(poly_degs, int):
        poly_degs = (1, poly_degs)

    min_degree, max_degree = poly_degs
    if min_degree > max_degree:
        raise ValueError("min_degree must be less than or equal to max_degree")

    if min_degree < 0:
        raise ValueError("min_degree must be non-negative")

    fit_loss = _process_loss_func(loss_func)

    fit_func, fit_args = _process_poly_fit(fit_func, basis)
    fit_kwargs |= fit_args

    proj_func = _process_domain_func(domain_func, basis)

    best_loss, best_poly = float("inf"), None
    for degree in range(min_degree, max_degree + 1):

        x_proj, y_proj = x_vec, y_vec
        x_domain = (x_vec[0], x_vec[-1])

        if proj_func is not None:
            x_proj = np.sort(proj_func(*x_domain, degree + 1))
            y_proj = target_func(x_proj)

        poly, stats = fit_func(
            x_proj, y_proj, degree, domain=x_domain, window=x_domain, **fit_kwargs
        )

        loss = fit_loss(stats) if loss_func is None else fit_loss(poly(x_vec), y_vec)
        if loss is None and loss_func is None:
            loss = float(np.sum((poly(x_vec) - y_vec) ** 2))

        if loss < best_loss:
            best_loss, best_poly = loss, poly
            if loss <= error_tol:
                break

    return degree, best_poly, best_loss


def _process_loss_func(loss_func: str | Callable | None) -> Callable:
    """Process the loss function."""
    if loss_func is None:

        def _loss_func(x, _: Any = None):
            if isinstance(x, (tuple, list)):
                return None if len(x[0]) == 0 else float(x[0][0])
            return float(x)

        return _loss_func

    match loss_func:
        case "mse":
            return lambda x, y: np.mean((x - y) ** 2)
        case "mae":
            return lambda x, y: np.mean(np.abs(x - y))
        case "rmse":
            return lambda x, y: np.sqrt(np.mean((x - y) ** 2))
        case "linf":
            return lambda x, y: np.max(np.abs(x - y))
        case _:
            return loss_func


def _process_poly_fit(fit_func: Callable | None, basis: str) -> tuple[Callable, Callable | None]:
    """Process the polynomial fitting function based on the basis.

    Args:
        fit_func (Callable | None): function that approximately fits the polynomial and has the signature:
            ``f(x_vec: np.ndarray, y_vec: np.ndarray, deg: int, **fit_kwargs) -> tuple[Callable, float]``.
        basis (str): basis to use for the polynomial. Available options are ``"monomial"``,
            ``"chebyshev"``, ``"legendre"``, ``"hermite"``. Any other basis will default to
            using ``np.polynomial.Polynomial.fit`` as the fitting function unless otherwise provided.

    Returns:
        tuple[Callable, Callable | None]: the fit polynomial function and the fit polynomial function.
    """
    fit_args = {}
    if fit_func is None:
        fit_args["full"] = True
        match basis:
            case "monomial":
                fit_func = np.polynomial.polynomial.Polynomial.fit
            case "chebyshev":
                fit_func = np.polynomial.chebyshev.Chebyshev.fit
            case "legendre":
                fit_func = np.polynomial.legendre.Legendre.fit
            case "hermite":
                fit_func = np.polynomial.hermite.Hermite.fit
            case _:
                fit_func = np.polynomial.polynomial.Polynomial.fit

    return fit_func, fit_args


def _process_domain_func(domain_func: str | Callable | None, basis: str) -> Callable | None:
    """Process the domain projection function.

    Args:
        domain_func (str | Callable | None): function to project the dense interval
            based on ``x_vec`` to a sparse one with ``[x_vec[0], x_vec[-1]]`` as the domain.
        basis (str): basis to use for the polynomial. Available options are ``"monomial"``,
            ``"chebyshev"``, ``"legendre"``, ``"hermite"``. Any other basis will default
            to using the ``np.linspace`` as the projection function unless otherwise provided.

    Returns:
        Callable | None: the domain projection function.
    """
    match domain_func:
        case "uniform":
            return np.linspace
        case "gauss-lobatto":
            match basis:
                case "monomial":
                    return np.linspace
                case "chebyshev":
                    return _chebyshev_gauss_lobatto
                case "legendre":
                    return _legendre_gauss_lobatto
                case _:
                    return np.linspace
        case _:
            return domain_func


def _chebyshev_gauss_lobatto(x_min: float, x_max: float, num_points: int) -> np.ndarray:
    """Generate `Chebyshev-Gauss-Lobatto nodes <https://en.wikipedia.org/wiki/Chebyshev_nodes>`_
    mapped to the interval :math:`[x_{\text{min}}, x_{\text{max}}]` from the standard Chebyshev
    nodes of the second kind on :math:`[-1, 1]` using an affine transformation.

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
    """Generate `Legendre-Gauss-Lobatto nodes <https://mathworld.wolfram.com/LobattoQuadrature.html>`_
    mapped to the interval :math:`[x_{\text{min}}, x_{\text{max}}]` from the roots of the derivative
    of (``num_points`` - 1)-th Legendre polynomial using an affine transformation.

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

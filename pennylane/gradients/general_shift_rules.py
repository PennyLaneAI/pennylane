# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains a function for generating generalized parameter shift rules."""
import functools
import itertools
import warnings

import numpy as np
import pennylane as qml


def _process_gradient_recipe(gradient_recipe, tol=1e-10):
    """Utility function to process gradient recipes.

    This utility function accepts coefficients and shift values, and performs the following
    processing:

    - Removes all small (within absolute tolerance ``tol``) coefficients and shifts

    - Removes terms with the coefficients are 0

    - Terms with the same shift value are combined into a single term.

    - Finally, the terms are sorted according to the absolute value of ``shift``,
      ensuring that, if there is a zero-shift term, this is returned first.
    """
    # remove all small coefficients and shifts
    gradient_recipe[np.abs(gradient_recipe) < tol] = 0

    # remove columns where the coefficients are 0
    gradient_recipe = gradient_recipe[:, ~(gradient_recipe[0] == 0)]

    # sort columns according to abs(shift)
    gradient_recipe = gradient_recipe[:, np.argsort(np.abs(gradient_recipe)[-1])]
    round_tol = int(-np.log10(tol))
    unique_shifts = np.unique(np.round(gradient_recipe[-1], round_tol))

    if gradient_recipe.shape[-1] != len(unique_shifts):
        # sum columns that have the same shift value
        gradient_recipe = np.stack(
            [
                np.stack(
                    [
                        np.sum(
                            gradient_recipe[:, np.nonzero(np.round(gradient_recipe[-1], round_tol) == b)[0]], axis=1
                        )[0]
                        for b in unique_shifts
                    ]
                ),
                unique_shifts,
            ]
        )

        # sort columns according to abs(shift)
        gradient_recipe = gradient_recipe[:, np.argsort(np.abs(gradient_recipe)[-1])]

    return gradient_recipe


@functools.lru_cache(maxsize=None)
def eigvals_to_frequencies(eigvals):
    r"""Convert an eigenvalue spectra to frequency values, defined
    as the the set of positive, unique, differences of the spectra.

    Args:
        eigvals (tuple[int, float]): eigenvalue spectra

    Returns:
        tuple[int, float]: frequencies

    **Example**

    >>> eigvals = (-0.5, 0, 0, 0.5)
    >>> eigvals_to_frequencies(eigvals)
    (0.5, 1.0)
    """
    unique_eigvals = sorted(set(eigvals))
    return tuple({abs(i - j) for i, j in itertools.combinations(unique_eigvals, 2)})


@functools.lru_cache(maxsize=None)
def frequencies_to_period(frequencies):
    r"""Returns the period of a Fourier series as defined
    by a set of frequencies. The period is simply :math:`2\pi/f_min`,
    where :math:`f_min` is the smallest positive frequency value.

    Args:
        spectra (tuple[int, float]): eigenvalue spectra

    Returns:
        tuple[int, float]: frequencies

    **Example**

    >>> frequencies = (0.5, 1.0)
    >>> frequencies_to_period(frequencies)
    12.566370614359172
    """
    f_min = min(f for f in frequencies if f > 0)
    return 2 * np.pi / f_min


@functools.lru_cache(maxsize=None)
def _get_shift_rule(frequencies, shifts=None):
    n_freqs = len(frequencies)
    frequencies = qml.math.sort(qml.math.stack(frequencies))
    freq_min = frequencies[0]

    if len(set(frequencies)) != n_freqs or freq_min <= 0:
        raise ValueError(
            "Expected frequencies to be a list of unique positive values, instead got {}.".format(
                frequencies
            )
        )

    mu = np.arange(1, n_freqs + 1)

    if shifts is None:  # assume equidistant shifts
        shifts = (2 * mu - 1) * np.pi / (2 * n_freqs * freq_min)
        equ_shifts = True
    else:
        shifts = qml.math.sort(qml.math.stack(shifts))
        if len(shifts) != n_freqs:
            raise ValueError(
                "Expected number of shifts to equal the number of frequencies ({}), instead got {}.".format(
                    n_freqs, shifts
                )
            )
        if len(set(shifts)) != n_freqs:
            raise ValueError("Shift values must be unique, instead got {}".format(shifts))

        equ_shifts = all(np.isclose(shifts, (2 * mu - 1) * np.pi / (2 * n_freqs * freq_min)))

    if len(set(np.round(np.diff(frequencies), 10))) <= 1 and equ_shifts:  # equidistant case
        coeffs = (
            freq_min
            * (-1) ** (mu - 1)
            / (4 * n_freqs * np.sin(np.pi * (2 * mu - 1) / (4 * n_freqs)) ** 2)
        )

    else:  # non-equidistant case
        sin_matr = -4 * np.sin(np.outer(shifts, frequencies))
        det_sin_matr = np.linalg.det(sin_matr)
        if abs(det_sin_matr) < 1e-6:
            warnings.warn(
                "Solving linear problem with near zero determinant ({}) may give unstable results for the parameter shift rules.".format(
                    det_sin_matr
                )
            )

        coeffs = -2 * np.linalg.solve(sin_matr.T, frequencies)

    coeffs = np.concatenate((coeffs, -coeffs))
    shifts = np.concatenate((shifts, -shifts))
    return np.stack([coeffs, shifts])


@functools.lru_cache()
def generate_shift_rule(frequencies, shifts=None, order=1):
    r"""Computes the parameter shift rule for a unitary based on its generator's eigenvalue frequency
    spectrum.

    To compute gradients of circuit parameters in variational quantum algorithms, expressions for
    cost function first derivatives with respect to the variational parameters can be cast into
    linear combinations of expectation values at shifted parameter values. These "gradient recipes"
    can be obtained from the unitary generator's eigenvalue frequency spectrum. Details can be
    found in https://arxiv.org/abs/2107.12390.

    Args:
        frequencies (tuple[int or float]): The tuple of eigenvalue frequencies. Eigenvalue
            frequencies are defined as the unique positive differences obtained from a set of
            eigenvalues.
        shifts (tuple[int or float]): the tuple of shift values. If unspecified, equidistant
            shifts are assumed. If supplied, the length of this tuple should match the number of
            given frequencies.
        order (int): the order of differentiation to compute the shift rule for

    Returns:
        tuple: a tuple of coefficients and shifts describing the gradient recipe
        for the parameter-shift method. For parameter :math:`\phi`, the
        coefficients :math:`c_i` and the shifts :math:`s_i` combine to give a gradient
        recipe of the following form:

        .. math:: \frac{\partial}{\partial\phi}f = \sum_{i} c_i f(\phi + s_i).

        where :math:`f(\phi) = \langle 0|U(\phi)^\dagger \hat{O} U(\phi)|0\rangle`
        for some observable :math:`\hat{O}` and some unitary :math:`U(\phi)`.

    Raises:
        ValueError: if ``frequencies`` is not a list of unique positive values, or if ``shifts``
            (if specified) is not a list of unique values the same length as ``frequencies``.

    **Examples**

    An example of obtaining the frequencies from generator eigenvalues, and obtaining the
    parameter shift rule:

    >>> eigvals = (-0.5, 0, 0, 0.5)
    >>> frequencies = eigvals_to_frequencies(eigvals)
    >>> generate_shift_rule(frequencies)
    [[ 0.85355339, -0.85355339, -0.14644661,  0.14644661],
     [ 0.78539816, -0.78539816,  2.35619449, -2.35619449]]

    An example with explicitly specified shift values:

    >>> frequencies = (1, 2, 4)
    >>> shifts = (np.pi / 3, 2 * np.pi / 3, np.pi / 4)
    >>> generate_shift_rule(frequencies, shifts)
    [[ 3.        , -3.        , -2.09077028,  2.09077028, 0.2186308, -0.2186308 ],
     [ 0.78539816, -0.78539816,  1.04719755, -1.04719755, 2.0943951, -2.0943951 ]]

    Higher order shift rules (corresponding to the :math:`n`-th derivative of the parameter)
    can be requested via the ``order`` argument. For example, to extract the
    second order shift rule for a gate with generator :math:`X/2`:

    >>> eigvals = (0.5, -0.5)
    >>> frequencies = eigvals_to_frequencies(eigvals)
    >>> generate_shift_rule(frequencies, order=2)
    [[-0.5       ,  0.5       ],
     [ 0.        , -3.14159265]]

    This corresponds to :math:`\frac{\partial^2 f}{\partial phi^2} = \frac{1}{2} \left[f(\phi) - f(\phi-\pi)]`.
    """
    frequencies = tuple(f for f in frequencies if f > 0)
    recipe = _get_shift_rule(frequencies, shifts=shifts)

    if order > 1:
        all_shifts = []
        T = frequencies_to_period(frequencies)

        for partial_recipe in itertools.product(recipe.T, repeat=order):
            c, s = np.stack(partial_recipe).T
            new_shift = np.mod(sum(s) + 0.5 * T, T) - 0.5 * T
            all_shifts.append(np.stack([np.prod(c), new_shift]))

        recipe = qml.math.stack(all_shifts).T

    return _process_gradient_recipe(recipe, tol=1e-10)


def generate_multi_shift_rule(frequencies, shifts=None, orders=None):
    r"""Computes the parameter shift rule with respect to two parametrized unitaries,
    given their generator's eigenvalue frequency spectrum. This corresponds to a
    shift rule that computes off-diagonal elements of the Hessian.

    Args:
        frequencies (list[tuple[int or float]]): List of eigenvalue frequencies corresponding
            to the each parametrized unitary.
        shifts (list[tuple[int or float]]): List of shift values corresponding to each parametrized
            unitary. If unspecified, equidistant shifts are assumed. If supplied, the length
            of each tuple in the list must be the same as the length of each tuple in
            ``frequencies``.
        orders (list[int]): the order of differentiation for each parametrized unitary.
            If unspecified, the first order derivative shift rule is computed for each parametrized
            unitary.

    Returns:
        tuple: a tuple of coefficients, shifts for the first parameter, and shifts for the
        second parameter, describing the gradient recipe
        for the parameter-shift method.

        For parameters :math:`\phi_a` and :math:`\phi_b`, the
        coefficients :math:`c_i` and the shifts :math:`s^{(a)}_i`, :math:`s^{(b)}_i`,
        combine to give a gradient recipe of the following form:

        .. math::

            \frac{\partial^2}{\partial\phi_a \partial\phi_b}f
            = \sum_{i} c_i \left[ f(\phi_a + s^{(a)}_i) + f(\phi_b + s^{(b)}_i) \right].

        where :math:`f(\phi) = \langle 0|U(\phi)^\dagger \hat{O} U(\phi)|0\rangle`
        for some observable :math:`\hat{O}` and some unitary :math:`U(\phi)`.

    **Example**

    >>> generate_multi_shift_rule([(1,), (1,)])
    [[ 0.25      , -0.25      , -0.25      ,  0.25      ],
     [ 1.57079633,  1.57079633, -1.57079633, -1.57079633],
     [ 1.57079633, -1.57079633,  1.57079633, -1.57079633]])

    This corresponds to the gradient recipe

    .. math::

        \frac{\partial^2 f}{\partial x\partial y}
        = \frac{1}{4} \left[f(x+\np/2, y+\np/2) - f(x+\np/2, y-\np/2) - f(x-\np/2, y+\np/2) + f(x-\np/2, y-\np/2)].
    """
    recipes = []
    shifts = shifts or [None] * len(frequencies)
    orders = orders or [1] * len(frequencies)

    for f, s, o in zip(frequencies, shifts, orders):
        rule = generate_shift_rule(f, shifts=s, order=o)
        recipes.append(_process_gradient_recipe(rule).T)

    all_shifts = []

    for partial_recipes in itertools.product(*recipes, repeat=1):
        c, s = np.stack(partial_recipes).T
        combined = np.concatenate([[np.prod(c)], s])
        all_shifts.append(np.stack(combined))

    return np.stack(all_shifts).T

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
import warnings
import numpy as np
import pennylane as qml


@functools.lru_cache(maxsize=None)
def get_shift_rule(frequencies, shifts=None):
    r"""Computes the parameter shift rule for a unitary based on its generator's eigenvalue frequency
    spectrum.

    To compute gradients of circuit parameters in variational quantum algorithms, expressions for
    cost function first derivatives with respect to the variational parameters can be cast into
    linear combinations of expectation values at shifted parameter values. These "gradient recipes"
    can be obtained from the unitary generator's eigenvalue frequency spectrum. Details can be
    found in `arXiv:2107.12390 <https://arxiv.org/abs/2107.12390>`__.

    Args:
        frequencies (tuple[int or float]): the tuple of eigenvalue frequencies. Eigenvalue
            frequencies are defined as the unique positive differences obtained from a set of
            eigenvalues.
        shifts (tuple[int or float]): the tuple of shift values. If unspecified, the equidistant
            shifts `(2 * mu - 1) * pi / (2 * n_freqs * freq_min)` are assumed, where `n_freqs` is
            the number of frequencies, `freq_min` is the smallest frequency, and
            `mu = 1, ..., n_freqs`. If supplied, the length of this tuple should match the number
            of given frequencies.

    Returns:
         tuple: a tuple of one nested list describing the gradient recipe
         for the parameter-shift method.
         This is a tuple with one nested list per operation parameter. For
         parameter :math:`\phi_k`, the nested list contains elements of the form
         :math:`[c_i, a_i, s_i]` where :math:`i` is the index of the
         term, resulting in a gradient recipe of

            .. math:: \frac{\partial}{\partial\phi_k}f = \sum_{i} c_i f(a_i \phi_k + s_i).


    Raises:
        ValueError: if ``frequencies`` is not a list of unique positive values, or if ``shifts``
            (if specified) is not a list of unique values the same length as ``frequencies``.

    **Examples**

    An example of obtaining the frequencies from a set of unique eigenvalues and obtaining the
    parameter shift rule:

    >>> unique_eigenvals = sorted([1, -1, 0])
    >>> from itertools import combinations
    >>> frequencies = {abs(i - j) for i, j in combinations(unique_eigenvals, 2)}
    >>> get_shift_rule(tuple(frequencies))
    ([[0.8535533905932737, 1, 0.7853981633974483], [-0.14644660940672624, 1, 2.356194490192345],
    [-0.8535533905932737, 1, -0.7853981633974483], [0.14644660940672624, 1, -2.356194490192345]],)

    An example with explicitly specified shift values:

    >>> frequencies = (1, 2, 4)
    >>> shifts = (np.pi / 3, 2 * np.pi / 3, np.pi / 4)
    >>> get_shift_rule(frequencies, shifts)
    ([[-2.0907702751760278, 1, 1.0471975511965976], [0.2186308015824754, 1, 2.0943951023931953],
    [3.0000000000000004, 1, 0.7853981633974483], [2.0907702751760278, 1, -1.0471975511965976],
    [-0.2186308015824754, 1, -2.0943951023931953], [-3.0000000000000004, 1, -0.7853981633974483]],)
    """

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
    return ([[coeffs[mu], 1, shifts[mu]] for mu in range(0, 2 * n_freqs)],)

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
import numpy as np


def get_shift_rule(frequencies, diff_shifts=None):
    r"""Computes the parameter shift rule for a unitary based on its generator's eigenvalue frequency
     spectrum.

    To compute gradients of circuit parameters in variational quantum algorithms, expressions for
    cost function first derivatives with respect to the variational parameters can be cast into
    linear combinations of expectation values at shifted parameter values. These "gradient recipes"
    can be obtained from the unitary generator's eigenvalue frequency spectrum. Details can be
    found in https://arxiv.org/abs/2107.12390.

    Args:
        frequencies (list): the list of eigenvalue frequencies. Eigenvalue frequencies are defined
            as the unique positive differences obtained from a set of eigenvalues.
        diff_shifts (list): the list of shift values. If unspecified, equidistant shifts are
            assumed. Note that if equidistant frequencies are given, any specified shifts will be
            ignored, and equidistant shifts will be used instead.

    Returns:
        tuple: a tuple of one nested list describing the gradient recipe for the parameter-shift
        method.

    Raises:
        ValueError: if `frequencies` is not a list of unique positive values, or if `diff_shifts`
            (if specified) is not a list of unique values the same length as `frequencies`.

    **Example**

    >>> frequencies = [1,2]
    >>> get_shift_rule(frequencies)
    ([[0.8535533905932737, 1, 0.7853981633974483], [-0.14644660940672624, 1, 2.356194490192345],
    [-0.8535533905932737, 1, -0.7853981633974483], [0.14644660940672624, 1, -2.356194490192345]],)

    """

    n_freqs = len(frequencies)

    if not (len(set(frequencies)) == n_freqs and all(freq > 0 for freq in frequencies)):
        raise ValueError(
            "Expected frequencies to be a list of unique positive values, instead got {}.".format(
                frequencies
            )
        )

    if diff_shifts is None:  # assume equidistant shifts
        diff_shifts = [(2 * mu - 1) * np.pi / (2 * n_freqs) for mu in range(1, n_freqs + 1)]
    else:
        diff_shifts = list(diff_shifts)
        if len(diff_shifts) != n_freqs:
            raise ValueError(
                "Expected number of shifts to equal the number of frequencies ({}), instead got {}.".format(
                    n_freqs, diff_shifts
                )
            )
        if len(set(diff_shifts)) != n_freqs:
            raise ValueError("Shift values must be unique, instead got {}".format(diff_shifts))

    frequencies = sorted(frequencies)
    freq_min = frequencies[0]

    if np.allclose(
        np.array(frequencies) / freq_min, range(1, len(frequencies) + 1)
    ):  # equidistant case
        diff_shifts = [
            (2 * mu - 1) * np.pi / (2 * n_freqs * freq_min) for mu in range(1, n_freqs + 1)
        ]
        diff_coeffs = [
            freq_min
            * (-1) ** (mu - 1)
            / (4 * n_freqs * np.sin(np.pi * (2 * mu - 1) / (4 * n_freqs)) ** 2)
            for mu in range(1, n_freqs + 1)
        ]

    else:  # non-equidistant case
        sin_matr = -4 * np.sin(np.outer(diff_shifts, frequencies))
        sin_matr_inv = np.linalg.inv(sin_matr)
        diff_coeffs = [-2 * np.vdot(frequencies, sin_matr_inv[:, mu]) for mu in range(n_freqs)]

    coeffs = diff_coeffs + [-coeff for coeff in diff_coeffs]
    shifts = diff_shifts + [-shift for shift in diff_shifts]
    return ([[coeffs[mu], 1, shifts[mu]] for mu in range(0, 2 * n_freqs)],)

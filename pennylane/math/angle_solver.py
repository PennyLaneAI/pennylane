# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains the implementation of the angle solver for QSP and QSVT.
"""

import pennylane as qml
import numpy as np
from numpy.polynomial import Polynomial, chebyshev
from functools import partial


def get_complementary_poly(P):
    """
    Computes the complementary polynomial Q given a polynomial P.

    The polynomial Q satisfies the following equation:

    .. math:

        |P(e^{i\theta})|^2 + |Q(e^{i\theta})|^2 = 1, \quad \forall \theta \in \left[0, 2\pi\right]

    The method is based on computing an auxiliary polynomial R, finding its roots,
    and reconstructing Q by using information extracted of those roots.
    For more details see reference `arXiv:2308.01501 <https://arxiv.org/abs/2308.01501>`_.

    Args:
        P (array-like): Coefficients of the complex polynomial P.

    Returns:
        array-like: Coefficients of the complementary polynomial Q.
    """
    poly_degree = len(P) - 1

    # Build the polynomial R(z) = z^degree * (1 - conj(P(z)) * P(z))
    R = Polynomial.basis(poly_degree) - Polynomial(P) * Polynomial(np.conj(P[::-1]))
    r_roots = R.roots()

    # Categorize the roots based on their magnitude: outside the unit circle or inside the closed unit circle
    inside_circle = [root for root in r_roots if np.abs(root) <= 1]
    outside_circle = [root for root in r_roots if np.abs(root) > 1]

    # Compute the scaling factor for Q, which depends on the leading coefficient of R
    scale_factor = np.sqrt(np.abs(R.coef[-1] * np.prod(outside_circle)))

    # Complementary polynomial Q is built using th roots inside the closed unit circle
    Q_poly = scale_factor * Polynomial.fromroots(inside_circle)

    return Q_poly.coef


def get_QSP_angles(F):
    """
    Computes the Quantum Signal Processing (QSP) angles given a polynomial F.

    Args:
        F (array-like): Coefficients of the input polynomial F.

    Returns:
        theta (array-like): QSP angles corresponding to the input polynomial F.
    """

    parity = (len(F) - 1) % 2

    # Construct the auxiliary polynomial P and its complementary Q based on appendix A in [arXiv:2406.04246]
    P = np.concatenate([np.zeros(len(F) // 2), chebyshev.poly2cheb(F)[parity::2]])
    Q = np.array(get_complementary_poly(P))

    S = np.array([P, Q])
    n = S.shape[1]
    theta = np.zeros(n)

    for d in reversed(range(n)):
        # Extract real parts of the current entries in S for the current column
        a, b = S[:, d]

        theta[d] = np.arctan(b.real / a.real)

        # Although the paper works with an RX gate as a reference, RY is used here to simplify
        # the calculations by working only with real numbers.
        matrix = qml.matrix(qml.RY(-2*theta[d], wires = 0))
        S =  matrix @ S
        S = np.array([S[0][1: d + 1], S[1][0:d]])

    if P[-1] < 0:
        theta[0] += np.pi

    return theta

def transform_angles(angles, routine1, routine2):
    """
    Transforms a set of angles from one routine's format to another.

    This function adjusts the angles according to the specified transformation
    between two routines, either from "Quantum Signal Processing" (QSP) to
    "Quantum Singular Value Transformation" (QSVT), or vice versa.

    Args:
        angles (array-like): A list or array of angles to be transformed.
        routine1 (str): The current routine of the angles. Must be either "QSP" or "QSVT".
        routine2 (str): The target routine to which the angles should be transformed.
                        Must be either "QSP" or "QSVT".

    Returns:
        array-like: The transformed angles as an array.

    """
    if routine1 == "QSP" and routine2 == "QSVT":
        num_angles = len(angles)
        update_vals = np.zeros(num_angles)

        update_vals[0] = 3 * np.pi / 4 - (3 + len(angles) % 4) * np.pi / 2
        update_vals[1:-1] = np.pi / 2
        update_vals[-1] = -np.pi / 4

        return angles + update_vals

    if routine1 == "QSVT" and routine2 == "QSP":
        num_angles = len(angles)
        update_vals = np.zeros(num_angles)

        update_vals[0] = 3 * np.pi / 4 - (3 + len(angles) % 4) * np.pi / 2
        update_vals[1:-1] = np.pi / 2
        update_vals[-1] = -np.pi / 4

        return angles - update_vals

def poly_to_angles(P, routine):

    if routine == "QSP":
        return get_QSP_angles(P)

    if routine == "QSVT":
        return transform_angles(get_QSP_angles(P), "QSP", "QSVT")



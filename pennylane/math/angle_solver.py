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

import numpy as np
from numpy.polynomial import Polynomial, chebyshev

import pennylane as qml


def complementary_poly(P):
    r"""
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

    # Build the polynomial R(z) = z^degree * (1 - conj(P(z)) * P(z)), deduced from (eq.33) and (eq.34)
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


def QSP_angles(F):
    r"""
    Computes the Quantum Signal Processing (QSP) angles given a polynomial F.
    Currently works up to polynomials of degree ~1000.

    Args:
        F (array-like): Coefficients of the input polynomial F.

    Returns:
        theta (array-like): QSP angles corresponding to the input polynomial F.
    """

    parity = (len(F) - 1) % 2

    # Construct the auxiliary polynomial P and its complementary Q based on appendix A in [arXiv:2406.04246]
    primary_poly = np.concatenate([np.zeros(len(F) // 2), chebyshev.poly2cheb(F)[parity::2]]) * (
        1 - 1e-12
    )
    secondary_poly = complementary_poly(primary_poly)

    polynomial_matrix = np.array([primary_poly, secondary_poly])
    num_terms = polynomial_matrix.shape[1]
    rotation_angles = np.zeros(num_terms)

    # This subroutine is an adaptation of Algorithm 1 in [arXiv:2308.01501]
    # in order to work in the context of QSP.
    with qml.QueuingManager.stop_recording():
        for idx in range(num_terms - 1, -1, -1):

            poly_a, poly_b = polynomial_matrix[:, idx]
            rotation_angles[idx] = np.arctan2(poly_b.real, poly_a.real)

            rotation_op = qml.matrix(qml.RY(-2 * rotation_angles[idx], wires=0))

            updated_poly_matrix = rotation_op @ polynomial_matrix
            polynomial_matrix = np.array(
                [updated_poly_matrix[0][1 : idx + 1], updated_poly_matrix[1][0:idx]]
            )

    return rotation_angles


def transform_angles(angles, routine1, routine2):
    r"""
    Transforms a set of angles from one routine's format to another.

    This function adjusts the angles according to the specified transformation
    between two routines, either from "Quantum Signal Processing" (QSP) to
    "Quantum Singular Value Transformation" (QSVT), or vice versa.
    By default, transform QSP into QSVT angles.

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

    # if routine1 == "QSVT" and routine2 == "QSP":
    num_angles = len(angles)
    update_vals = np.zeros(num_angles)

    update_vals[0] = 3 * np.pi / 4 - (3 + len(angles) % 4) * np.pi / 2
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = -np.pi / 4

    return angles - update_vals


def poly_to_angles(P, routine):
    r"""
    Converts a given polynomial's coefficients into angles for specific quantum signal processing (QSP),
    quantum singular value transformation (QSVT) or generalizeed quantum signal processing (GQSP) routines.
    By default, returns QSP angles.

    Args:
        P (array-like): Coefficients of the polynomial, ordered from lowest to higher degree.
                        The polynomial must have defined parity and real coefficients.

        routine (str):  Specifies the type of angle transformation required. Must be either: "QSP", "QSVT" or "GQSP".

    Returns:
        (array-like): Angles corresponding to the specified transformation routine.
    """

    if routine == "GQSP":
        return GQSP_angles(P)

    parity = (len(P) - 1) % 2
    assert np.allclose(P[1 - parity :: 2], 0), "Polynomial must have defined parity"
    assert np.allclose(
        np.array(P, dtype=np.complex128).imag, 0
    ), "Array must not have an imaginary part"

    if routine == "QSVT":
        return transform_angles(QSP_angles(P), "QSP", "QSVT")

    return QSP_angles(P)


def GQSP_angles(P):
    r"""
    Computes the Generalized Quantum Signal Processing (GQSP) angles given a polynomial P [arXiv:2308.01501].

    Args:
        P (array-like): Coefficients of the input polynomial P.

    Returns:
        angles (array-like): GQSP angles corresponding to the input polynomial P. The shape is (3, P-degree)
    """

    Q = complementary_poly(P)

    def gqsp_u3_gate(theta, phi, lambd):
        # Matrix definition of U3 gate chosen in the GQSP paper

        exp_phi = np.exp(1j * phi)
        exp_lambda = np.exp(1j * lambd)
        exp_lambda_phi = np.exp(1j * (lambd + phi))

        matrix = np.array(
            [
                [exp_lambda_phi * np.cos(theta), exp_phi * np.sin(theta)],
                [exp_lambda * np.sin(theta), -np.cos(theta)],
            ],
            dtype=complex,
        )

        return matrix

    # This subroutine is an adaptation of Algorithm 1 in [arXiv:2308.01501]
    input_data = np.array([P, Q])
    num_elements = input_data.shape[1]

    angles_theta, angles_phi, angles_lambda = np.zeros([3, num_elements])

    for idx in range(num_elements - 1, -1, -1):

        component_a, component_b = input_data[:, idx]
        angles_theta[idx] = np.arctan2(np.abs(component_b), np.abs(component_a))
        angles_phi[idx] = (
            0
            if np.isclose(np.abs(component_b), 0, atol=1e-10)
            else np.angle(component_a * np.conj(component_b))
        )

        if idx == 0:
            angles_lambda[0] = np.angle(component_b)
        else:
            updated_matrix = (
                gqsp_u3_gate(angles_theta[idx], angles_phi[idx], 0).conj().T @ input_data
            )
            input_data = np.array([updated_matrix[0][1 : idx + 1], updated_matrix[1][0:idx]])

    return angles_theta, angles_phi, angles_lambda

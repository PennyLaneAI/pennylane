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
"""This module contains functions to fit potential energy surfaces
per normal modes on a grid."""

import numpy as np
from pennylane import qchem
from numpy.linalg import eigvalsh

from pennylane.labs.vibronic.pes_vibronic_utils import harmonic_analysis, generate_grid
from pennylane.labs.vibronic.pes_vibronic import pes_mode


def create_matrix(params_matrix, coeffs_matrix):
    """
    Creates a matrix from a set of coefficients and variables.

    Args:
        params_matrix (np.array): A 3D array of size NxNxM containing the parameters.
        coeffs_matrix_set (np.array): A 3D array of size NxNxM containing one set of coefficients.

    Returns:
        np.array: The NxN matrix A.
    """
    N, _, _ = params_matrix.shape
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            delta = 1 if i == j else 0
            A[i, j] = np.dot(
                coeffs_matrix[i, j][: N + 1], params_matrix[i, j][: N + 1]
            ) * delta + np.dot(coeffs_matrix[i, j][N + 1 :], params_matrix[i, j][N + 1 :])

    return A


def cost_function(a_flat, N, M, Q, coeffs_matrix_4d, E_target_vectors):
    """
    Calculates the cost from a flattened parameter vector using a double sum
    over multiple mode displacements.

    Args:
        a_flat (np.array): The flattened 1D vector of parameters.
        N (int): The matrix size.
        M (int): The number of parameters per element.
        Q (int): The number of coefficient/target vector sets.
        coeffs_matrix_4d (np.array): The 4D coefficient array of size QxNxNxM.
        E_target_vectors (np.array): A 2D array of size QxN containing target
                                     eigenvalue vectors.

    Returns:
        float: The total cost.
    """
    a = a_flat.reshape((N, N, M))

    total_cost = 0.0

    for q in range(Q):
        current_coeffs = coeffs_matrix_4d[q]
        current_E_target = E_target_vectors[q]
        A = create_matrix(a, current_coeffs)
        eigenvalues = eigvalsh(A)
        total_cost += np.sum((eigenvalues - np.sort(current_E_target)) ** 2)

    return total_cost


def coeff_element(lam, freq, q_vector, two_mode=False):
    """
    Generates one matrix elements of a potential energy matrix.

    Args:
        lam (float): The zero-displacement energy value.
        freq (float): Vibrational frequencies.
        q_vector (np.array): A 1D array of displacements.
        two_mode (bool): If True, includes two-mode interaction terms.

    Returns:
        np.array: A 1D array of coefficients.
    """
    coeffs = [lam] + [val for d in q_vector for val in (d * freq / 2, d)]
    if two_mode:
        coeffs.extend([d1 * d2 for d1 in q_vector for d2 in q_vector])
    return np.array(coeffs)


def potential_matrix(freqs, displacements, e_zero):
    """
    Generates the full potential energy matrix.

    Args:
        freqs (np.array): A 1D array of frequencies.
        displacements (np.array): A 1D array of displacements.
        e_zero (np.array): A 1D array of zero-displacement energies.

    Returns:
        list: A list of 3D coefficient matrices.
    """
    n_modes = len(freqs)
    e_states = len(e_zero)

    coeffs_matrix_total = []

    for n in range(n_modes):
        for d in displacements:
            q_vector = np.zeros(n_modes)
            q_vector[n] = d

            coeff_vec = coeff_element(e_zero[n], freqs[n], q_vector, two_mode=False)
            coeffs_matrix = np.full((e_states, e_states, len(coeff_vec)), coeff_vec)
            coeffs_matrix_total.append(coeffs_matrix)

    return coeffs_matrix_total

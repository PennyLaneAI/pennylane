# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the ``MottonenStatePreparation`` template.
"""
import math
import numpy as np
from scipy import sparse

import pennylane as qml

from pennylane.templates.decorator import template
from pennylane.templates.utils import check_wires, check_shape, get_shape
from pennylane.variable import Variable


# pylint: disable=len-as-condition,arguments-out-of-order
def gray_code(rank):
    """Generates the Gray code of given rank.

    Args:
        rank (int): rank of the Gray code (i.e. number of bits)
    """

    def gray_code_recurse(g, rank):
        k = len(g)
        if rank <= 0:
            return

        for i in range(k - 1, -1, -1):
            char = "1" + g[i]
            g.append(char)
        for i in range(k - 1, -1, -1):
            g[i] = "0" + g[i]

        gray_code_recurse(g, rank - 1)

    g = ["0", "1"]
    gray_code_recurse(g, rank - 1)

    return g


def _matrix_M_entry(row, col):
    """Returns one entry for the matrix that maps alpha to theta.

    Args:
        row (int): one-based row number
        col (int): one-based column number

    Returns:
        (float): transformation matrix entry at given row and column
    """
    # (col >> 1) ^ col is the Gray code of col
    b_and_g = row & ((col >> 1) ^ col)
    sum_of_ones = 0
    while b_and_g > 0:
        if b_and_g & 0b1:
            sum_of_ones += 1

        b_and_g = b_and_g >> 1

    return (-1) ** sum_of_ones


def _compute_theta(alpha):
    """Calculates the rotation angles from the alpha vector.

    Args:
        alpha (array[float]): alpha parameters

    Returns:
        (array[float]): rotation angles theta
    """
    k = np.log2(alpha.shape[0])
    factor = 2 ** (-k)

    theta = sparse.dok_matrix(alpha.shape, dtype=np.float64)  # type: sparse.dok_matrix

    for row in range(alpha.shape[0]):
        # Use transpose of M:
        entry = sum([_matrix_M_entry(col, row) * a for (col, _), a in alpha.items()])
        entry *= factor
        if abs(entry) > 1e-6:
            theta[row, 0] = entry

    return theta


def _uniform_rotation_dagger(gate, alpha, control_wires, target_wire):
    """Applies a given inverse rotation to the target qubit
    that is uniformly controlled by the control qubits.

    Args:
        gate (~.Operation): gate to be applied, needs to have exactly
            one parameter
        alpha (array[float]): alpha parameters
        control_wires (array[int]): wires that act as control
        target_wire (int): wire that acts as target
    """

    theta = _compute_theta(alpha)  # type: sparse.dok_matrix

    gray_code_rank = len(control_wires)

    if gray_code_rank == 0:
        gate(theta[0, 0], wires=[target_wire])
        return

    code = gray_code(gray_code_rank)
    num_selections = len(code)

    control_indices = [
        int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2)))
        for i in range(num_selections)
    ]

    for i, control_index in enumerate(control_indices):
        gate(theta[i, 0], wires=[target_wire])
        qml.CNOT(wires=[control_wires[control_index], target_wire])


def _uniform_rotation_z_dagger(alpha, control_wires, target_wire):
    """Applies the inverse of a Z rotation to the target qubit
    that is uniformly controlled by the control qubits.

    Args:
        alpha (array[float]): alpha parameters
        control_wires (array[int]): wires that act as control
        target_wire (int): wire that acts as target
    """

    _uniform_rotation_dagger(qml.RZ, alpha, control_wires, target_wire)


def _uniform_rotation_y_dagger(alpha, control_wires, target_wire):
    """Applies the inverse of a Y rotation to the target qubit
    that is uniformly controlled by the control qubits.

    Args:
        alpha (array[float]): alpha parameters
        control_wires (array[int]): wires that act as control
        target_wire (int): wire that acts as target
    """

    _uniform_rotation_dagger(qml.RY, alpha, control_wires, target_wire)


def _get_alpha_z(omega, n, k):
    r"""Computes the rotation angles alpha for the Z rotations.

    Args:
        omega (float): phase of the input
        n (int): total number of qubits
        k (int): current qubit

    Returns:
        scipy.sparse.dok_matrix[np.float64]: a sparse vector representing :math:`\alpha^z_k`
    """
    alpha_z_k = sparse.dok_matrix((2 ** (n - k), 1), dtype=np.float64)

    for (i, _), om in omega.items():
        i += 1
        j = int(np.ceil(i * 2 ** (-k)))
        s_condition = 2 ** (k - 1) * (2 * j - 1)
        s_i = 1.0 if i > s_condition else -1.0
        alpha_z_k[j - 1, 0] = alpha_z_k[j - 1, 0] + s_i * om / 2 ** (k - 1)

    return alpha_z_k


def _get_alpha_y(a, n, k):
    r"""Computes the rotation angles alpha for the Y rotations.

    Args:
        omega (float): phase of the input
        n (int): total number of qubits
        k (int): current qubit

    Returns:
        scipy.sparse.dok_matrix[np.float64]: a sparse vector representing :math:`\alpha^y_k`
    """
    alpha = sparse.dok_matrix((2 ** (n - k), 1), dtype=np.float64)

    numerator = sparse.dok_matrix((2 ** (n - k), 1), dtype=np.float64)
    denominator = sparse.dok_matrix((2 ** (n - k), 1), dtype=np.float64)

    for (i, _), e in a.items():
        j = int(math.ceil((i + 1) / 2 ** k))
        l = (i + 1) - (2 * j - 1) * 2 ** (k - 1)
        is_part_numerator = 1 <= l <= 2 ** (k - 1)

        if is_part_numerator:
            numerator[j - 1, 0] += e * e
        denominator[j - 1, 0] += e * e

    for (j, _), e in numerator.items():
        numerator[j, 0] = math.sqrt(e)
    for (j, _), e in denominator.items():
        denominator[j, 0] = 1 / math.sqrt(e)

    pre_alpha = numerator.multiply(denominator)  # type: sparse.csr_matrix
    for (j, _), e in pre_alpha.todok().items():
        alpha[j, 0] = 2 * np.arcsin(e)

    return alpha


@template
def MottonenStatePreparation(state_vector, wires):
    r"""
    Prepares an arbitrary state on the given wires using a decomposition into gates developed
    by Möttönen et al. (Quantum Info. Comput., 2005).

    The state is prepared via a sequence
    of "uniformly controlled rotations". A uniformly controlled rotation on a target qubit is
    composed from all possible controlled rotations on said qubit and can be used to address individual
    elements of the state vector. In the work of Mottonen et al., the inverse of their state preparation
    is constructed by first equalizing the phases of the state vector via uniformly controlled Z rotations
    and then rotating the now real state vector into the direction of the state :math:`|0\rangle` via
    uniformly controlled Y rotations.

    This code is adapted from code written by Carsten Blank for PennyLane-Qiskit.

    Args:
        state_vector (array): Input array of shape ``(2^N,)``, where N is the number of wires
            the state preparation acts on. ``N`` must be smaller or equal to the total
            number of wires.
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Raises:
        ValueError: if inputs do not have the correct format
    """

    ###############
    # Input checks

    wires = check_wires(wires)

    n_wires = len(wires)
    expected_shape = (2 ** n_wires,)
    check_shape(
        state_vector,
        expected_shape,
        msg="'state_vector' must be of shape {}; got {}."
        "".format(expected_shape, get_shape(state_vector)),
    )

    # check if state_vector is normalized
    if isinstance(state_vector[0], Variable):
        state_vector_values = [s.val for s in state_vector]
        norm = np.sum(np.abs(state_vector_values) ** 2)
    else:
        norm = np.sum(np.abs(state_vector) ** 2)
    if not np.isclose(norm, 1.0, atol=1e-3):
        raise ValueError("'state_vector' has to be of length 1.0, got {}".format(norm))

    #######################

    # Change ordering of indices, original code was for IBM machines
    state_vector = np.array(state_vector).reshape([2] * n_wires).T.flatten()[:, np.newaxis]
    state_vector = sparse.dok_matrix(state_vector)

    wires = np.array(wires)

    a = sparse.dok_matrix(state_vector.shape)
    omega = sparse.dok_matrix(state_vector.shape)

    for (i, j), v in state_vector.items():
        if isinstance(v, Variable):
            a[i, j] = np.absolute(v.val)
            omega[i, j] = np.angle(v.val)
        else:
            a[i, j] = np.absolute(v)
            omega[i, j] = np.angle(v)
    # This code is directly applying the inverse of Carsten Blank's
    # code to avoid inverting at the end

    # Apply y rotations
    for k in range(n_wires, 0, -1):
        alpha_y_k = _get_alpha_y(a, n_wires, k)  # type: sparse.dok_matrix
        control = wires[k:]
        target = wires[k - 1]
        _uniform_rotation_y_dagger(alpha_y_k, control, target)

    # Apply z rotations
    for k in range(n_wires, 0, -1):
        alpha_z_k = _get_alpha_z(omega, n_wires, k)
        control = wires[k:]
        target = wires[k - 1]
        if len(alpha_z_k) > 0:
            _uniform_rotation_z_dagger(alpha_z_k, control, target)

# Copyright 2018 Xanadu Quantum Technologies Inc.

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
State Preparations
==================

**Module name:** :mod:`pennylane.templates.state_preparations`

.. currentmodule:: pennylane.templates.state_preparations

This module provides routines that prepare a given state using only
elementary gates.

Qubit architectures
-------------------

.. autosummary::

    BasisStatePreparation
    MottonenStatePreparation

Code details
^^^^^^^^^^^^
"""
from collections.abc import Iterable
import numpy as np
import math
from scipy import sparse
from sympy.combinatorics.graycode import GrayCode

import pennylane as qml


def BasisStatePreparation(basis_state, wires):
    r"""
    Prepares a basis state on the given wires using a sequence of Pauli X gates.

    Args:
        basis_state (array): Input array of shape ``(N,)``, where N is the number of qubits,
            with :math:`N\leq n`
        wires (Sequence[int]): sequence of qubit indices that the template acts on
    """

    if not isinstance(wires, Iterable):
        raise ValueError(
            "Wires must be passed as a list of integers; got {}.".format(wires)
        )

    if not len(basis_state) == len(wires):
        raise ValueError(
            "Number of qubits must be equal to the number of wires, which is {}; "
            "got {}.".format(len(wires), len(basis_state))
        )

    if any([x not in [0, 1] for x in basis_state]):
        raise ValueError(
            "Basis state must only consist of 0s and 1s, got {}".format(basis_state)
        )

    for wire, state in zip(wires, basis_state):
        if state == 1:
            qml.PauliX(wire)

def matrix_M_entry(row, col):
    """The matrix entry for the angle computation.

    Args:
        row (int): one-based row number
        col (int): one-based column number

    Returns:
        (float): matrix entry
    """
    # (col >> 1) ^ col is the Gray code of col
    b_and_g = row & ((col >> 1) ^ col)
    sum_of_ones = 0
    while b_and_g > 0:
        if b_and_g & 0b1:
            sum_of_ones += 1

        b_and_g = b_and_g >> 1

    return (-1)**sum_of_ones

def compute_theta(alpha):
    """Calculates the rotation angles from the alpha vector

    Args:
        alpha (array[float]): alpha parameters

    Returns:
        (array[float]): rotation angles theta
    """
    k = np.log2(alpha.shape[0])
    factor = 2**(-k)

    theta = sparse.dok_matrix(alpha.shape, dtype=np.float64)  # type: sparse.dok_matrix

    for row in range(alpha.shape[0]):
        # Use transpose of M:
        entry = sum([matrix_M_entry(col, row) * a for (col, _), a in alpha.items()])
        entry *= factor
        if abs(entry) > 1e-6:
            theta[row, 0] = entry

    return theta

def uniform_rotation(gate, alpha, control_wires, target_wire):
    """Applies a Y or Z rotation to the target qubit
    that is uniformly controlled by the control qubits.

    Args:
        gate (qml.Operation): gate to be applied
        alpha (array[float]): alpha parameters
        control_wires (array[int]): wires that act as control
        target_wire (int): wire that acts as target
    """

    theta = compute_theta(alpha)  # type: sparse.dok_matrix

    gray_code_rank = len(control_wires)

    if gray_code_rank == 0:
        gate(theta[0, 0], wires=[target_wire], do_queue=False)
        return

    gc = GrayCode(gray_code_rank)

    current_gray = gc.current
    for i in range(gc.selections):
        gate(theta[i, 0], wires=[target_wire], do_queue=False)
        next_gray = gc.next(i + 1).current

        control_index = int(np.log2(int(current_gray, 2) ^ int(next_gray, 2)))
        qml.CNOT(wires=[control_wires[control_index], target_wire], do_queue=False)

        current_gray = next_gray

def unirz_dg(alpha, control_wires, target_wire):
    """Applies a Z rotation to the target qubit
    that is uniformly controlled by the control qubits.

    Args:
        alpha (array[float]): alpha parameters
        control_wires (array[int]): wires that act as control
        target_wire (int): wire that acts as target
    """

    uniform_rotation(qml.RZ, alpha, control_wires, target_wire)

def uniry_dg(alpha, control_wires, target_wire):
    """Applies a Y rotation to the target qubit
    that is uniformly controlled by the control qubits.

    Args:
        alpha (array[float]): alpha parameters
        control_wires (array[int]): wires that act as control
        target_wire (int): wire that acts as target
    """

    uniform_rotation(qml.RY, alpha, control_wires, target_wire)


def get_alpha_z(omega, n, k):
    """Computes the rotation angles alpha for the z-rotations

    Args:
        omega (float): phase of the input
        n (int): total number of qubits
        k (int): current qubit
        
    Returns:
        a sparse vector representing :math:`\alpha^z_k`
    """
    alpha_z_k = sparse.dok_matrix((2 ** (n - k), 1), dtype=np.float64)

    for (i, _), om in omega.items():
        i += 1
        j = int(np.ceil(i * 2 ** (-k)))
        s_condition = 2 ** (k - 1) * (2 * j - 1)
        s_i = 1.0 if i > s_condition else -1.0
        alpha_z_k[j - 1, 0] = alpha_z_k[j - 1, 0] + s_i * om / 2 ** (k - 1)

    return alpha_z_k


def get_alpha_y(a, n, k):
    """Computes the rotation angles alpha for the z-rotations

    Args:
        omega (float): phase of the input
        n (int): total number of qubits
        k (int): current qubit
        
    Returns:
        a sparse vector representing :math:`\alpha^y_k`
    """
    alpha = sparse.dok_matrix((2**(n - k), 1), dtype=np.float64)

    numerator = sparse.dok_matrix((2 ** (n - k), 1), dtype=np.float64)
    denominator = sparse.dok_matrix((2 ** (n - k), 1), dtype=np.float64)

    for (i, _), e in a.items():
        j = int(math.ceil((i + 1) / 2**k))
        l = (i + 1) - (2*j - 1)*2**(k-1)
        is_part_numerator = 1 <= l <= 2**(k-1)

        if is_part_numerator:
            numerator[j - 1, 0] += e*e
        denominator[j - 1, 0] += e*e

    for (j, _), e in numerator.items():
        numerator[j, 0] = np.sqrt(e)
    for (j, _), e in denominator.items():
        denominator[j, 0] = 1/np.sqrt(e)

    pre_alpha = numerator.multiply(denominator)  # type: sparse.csr_matrix
    for (j, _), e in pre_alpha.todok().items():
        alpha[j, 0] = 2*np.arcsin(e)

    return alpha
def MottonenStatePreparation(state_vector, wires):
    r"""
    Prepares an arbitrary state on the given wires using a decomposition into gates developed
    by Möttönen et al. :cite:`mottonen2005transformation`.

    This code is adapted from code written by Carsten Blank for PennyLane-Qiskit.

    Args:
        state_vector (array): Input array of shape ``(2^N,)``, where N is the number of qubits,
            with :math:`N\leq n`
        wires (Sequence[int]): sequence of qubit indices that the template acts on
    """

    if not isinstance(wires, Iterable):
        raise ValueError(
            "Wires needs to be a list of wires that the embedding uses; got {}.".format(wires)
        )

    n = len(wires)

    if not len(state_vector) == 2**n:
        raise ValueError(
            "Number of qubits must be equal to 2 to the power of the number of wires, which is {}; "
            "got {}.".format(2**n, len(state_vector))
        )

    probability_sum = np.sum(np.abs(state_vector)**2)

    if not np.isclose(probability_sum, 1.0, atol=1E-3):
        raise ValueError(
            "State vector probabilities have to sum up to 1.0, got {}".format(probability_sum)
        )
    
    state_vector = np.array(state_vector)[:, np.newaxis]
    state_vector = sparse.dok_matrix(state_vector)
    wires = np.array(wires)

    a = sparse.dok_matrix(state_vector.shape)
    omega = sparse.dok_matrix(state_vector.shape)

    for (i, j), v in state_vector.items():
        a[i, j] = np.absolute(v)
        omega[i, j] = np.angle(v)

    # This code is directly applying the inverse of Carsten Blank's
    # code to avoid inverting at the end

    # Apply y rotations
    for k in range(n, 0, -1):
        alpha_y_k = get_alpha_y(a, n, k)  # type: sparse.dok_matrix
        control = wires[k:]
        target = wires[k - 1]
        uniry_dg(alpha_y_k, control, target)
    
    # Apply z rotations
    for k in range(n, 0, -1):
        alpha_z_k = get_alpha_z(omega, n, k)
        control = wires[k:]
        target = wires[k - 1]
        if len(alpha_z_k) > 0:
            unirz_dg(alpha_z_k, control, target)

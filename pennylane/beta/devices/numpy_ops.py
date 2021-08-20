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
r"""
Utility functions and numerical implementations of operations for
the default.tensor device.
"""
import math
import cmath
import numpy as np
from numpy.linalg import eigh

# ========================================================
#  utilities
# ========================================================


def spectral_decomposition(A):
    r"""Spectral decomposition of a Hermitian matrix.

    Args:
        A (array): Hermitian matrix

    Returns:
        (vector[float], list[array[complex]]): (a, P): eigenvalues and hermitian projectors
            such that :math:`A = \sum_k a_k P_k`.
    """
    d, v = eigh(A)
    P = []
    for k in range(d.shape[0]):
        temp = v[:, k]
        P.append(np.outer(temp, temp.conj()))
    return d, P


# ========================================================
#  fixed gates/observables
# ========================================================

I = np.eye(2)
# Pauli matrices
X = np.array([[0, 1], [1, 0]])  #: Pauli-X matrix
Y = np.array([[0, -1j], [1j, 0]])  #: Pauli-Y matrix
Z = np.array([[1, 0], [0, -1]])  #: Pauli-Z matrix

H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)  #: Hadamard gate
# Two qubit gates
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])  #: CNOT gate
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  #: SWAP gate
CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])  #: CZ gate
S = np.array([[1, 0], [0, 1j]])  #: Phase Gate
T = np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]])  #: T Gate
# Three qubit gates
CSWAP = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)  #: CSWAP gate

Toffoli = np.diag([1 for i in range(8)])
Toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])


def identity(*_):
    """Identity matrix observable.

    Returns:
        array: 2x2 identity matrix
    """
    return np.identity(2)


# ========================================================
#  parametrized gates
# ========================================================


def Rphi(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle
    Returns:
        array: unitary 2x2 phase shift matrix
    """
    return np.array([[1, 0], [0, cmath.exp(1j * phi)]])


def Rotx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * X


def Roty(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * Y


def Rotz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return math.cos(theta / 2) * I + 1j * math.sin(-theta / 2) * Z


def Rot3(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return Rotz(c) @ (Roty(b) @ Rotz(a))


def CRotx(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, math.cos(theta / 2), -1j * math.sin(theta / 2)],
            [0, 0, -1j * math.sin(theta / 2), math.cos(theta / 2)],
        ]
    )


def CRoty(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, math.cos(theta / 2), -math.sin(theta / 2)],
            [0, 0, math.sin(theta / 2), math.cos(theta / 2)],
        ]
    )


def CRotz(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cmath.exp(-1j * theta / 2), 0],
            [0, 0, 0, cmath.exp(1j * theta / 2)],
        ]
    )


def CRot3(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [
                0,
                0,
                cmath.exp(-1j * (a + c) / 2) * math.cos(b / 2),
                -cmath.exp(1j * (a - c) / 2) * math.sin(b / 2),
            ],
            [
                0,
                0,
                cmath.exp(-1j * (a - c) / 2) * math.sin(b / 2),
                cmath.exp(1j * (a + c) / 2) * math.cos(b / 2),
            ],
        ]
    )


# ========================================================
#  General gates and observables
# ========================================================


def unitary(*args):
    r"""Represent input argument as a unitary matrix.

    Args:
        args (array): matrix to be represented as a unitary operator

    Raises:
        ValueError: if the matrix is not unitary or square

    Returns:
        array: square unitary matrix
    """
    U = np.asarray(args[0])

    if U.shape[0] != U.shape[1]:
        raise ValueError("Operator must be a square matrix.")

    if not np.allclose(U @ U.conj().T, np.identity(U.shape[0])):
        raise ValueError("Operator must be unitary.")

    return U


def hermitian(*args):
    r"""Represent input argument as a hermitian matrix.

    Args:
        args (array): hermitian

    Raises:
        ValueError: if the matrix is not Hermitian or square

    Returns:
        array: square hermitian matrix
    """
    A = np.asarray(args[0])
    if A.shape[0] != A.shape[1]:
        raise ValueError("Expectation must be a square matrix.")

    if not np.allclose(A, A.conj().T):
        raise ValueError("Expectation must be Hermitian.")

    return A

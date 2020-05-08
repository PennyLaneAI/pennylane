"""Convenience gate representations for testing"""
import math
import cmath
import numpy as np

# ========================================================
#  fixed gates
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

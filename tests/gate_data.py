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

II = np.eye(4, dtype=np.complex128)
XX = np.array(np.kron(X, X), dtype=np.complex128)
YY = np.array(np.kron(Y, Y), dtype=np.complex128)

# Single-qubit projectors
StateZeroProjector = np.array([[1, 0], [0, 0]])
StateOneProjector = np.array([[0, 0], [0, 1]])

# Two qubit gates
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])  #: CNOT gate
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  #: SWAP gate
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])  #: ISWAP gate
SISWAP = np.array(
    [
        [1, 0, 0, 0],
        [0, 1 / math.sqrt(2), 1 / math.sqrt(2) * 1j, 0],
        [0, 1 / math.sqrt(2) * 1j, 1 / math.sqrt(2), 0],
        [0, 0, 0, 1],
    ]
)
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

w = np.exp(2 * np.pi * 1j / 8)
QFT = (
    np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, w, w ** 2, w ** 3, w ** 4, w ** 5, w ** 6, w ** 7],
            [1, w ** 2, w ** 4, w ** 6, 1, w ** 2, w ** 4, w ** 6],
            [1, w ** 3, w ** 6, w, w ** 4, w ** 7, w ** 2, w ** 5],
            [1, w ** 4, 1, w ** 4, 1, w ** 4, 1, w ** 4],
            [1, w ** 5, w ** 2, w ** 7, w ** 4, w, w ** 6, w ** 3],
            [1, w ** 6, w ** 4, w ** 2, 1, w ** 6, w ** 4, w ** 2],
            [1, w ** 7, w ** 6, w ** 5, w ** 4, w ** 3, w ** 2, w],
        ]
    )
    / np.sqrt(8)
)

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


def MultiRZ1(theta):
    r"""Arbitrary multi Z rotation on one wire.

    Args:
        theta (float): rotation angle

    Returns:
        array: the one-wire MultiRZ matrix
    """
    return np.array([[np.exp(-1j * theta / 2), 0.0 + 0.0j], [0.0 + 0.0j, np.exp(1j * theta / 2)]])


def MultiRZ2(theta):
    r"""Arbitrary multi Z rotation on two wires.

    Args:
        theta (float): rotation angle

    Returns:
        array: the two-wire MultiRZ matrix
    """
    return np.array(
        [
            [np.exp(-1j * theta / 2), 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, np.exp(1j * theta / 2), 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, np.exp(1j * theta / 2), 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, np.exp(-1j * theta / 2)],
        ]
    )


def IsingXX(phi):
    r"""Ising XX coupling gate

    .. math:: XX(\phi) = \begin{bmatrix}
        \cos(\phi / 2) & 0 & 0 & -i \sin(\phi / 2) \\
        0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
        0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
        -i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    Args:
        phi (float): rotation angle :math:`\phi`
    Returns:
        array[complex]: unitary 4x4 rotation matrix
    """
    return np.cos(phi / 2) * II - 1j * np.sin(phi / 2) * XX


def IsingYY(phi):
    r"""Ising YY coupling gate.

    .. math:: YY(\phi) = \begin{bmatrix}
        \cos(\phi / 2) & 0 & 0 & i \sin(\phi / 2) \\
        0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
        0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
        i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    Args:
        phi (float): rotation angle :math:`\phi`
    Returns:
        array[complex]: unitary 4x4 rotation matrix
    """
    return np.cos(phi / 2) * II - 1j * np.sin(phi / 2) * YY


def IsingZZ(phi):
    r"""Ising ZZ coupling gate

    .. math:: ZZ(\phi) = \begin{bmatrix}
        e^{-i \phi / 2} & 0 & 0 & 0 \\
        0 & e^{i \phi / 2} & 0 & 0 \\
        0 & 0 & e^{i \phi / 2} & 0 \\
        0 & 0 & 0 & e^{-i \phi / 2}
        \end{bmatrix}.

    Args:
        phi (float): rotation angle :math:`\phi`
    Returns:
        array[complex]: unitary 4x4 rotation matrix
    """
    e_m = np.exp(-1j * phi / 2)
    e = np.exp(1j * phi / 2)
    return np.array([[e_m, 0, 0, 0], [0, e, 0, 0], [0, 0, e, 0], [0, 0, 0, e_m]])


def ControlledPhaseShift(phi):
    r"""Controlled phase shift.

    Args:
        phi (float): rotation angle

    Returns:
        array: the two-wire controlled-phase matrix
    """
    return np.diag([1, 1, 1, np.exp(1j * phi)])


def SingleExcitation(phi):
    r"""Single excitation rotation.

    Args:
        phi (float): rotation angle

    Returns:
        array: the two-qubit Givens rotation describing the single excitation operation
    """

    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi / 2), -np.sin(phi / 2), 0],
            [0, np.sin(phi / 2), np.cos(phi / 2), 0],
            [0, 0, 0, 1],
        ]
    )


def SingleExcitationPlus(phi):
    r"""Single excitation rotation with positive phase shift.

    Args:
    phi (float): rotation angle

    Returns:
        array: the two-qubit Givens rotation describing the single excitation operation
    """

    return np.array(
        [
            [np.exp(1j * phi / 2), 0, 0, 0],
            [0, np.cos(phi / 2), -np.sin(phi / 2), 0],
            [0, np.sin(phi / 2), np.cos(phi / 2), 0],
            [0, 0, 0, np.exp(1j * phi / 2)],
        ]
    )


def SingleExcitationMinus(phi):
    r"""Single excitation rotation with negative phase shift.

    Args:
        phi (float): rotation angle

    Returns:
        array: the two-qubit matrix describing the operation
    """

    return np.array(
        [
            [np.exp(-1j * phi / 2), 0, 0, 0],
            [0, np.cos(phi / 2), -np.sin(phi / 2), 0],
            [0, np.sin(phi / 2), np.cos(phi / 2), 0],
            [0, 0, 0, np.exp(-1j * phi / 2)],
        ]
    )


def DoubleExcitation(phi):
    r"""Double excitation rotation.

    Args:
        phi (float): rotation angle
    Returns:
        array: the four-qubit Givens rotation describing the double excitation
    """

    c = math.cos(phi / 2)
    s = math.sin(phi / 2)

    U = np.eye(16)
    U[3, 3] = c  # 3 (dec) = 0011 (bin)
    U[3, 12] = -s  # 12 (dec) = 1100 (bin)
    U[12, 3] = s
    U[12, 12] = c

    return U


def DoubleExcitationPlus(phi):
    r"""Double excitation rotation with positive phase shift.

    Args:
        phi (float): rotation angle

    Returns:
        array: the four-qubit matrix describing the operation
    """

    c = math.cos(phi / 2)
    s = math.sin(phi / 2)
    e = cmath.exp(1j * phi / 2)

    U = e * np.eye(16, dtype=np.complex64)
    U[3, 3] = c  # 3 (dec) = 0011 (bin)
    U[3, 12] = -s  # 12 (dec) = 1100 (bin)
    U[12, 3] = s
    U[12, 12] = c

    return U


def DoubleExcitationMinus(phi):
    r"""Double excitation rotation with negative phase shift.

    Args:
        phi (float): rotation angle
    Returns:
        array: the four-qubit matrix describing the operation
    """

    c = math.cos(phi / 2)
    s = math.sin(phi / 2)
    e = cmath.exp(-1j * phi / 2)

    U = e * np.eye(16, dtype=np.complex64)
    U[3, 3] = c  # 3 (dec) = 0011 (bin)
    U[3, 12] = -s  # 12 (dec) = 1100 (bin)
    U[12, 3] = s
    U[12, 12] = c

    return U

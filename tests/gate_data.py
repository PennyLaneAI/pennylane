"""Convenience gate representations for testing"""

import pennylane as qml
from pennylane import math
from pennylane import numpy as np

# ========================================================
#  fixed gates
# ========================================================

I = math.eye(2)

# Pauli matrices
X = math.array([[0, 1], [1, 0]])  #: Pauli-X matrix
Y = math.array([[0, -1j], [1j, 0]])  #: Pauli-Y matrix
Z = math.array([[1, 0], [0, -1]])  #: Pauli-Z matrix

H = math.array([[1, 1], [1, -1]]) / math.sqrt(2)  #: Hadamard gate

II = math.eye(4, dtype=np.complex128) + 0j
XX = math.array(math.kron(X, X), dtype=np.complex128)
YY = math.array(math.kron(Y, Y), dtype=np.complex128)

# Single-qubit projectors
StateZeroProjector = math.array([[1, 0], [0, 0]])
StateOneProjector = math.array([[0, 0], [0, 1]])

# Two qubit gates
CNOT = math.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])  #: CNOT gate
SWAP = math.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  #: SWAP gate
ISWAP = math.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])  #: ISWAP gate
SISWAP = math.array(
    [
        [1, 0, 0, 0],
        [0, 1 / math.sqrt(2), 1 / math.sqrt(2) * 1j, 0],
        [0, 1 / math.sqrt(2) * 1j, 1 / math.sqrt(2), 0],
        [0, 0, 0, 1],
    ]
)
CZ = math.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])  #: CZ gate
CY = math.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])  #: CY gate
S = math.array([[1, 0], [0, 1j]])  #: Phase Gate
T = math.array([[1, 0], [0, math.exp(1j * np.pi / 4)]])  #: T Gate
SX = 0.5 * math.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])  #: SX Gate
ECR = math.array(
    [
        [0, 0, 1 / math.sqrt(2), 1j * 1 / math.sqrt(2)],  # ECR Gate
        [0, 0, 1j * 1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / math.sqrt(2), -1j * 1 / math.sqrt(2), 0, 0],
        [-1j * 1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0],
    ]
)
CH = math.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1 / math.sqrt(2), 1 / math.sqrt(2)],
        [0, 0, 1 / math.sqrt(2), -1 / math.sqrt(2)],
    ]
)  # CH gate

# Three qubit gates
CSWAP = math.array(
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

Toffoli = math.diag([1 for i in range(8)])
Toffoli[6:8, 6:8] = math.array([[0, 1], [1, 0]])

CCZ = math.diag([1] * 7 + [-1])

w = math.exp(2 * np.pi * 1j / 8)
QFT = math.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, w, w**2, w**3, w**4, w**5, w**6, w**7],
        [1, w**2, w**4, w**6, 1, w**2, w**4, w**6],
        [1, w**3, w**6, w, w**4, w**7, w**2, w**5],
        [1, w**4, 1, w**4, 1, w**4, 1, w**4],
        [1, w**5, w**2, w**7, w**4, w, w**6, w**3],
        [1, w**6, w**4, w**2, 1, w**6, w**4, w**2],
        [1, w**7, w**6, w**5, w**4, w**3, w**2, w],
    ]
) / math.sqrt(8)

# Qutrit gates
OMEGA = np.exp(2 * np.pi * 1j / 3)

TSHIFT = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # Qutrit right-shift gate

TCLOCK = np.array([[1, 0, 0], [0, OMEGA, 0], [0, 0, OMEGA**2]])  # Qutrit clock gate

TH = (-1j / np.sqrt(3)) * np.array(
    [[1, 1, 1], [1, OMEGA, OMEGA**2], [1, OMEGA**2, OMEGA]]
)  # hadamard gate

TSWAP = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=np.complex128,
)  # Ternary swap gate

TADD = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
    ],
    dtype=np.complex128,
)  # Ternary add gate

GELL_MANN = np.zeros((8, 3, 3), dtype=np.complex128)
GELL_MANN[0] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
GELL_MANN[1] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
GELL_MANN[2] = np.diag([1, -1, 0])
GELL_MANN[3] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
GELL_MANN[4] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
GELL_MANN[5] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
GELL_MANN[6] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
GELL_MANN[7] = np.diag([1, 1, -2]) / np.sqrt(3)


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
    return math.array([[1, 0], [0, math.exp(1j * phi)]], like=phi)


def Rotx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return qml.math.array(
        [
            [qml.math.cos(0.5 * theta), -1j * qml.math.sin(0.5 * theta)],
            [-1j * qml.math.sin(0.5 * theta), qml.math.cos(0.5 * theta)],
        ],
        like=theta,
    )


def Roty(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return (
        qml.math.array(
            [
                [qml.math.cos(0.5 * theta), -qml.math.sin(0.5 * theta)],
                [qml.math.sin(0.5 * theta), qml.math.cos(0.5 * theta)],
            ],
            like=theta,
        )
        + 0j
    )


def Rotz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return qml.math.array(
        [[qml.math.exp(-0.5j * theta), 0.0], [0.0, qml.math.exp(0.5j * theta)]], like=theta
    )


def Rot3(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return Rotz(c) @ (Roty(b) @ Rotz(a))


def U1(phi):
    r""" Return the matrix representation of the U1 gate.

    .. math:: U_1(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
            1 & 0 \\
            0 & e^{i\phi}
        \end{bmatrix}.

    Args:
        phi (float): rotation angle :math:`\phi`
    """
    return math.array([[1.0, 0.0], [0.0, math.exp(phi * 1j)]], like=phi)


def U2(phi, delta):
    r"""Return the matrix representation of the U2 gate.

    .. math::

        U_2(\phi, \delta) = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -\exp(i \delta)
        \\ \exp(i \phi) & \exp(i (\phi + \delta)) \end{bmatrix}

    Args:dd
        phi (float): azimuthal angle :math:`\phi`
        delta (float): quantum phase :math:`\delta`
    """
    return (
        1
        / math.sqrt(2)
        * math.array(
            [[1.0, -math.exp(delta * 1j)], [math.exp(phi * 1j), math.exp((phi + delta) * 1j)]]
        )
    )


def U3(theta, phi, delta):
    r"""
    Arbitrary single qubit unitary.

    .. math::

        U_3(\theta, \phi, \delta) = \begin{bmatrix} \cos(\theta/2) & -\exp(i \delta)\sin(\theta/2) \\
        \exp(i \phi)\sin(\theta/2) & \exp(i (\phi + \delta))\cos(\theta/2) \end{bmatrix}

    Args:dd
        theta (float): polar angle :math:`\theta`
        phi (float): azimuthal angle :math:`\phi`
        delta (float): quantum phase :math:`\delta`
    """
    return math.array(
        [
            [math.cos(theta / 2), -math.exp(delta * 1j) * math.sin(theta / 2)],
            [
                math.exp(phi * 1j) * math.sin(theta / 2),
                math.exp((phi + delta) * 1j) * math.cos(theta / 2),
            ],
        ]
    )


def CRotx(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    return math.array(
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
    return math.array(
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
    return math.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, math.exp(-1j * theta / 2), 0],
            [0, 0, 0, math.exp(1j * theta / 2)],
        ],
        like=theta,
    )


def CRot3(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return math.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [
                0,
                0,
                math.exp(-1j * (a + c) / 2) * math.cos(b / 2),
                -math.exp(1j * (a - c) / 2) * math.sin(b / 2),
            ],
            [
                0,
                0,
                math.exp(-1j * (a - c) / 2) * math.sin(b / 2),
                math.exp(1j * (a + c) / 2) * math.cos(b / 2),
            ],
        ],
        like=a,
    )


def MultiRZ1(theta):
    r"""Arbitrary multi Z rotation on one wire.

    Args:
        theta (float): rotation angle

    Returns:
        array: the one-wire MultiRZ matrix
    """
    return math.array(
        [[math.exp(-1j * theta / 2), 0.0 + 0.0j], [0.0 + 0.0j, math.exp(1j * theta / 2)]]
    )


def MultiRZ2(theta):
    r"""Arbitrary multi Z rotation on two wires.

    Args:
        theta (float): rotation angle

    Returns:
        array: the two-wire MultiRZ matrix
    """
    return math.array(
        [
            [math.exp(-1j * theta / 2), 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, math.exp(1j * theta / 2), 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, math.exp(1j * theta / 2), 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, math.exp(-1j * theta / 2)],
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
    return math.cos(phi / 2) * II - 1j * math.sin(phi / 2) * XX


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
    return math.cos(phi / 2) * II - 1j * math.sin(phi / 2) * YY


def IsingXY(phi):
    r"""Ising XY coupling gate.

    .. math:: \mathtt{XY}(\phi) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & \cos(\phi / 2) & i \sin(\phi / 2) & 0 \\
            0 & i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    Args:
        phi (float): rotation angle :math:`\phi`
    Returns:
        array[complex]: unitary 4x4 rotation matrix
    """
    mat = II.copy()
    mat[1][1] = math.cos(phi / 2)
    mat[2][2] = math.cos(phi / 2)
    mat[1][2] = 1j * math.sin(phi / 2)
    mat[2][1] = 1j * math.sin(phi / 2)
    return mat


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
    e_m = math.exp(-1j * phi / 2)
    e = math.exp(1j * phi / 2)
    return math.array([[e_m, 0, 0, 0], [0, e, 0, 0], [0, 0, e, 0], [0, 0, 0, e_m]])


def ControlledPhaseShift(phi):
    r"""Controlled phase shift.

    Args:
        phi (float): rotation angle

    Returns:
        array: the two-wire controlled-phase matrix
    """
    return math.diag([1, 1, 1, math.exp(1j * phi)])


def CPhaseShift00(phi):
    r"""Controlled phase shift 00.

    Args:
        phi (float): rotation angle

    Returns:
        array: the two-wire controlled-phase matrix
    """
    return np.diag([np.exp(1j * phi), 1, 1, 1])


def CPhaseShift01(phi):
    r"""Controlled phase shift 01.

    Args:
        phi (float): rotation angle

    Returns:
        array: the two-wire controlled-phase matrix
    """
    return np.diag([1, np.exp(1j * phi), 1, 1])


def CPhaseShift10(phi):
    r"""Controlled phase shift 10.

    Args:
        phi (float): rotation angle

    Returns:
        array: the two-wire controlled-phase matrix
    """
    return np.diag([1, 1, np.exp(1j * phi), 1])


def SingleExcitation(phi):
    r"""Single excitation rotation.

    Args:
        phi (float): rotation angle

    Returns:
        array: the two-qubit Givens rotation describing the single excitation operation
    """

    return math.array(
        [
            [1, 0, 0, 0],
            [0, math.cos(phi / 2), -math.sin(phi / 2), 0],
            [0, math.sin(phi / 2), math.cos(phi / 2), 0],
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

    return math.array(
        [
            [math.exp(1j * phi / 2), 0, 0, 0],
            [0, math.cos(phi / 2), -math.sin(phi / 2), 0],
            [0, math.sin(phi / 2), math.cos(phi / 2), 0],
            [0, 0, 0, math.exp(1j * phi / 2)],
        ]
    )


def SingleExcitationMinus(phi):
    r"""Single excitation rotation with negative phase shift.

    Args:
        phi (float): rotation angle

    Returns:
        array: the two-qubit matrix describing the operation
    """

    return math.array(
        [
            [math.exp(-1j * phi / 2), 0, 0, 0],
            [0, math.cos(phi / 2), -math.sin(phi / 2), 0],
            [0, math.sin(phi / 2), math.cos(phi / 2), 0],
            [0, 0, 0, math.exp(-1j * phi / 2)],
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

    U = math.eye(16)
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
    e = math.exp(1j * phi / 2)

    U = e * math.eye(16, dtype=np.complex128)
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
    e = math.exp(-1j * phi / 2)

    U = e * math.eye(16, dtype=np.complex128)
    U[3, 3] = c  # 3 (dec) = 0011 (bin)
    U[3, 12] = -s  # 12 (dec) = 1100 (bin)
    U[12, 3] = s
    U[12, 12] = c

    return U


def OrbitalRotation(phi):
    r"""Quantum number preserving four-qubit one-parameter gate.

    Args:
        phi (float): rotation angle
    Returns:
        array: the four-qubit matrix describing the operation
    """

    c = math.cos(phi / 2)
    s = math.sin(phi / 2)

    return math.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, c, 0, 0, -s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, c, 0, 0, 0, 0, 0, -s, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, c**2, 0, 0, c * s, 0, 0, -c * s, 0, 0, s**2, 0, 0, 0],
            [0, s, 0, 0, c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -c * s, 0, 0, c**2, 0, 0, s**2, 0, 0, c * s, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, c, 0, 0, 0, 0, 0, s, 0, 0],
            [0, 0, s, 0, 0, 0, 0, 0, c, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, c * s, 0, 0, s**2, 0, 0, c**2, 0, 0, -c * s, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, c, 0, 0, s, 0],
            [0, 0, 0, s**2, 0, 0, -c * s, 0, 0, c * s, 0, 0, c**2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -s, 0, 0, 0, 0, 0, c, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -s, 0, 0, c, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )


def PSWAP(phi):
    r"""Phase SWAP gate
    .. math:: \mathtt{PSWAP}(\phi) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & e^{i \phi} & 0 \\
            0 & e^{i \phi} & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{bmatrix}.
    Args:
        phi (float): rotation angle :math:`\phi`
    Returns:
        array[complex]: unitary 4x4 rotation matrix
    """
    e = math.exp(1j * phi)
    return math.array([[1, 0, 0, 0], [0, 0, e, 0], [0, e, 0, 0], [0, 0, 0, 1]])


def FermionicSWAP(phi):
    r"""Fermionic SWAP rotation gate.

    Args:
        phi (float): rotation angle :math:`\phi`
    Returns:
        array[complex]: unitary 4x4 rotation matrix
    """
    c = math.cos(phi / 2)
    s = math.sin(phi / 2)
    g = math.exp(1j * phi / 2)
    p = math.exp(1j * phi)
    return math.array(
        [[1, 0, 0, 0], [0, g * c, -1j * g * s, 0], [0, -1j * g * s, g * c, 0], [0, 0, 0, p]]
    )

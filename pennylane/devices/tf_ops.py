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
Utility functions and numerical implementations of quantum operations TensorFlow devices.
"""
import tensorflow as tf
from numpy import kron
from pennylane.utils import pauli_eigs

C_DTYPE = tf.complex128
R_DTYPE = tf.float64

I = tf.constant([[1, 0], [0, 1]], dtype=C_DTYPE)
X = tf.constant([[0, 1], [1, 0]], dtype=C_DTYPE)
Y = tf.constant([[0j, -1j], [1j, 0j]], dtype=C_DTYPE)
Z = tf.constant([[1, 0], [0, -1]], dtype=C_DTYPE)

II = tf.eye(4, dtype=C_DTYPE)
ZZ = tf.constant(kron(Z, Z), dtype=C_DTYPE)
XX = tf.constant(kron(X, X), dtype=C_DTYPE)
YY = tf.constant(kron(Y, Y), dtype=C_DTYPE)

IX = tf.constant(kron(I, X), dtype=C_DTYPE)
IY = tf.constant(kron(I, Y), dtype=C_DTYPE)
IZ = tf.constant(kron(I, Z), dtype=C_DTYPE)

ZI = tf.constant(kron(Z, I), dtype=C_DTYPE)
ZX = tf.constant(kron(Z, X), dtype=C_DTYPE)
ZY = tf.constant(kron(Z, Y), dtype=C_DTYPE)


def PhaseShift(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        tf.Tensor[complex]: diagonal part of the phase shift matrix
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    return tf.convert_to_tensor([1.0, tf.exp(1j * phi)])


def ControlledPhaseShift(phi):
    r"""Two-qubit controlled phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        tf.Tensor[complex]: diagonal part of the controlled phase shift matrix
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    return tf.convert_to_tensor([1.0, 1.0, 1.0, tf.exp(1j * phi)])


def RX(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        tf.Tensor[complex]: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return tf.cos(theta / 2) * I + 1j * tf.sin(-theta / 2) * X


def RY(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle

    Returns:
        tf.Tensor[complex]: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return tf.cos(theta / 2) * I + 1j * tf.sin(-theta / 2) * Y


def RZ(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle

    Returns:
        tf.Tensor[complex]: the diagonal part of the rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    p = tf.exp(-0.5j * theta)
    return tf.convert_to_tensor([p, tf.math.conj(p)])


def Rot(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles

    Returns:
        tf.Tensor[complex]: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return tf.linalg.diag(RZ(c)) @ RY(b) @ tf.linalg.diag(RZ(a))


def MultiRZ(theta, n):
    r"""Arbitrary multi Z rotation.

    Args:
        theta (float): rotation angle
        n (int): number of wires the rotation acts on

    Returns:
        tf.Tensor[complex]: diagonal part of the MultiRZ matrix
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    multi_Z_rot_eigs = tf.exp(-1j * theta / 2 * pauli_eigs(n))
    return tf.convert_to_tensor(multi_Z_rot_eigs)


def CRX(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        tf.Tensor[complex]: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return (
        tf.cos(theta / 4) ** 2 * II
        - 1j * tf.sin(theta / 2) / 2 * IX
        + tf.sin(theta / 4) ** 2 * ZI
        + 1j * tf.sin(theta / 2) / 2 * ZX
    )


def CRY(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        tf.Tensor[complex]: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return (
        tf.cos(theta / 4) ** 2 * II
        - 1j * tf.sin(theta / 2) / 2 * IY
        + tf.sin(theta / 4) ** 2 * ZI
        + 1j * tf.sin(theta / 2) / 2 * ZY
    )


def CRZ(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        tf.Tensor[complex]: diagonal part of the 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    p = tf.exp(-0.5j * theta)
    return tf.convert_to_tensor([1.0, 1.0, p, tf.math.conj(p)])


def CRot(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        tf.Tensor[complex]: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return tf.linalg.diag(CRZ(c)) @ (CRY(b) @ tf.linalg.diag(CRZ(a)))


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
        tf.Tensor[complex]: unitary 4x4 rotation matrix
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    return tf.cos(phi / 2) * II - 1j * tf.sin(phi / 2) * XX


def IsingYY(phi):
    r"""Ising YY coupling gate

    .. math:: YY(\phi) = \begin{bmatrix}
        \cos(\phi / 2) & 0 & 0 & i \sin(\phi / 2) \\
        0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
        0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
        i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    Args:
        phi (float): rotation angle :math:`\phi`
    Returns:
        tf.Tensor[complex]: unitary 4x4 rotation matrix
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    return tf.cos(phi / 2) * II - 1j * tf.sin(phi / 2) * YY


def IsingZZ(phi):
    r"""Ising ZZ coupling gate

    .. math:: ZZ(\phi) = \begin{bmatrix}
        e^{-i \phi / 2} & 0 & 0 & 0 \\
        0 & e^{i \phi / 2} & 0 & 0 \\
        0 & 0 & e^{i \phi / 2} & 0 \\
        0 & 0 & 0 & e^{-i \phi / 2}
        \end{bmatrix}.

    Args:
        phi (float): rotation :math:`\phi`
    Returns:
        tf.Tensor[complex]: unitary 4x4 rotation matrix
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    e_m = tf.exp(-1j * phi / 2)
    e = tf.exp(1j * phi / 2)
    return tf.convert_to_tensor([[e_m, 0, 0, 0], [0, e, 0, 0], [0, 0, e, 0], [0, 0, 0, e_m]])


def SingleExcitation(phi):
    r"""Single excitation rotation.

    Args:
        phi (float): rotation angle

    Returns:
        tf.Tensor[complex]: Single excitation rotation matrix
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    c = tf.cos(phi / 2)
    s = tf.sin(phi / 2)

    return tf.convert_to_tensor([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def SingleExcitationPlus(phi):
    r"""Single excitation rotation with positive phase-shift outside the rotation subspace.

    Args:
        phi (float): rotation angle

    Returns:
        tf.Tensor[complex]: Single excitation rotation matrix with positive phase-shift
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    c = tf.cos(phi / 2)
    s = tf.sin(phi / 2)
    e = tf.exp(1j * phi / 2)
    return tf.convert_to_tensor([[e, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, e]])


def SingleExcitationMinus(phi):
    r"""Single excitation rotation with negative phase-shift outside the rotation subspace.

    Args:
        phi (float): rotation angle

    Returns:
        tf.Tensor[complex]: Single excitation rotation matrix with negative phase-shift
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    c = tf.cos(phi / 2)
    s = tf.sin(phi / 2)
    e = tf.exp(-1j * phi / 2)
    return tf.convert_to_tensor([[e, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, e]])


def DoubleExcitation(phi):
    r"""Double excitation rotation.

    Args:
        phi (float): rotation angle
    Returns:
        tf.Tensor[complex]: Double excitation rotation matrix
    """

    phi = tf.cast(phi, dtype=C_DTYPE)
    c = tf.cos(phi / 2)
    s = tf.sin(phi / 2)

    U = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, c, 0, 0, 0, 0, 0, 0, 0, 0, -s, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, s, 0, 0, 0, 0, 0, 0, 0, 0, c, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]

    return tf.convert_to_tensor(U)


def DoubleExcitationPlus(phi):
    r"""Double excitation rotation with positive phase-shift.

    Args:
        phi (float): rotation angle
    Returns:
        tf.Tensor[complex]: rotation matrix
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    c = tf.cos(phi / 2)
    s = tf.sin(phi / 2)
    e = tf.exp(1j * phi / 2)

    U = [
        [e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, c, 0, 0, 0, 0, 0, 0, 0, 0, -s, 0, 0, 0],
        [0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0],
        [0, 0, 0, s, 0, 0, 0, 0, 0, 0, 0, 0, c, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e],
    ]

    return tf.convert_to_tensor(U)


def DoubleExcitationMinus(phi):
    r"""Double excitation rotation with negative phase-shift.

    Args:
        phi (float): rotation angle
    Returns:
        tf.Tensor[complex]: rotation matrix
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    c = tf.cos(phi / 2)
    s = tf.sin(phi / 2)
    e = tf.exp(-1j * phi / 2)

    U = [
        [e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, c, 0, 0, 0, 0, 0, 0, 0, 0, -s, 0, 0, 0],
        [0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0],
        [0, 0, 0, s, 0, 0, 0, 0, 0, 0, 0, 0, c, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e],
    ]

    return tf.convert_to_tensor(U)

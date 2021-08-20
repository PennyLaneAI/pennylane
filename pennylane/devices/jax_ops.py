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
Utility functions and numerical implementations of quantum operations for JAX-based devices.

"""
import jax.numpy as jnp
from pennylane.utils import pauli_eigs

C_DTYPE = jnp.complex64  # Use lower precision for better speed on JAX.
R_DTYPE = jnp.float32

I = jnp.array([[1, 0], [0, 1]], dtype=C_DTYPE)
X = jnp.array([[0, 1], [1, 0]], dtype=C_DTYPE)
Y = jnp.array([[0j, -1j], [1j, 0j]], dtype=C_DTYPE)
Z = jnp.array([[1, 0], [0, -1]], dtype=C_DTYPE)

II = jnp.eye(4, dtype=C_DTYPE)
ZZ = jnp.array(jnp.kron(Z, Z), dtype=C_DTYPE)
XX = jnp.array(jnp.kron(X, X), dtype=C_DTYPE)
YY = jnp.array(jnp.kron(Y, Y), dtype=C_DTYPE)

IX = jnp.array(jnp.kron(I, X), dtype=C_DTYPE)
IY = jnp.array(jnp.kron(I, Y), dtype=C_DTYPE)
IZ = jnp.array(jnp.kron(I, Z), dtype=C_DTYPE)

ZI = jnp.array(jnp.kron(Z, I), dtype=C_DTYPE)
ZX = jnp.array(jnp.kron(Z, X), dtype=C_DTYPE)
ZY = jnp.array(jnp.kron(Z, Y), dtype=C_DTYPE)


def PhaseShift(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        array[complex]: diagonal part of the phase shift matrix
    """
    return jnp.array([1.0, jnp.exp(1j * phi)])


def ControlledPhaseShift(phi):
    r"""Two-qubit controlled phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        array[complex]: diagonal part of the controlled phase shift matrix
    """
    return jnp.array([1.0, 1.0, 1.0, jnp.exp(1j * phi)])


def RX(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        array[complex]: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    return jnp.cos(theta / 2) * I + 1j * jnp.sin(-theta / 2) * X


def RY(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle

    Returns:
        array[complex]: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    return jnp.cos(theta / 2) * I + 1j * jnp.sin(-theta / 2) * Y


def RZ(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle

    Returns:
        array[complex]: the diagonal part of the rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    p = jnp.exp(-0.5j * theta)
    return jnp.array([p, jnp.conj(p)])


def Rot(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles

    Returns:
        array[complex]: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return jnp.diag(RZ(c)) @ RY(b) @ jnp.diag(RZ(a))


def CRX(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        array[complex]: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    return (
        jnp.cos(theta / 4) ** 2 * II
        - 1j * jnp.sin(theta / 2) / 2 * IX
        + jnp.sin(theta / 4) ** 2 * ZI
        + 1j * jnp.sin(theta / 2) / 2 * ZX
    )


def CRY(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array[complex]: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    return (
        jnp.cos(theta / 4) ** 2 * II
        - 1j * jnp.sin(theta / 2) / 2 * IY
        + jnp.sin(theta / 4) ** 2 * ZI
        + 1j * jnp.sin(theta / 2) / 2 * ZY
    )


def CRZ(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array[complex]: diagonal part of the 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    p = jnp.exp(-0.5j * theta)
    return jnp.array([1.0, 1.0, p, jnp.conj(p)])


def CRot(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array[complex]: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return jnp.diag(CRZ(c)) @ (CRY(b) @ jnp.diag(CRZ(a)))


def MultiRZ(theta, n):
    r"""Arbitrary multi Z rotation.

    Args:
        theta (float): rotation angle :math:`\theta`
        wires (Sequence[int] or int): the wires the operation acts on
    Returns:
        array[complex]: diagonal part of the multi-qubit rotation matrix
    """
    return jnp.exp(-1j * theta / 2 * pauli_eigs(n))


def IsingXX(phi):
    r"""Ising XX coupling gate.

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
    return jnp.cos(phi / 2) * II - 1j * jnp.sin(phi / 2) * XX


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
    return jnp.cos(phi / 2) * II - 1j * jnp.sin(phi / 2) * YY


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
    e_m = jnp.exp(-1j * phi / 2)
    e = jnp.exp(1j * phi / 2)
    return jnp.array([[e_m, 0, 0, 0], [0, e, 0, 0], [0, 0, e, 0], [0, 0, 0, e_m]])


def SingleExcitation(phi):
    r"""Single excitation rotation.

    Args:
        phi (float): rotation angle

    Returns:
        jnp.Tensor[float]: Single excitation rotation matrix
    """
    c = jnp.cos(phi / 2)
    s = jnp.sin(phi / 2)
    return jnp.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def SingleExcitationPlus(phi):
    r"""Single excitation rotation with positive phase-shift outside the rotation subspace.

    Args:
        phi (float): rotation angle

    Returns:
        jnp.Tensor[complex]: Single excitation rotation matrix with positive phase-shift
    """
    c = jnp.cos(phi / 2)
    s = jnp.sin(phi / 2)
    e = jnp.exp(1j * phi / 2)
    return jnp.array([[e, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, e]])


def SingleExcitationMinus(phi):
    r"""Single excitation rotation with negative phase-shift outside the rotation subspace.

    Args:
        phi (float): rotation angle

    Returns:
        tf.Tensor[complex]: Single excitation rotation matrix with negative phase-shift
    """
    c = jnp.cos(phi / 2)
    s = jnp.sin(phi / 2)
    e = jnp.exp(-1j * phi / 2)
    return jnp.array([[e, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, e]])


def DoubleExcitation(phi):
    r"""Double excitation rotation.

    Args:
        phi (float): rotation angle
    Returns:
        jnp.Tensor[float]: Double excitation rotation matrix

    """
    c = jnp.cos(phi / 2)
    s = jnp.sin(phi / 2)

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

    return jnp.array(U)


def DoubleExcitationPlus(phi):
    r"""Double excitation rotation with positive phase-shift.

    Args:
        phi (float): rotation angle
    Returns:
        jnp.Tensor[complex]: rotation matrix
    """
    c = jnp.cos(phi / 2)
    s = jnp.sin(phi / 2)
    e = jnp.exp(1j * phi / 2)

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

    return jnp.array(U)


def DoubleExcitationMinus(phi):
    r"""Double excitation rotation with negative phase-shift.

    Args:
        phi (float): rotation angle
    Returns:
        jnp.Tensor[complex]: rotation matrix
    """
    c = jnp.cos(phi / 2)
    s = jnp.sin(phi / 2)
    e = jnp.exp(-1j * phi / 2)

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

    return jnp.array(U)

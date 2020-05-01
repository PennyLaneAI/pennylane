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
Utility functions and numerical implementations of operations for
the default.tensor.tf device.
"""
import tensorflow as tf
from numpy import kron
from pennylane.beta.plugins.numpy_ops import I, X, Y, Z

C_DTYPE = tf.complex128
R_DTYPE = tf.float64

I = tf.constant(I, dtype=C_DTYPE)
X = tf.constant(X, dtype=C_DTYPE)

II = tf.eye(4, dtype=C_DTYPE)
ZZ = tf.constant(kron(Z, Z), dtype=C_DTYPE)

IX = tf.constant(kron(I, X), dtype=C_DTYPE)
IY = tf.constant(kron(I, Y), dtype=C_DTYPE)
IZ = tf.constant(kron(I, Z), dtype=C_DTYPE)

ZI = tf.constant(kron(Z, I), dtype=C_DTYPE)
ZX = tf.constant(kron(Z, X), dtype=C_DTYPE)
ZY = tf.constant(kron(Z, Y), dtype=C_DTYPE)


def Rphi(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        array: unitary 2x2 phase shift matrix
    """
    phi = tf.cast(phi, dtype=C_DTYPE)
    return ((1 + tf.exp(1j * phi)) * I + (1 - tf.exp(1j * phi)) * Z) / 2


def Rotx(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return tf.cos(theta / 2) * I + 1j * tf.sin(-theta / 2) * X


def Roty(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return tf.cos(theta / 2) * I + 1j * tf.sin(-theta / 2) * Y


def Rotz(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 2x2 rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return tf.cos(theta / 2) * I + 1j * tf.sin(-theta / 2) * Z


def Rot3(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles

    Returns:
        array: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return Rotz(c) @ Roty(b) @ Rotz(a)


def CRotx(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return (
        tf.cos(theta / 4) ** 2 * II
        - 1j * tf.sin(theta / 2) / 2 * IX
        + tf.sin(theta / 4) ** 2 * ZI
        + 1j * tf.sin(theta / 2) / 2 * ZX
    )


def CRoty(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return (
        tf.cos(theta / 4) ** 2 * II
        - 1j * tf.sin(theta / 2) / 2 * IY
        + tf.sin(theta / 4) ** 2 * ZI
        + 1j * tf.sin(theta / 2) / 2 * ZY
    )


def CRotz(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    theta = tf.cast(theta, dtype=C_DTYPE)
    return (
        tf.cos(theta / 4) ** 2 * II
        - 1j * tf.sin(theta / 2) / 2 * IZ
        + tf.sin(theta / 4) ** 2 * ZI
        + 1j * tf.sin(theta / 2) / 2 * ZZ
    )


def CRot3(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        array: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return CRotz(c) @ (CRoty(b) @ CRotz(a))

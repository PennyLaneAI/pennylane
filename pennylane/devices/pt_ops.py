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
Utility functions and numerical implementations of quantum operations PyTorch devices.
"""
import tensorflow as tf
from numpy import kron
from pennylane.utils import pauli_eigs

C_DTYPE = torch.complex128
R_DTYPE = torch.float64


# Notes::
#
# tf.cast will be using tensor.complex()
# # try torch.tensor or torch.nn.cons
#
# real = torch.tensor([[0, 1],[1,0]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# z = torch.complex(real, imag)
#
# tensor([[1.+0.j, 0.+0.j],
#         [0.+0.j, 1.+0.j]], dtype=torch.complex128)
#
# array([[1.+0.j, 0.+0.j],
#        [0.+0.j, 1.+0.j]])>
#
# tensor([[0.+1.j, 0.+0.j],
#         [0.+0.j, 0.+1.j]], dtype=torch.complex128)
#
# tensor([[0.+1.j, 0.+0.j],
#         [0.+0.j, 0.+1.j]], dtype=torch.complex128)
#
# torch.as_tensor(
# [[0.+0.j 1.+0.j]
#  [1.+0.j 0.+0.j]], shape=(2, 2), dtype=complex128)
#
# ensor([[0.+0.j, 1.+0.j],
#         [1.+0.j, 0.+0.j]], dtype=torch.complex128)
#
#
# torch.as_tensor([[ 0.+0.j -0.-1.j]
#  [ 0.+1.j  0.+0.j]], shape=(2, 2), dtype=complex128)
#
# convert_to_tensor
# >>> a = numpy.array([1, 2, 3])
# >>> t = torch.as_tensor(a)
#
# >>> torch.cos(a)
# tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
# cos and sin are same
#
# >>> z = torch.complex(real, imag)
# >>> z
# tensor([(1.+3.j),
#          (2.+4.j)])

#I = tf.constant([[1, 0],[0, 1]], dtype=C_DTYPE)

#real =
#imag =
I = torch.complex(torch.tensor([[1, 0],[0,1]], dtype=R_DTYPE),
                    torch.tensor([[0, 0],[0,0]], dtype=R_DTYPE))

# X = tf.constant([[0, 1], [1, 0]], dtype=C_DTYPE)
# real = torch.tensor([[0, 1],[1,0]], dtype=R_DTYPE)
# imag =

X = torch.complex(torch.tensor([[0, 1],[1,0]], dtype=R_DTYPE),
                    torch.tensor([[0, 0],[0,0]], dtype=R_DTYPE))


# Y = tf.constant([[0j, -1j], [1j, 0j]], dtype=C_DTYPE)
# real =
# imag =

y = torch.complex(torch.tensor([[0, -0],[0,0]], dtype=torch.float64), torch.tensor([[0, -1],[1,0]], dtype=torch.float64))

# Z = tf.constant([[1, 0], [0, -1]], dtype=C_DTYPE)
#
# real =
# imag =
Z = torch.complex(torch.tensor([[1, 0],[0,-1]], dtype=torch.float64), torch.tensor([[0, 0],[0,0]], dtype=torch.float64))

# # torch.eye
# II = tf.eye(4, dtype=C_DTYPE)

II = torch.eye(4)
II = II.type(torch.complex128)
#
# ZZ = tf.constant(kron(Z, Z), dtype=C_DTYPE)
# real = torch.tensor([[1, 0],[0,-1]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# z = torch.complex(real, imag)

ZZ = torch.tensor(kron(Z,Z))


# IX = tf.constant(kron(I, X), dtype=C_DTYPE)
# real = torch.tensor([[1, 0],[0,1]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# i = torch.complex(real, imag)
#
# real = torch.tensor([[0, 1],[1,0]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# x = torch.complex(real, imag)

IX = torch.tensor(kron(I,X))




# IY = tf.constant(kron(I, Y), dtype=C_DTYPE)
# real = torch.tensor([[1, 0],[0,1]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# i = torch.complex(real, imag)
# real = torch.tensor([[0, -0],[0,0]], dtype=torch.float64)
# imag = torch.tensor([[0, -1],[1,0]], dtype=torch.float64)
# y = torch.complex(real, imag)

IY = torch.tensor(kron(I,Y))

#
# IZ = tf.constant(kron(I, Z), dtype=C_DTYPE)
# real = torch.tensor([[1, 0],[0,1]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# i = torch.complex(real, imag)
# real = torch.tensor([[1, 0],[0,-1]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# z = torch.complex(real, imag)

IZ = torch.tensor(I,Z)

#
# ZI = tf.constant(kron(Z, I), dtype=C_DTYPE)
# real = torch.tensor([[1, 0],[0,-1]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# z = torch.complex(real, imag)
# real = torch.tensor([[1, 0],[0,1]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# i = torch.complex(real, imag)

ZI = torch.tensor(Z,I)

# ZX = tf.constant(kron(Z, X), dtype=C_DTYPE)
# real = torch.tensor([[1, 0],[0,-1]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# z = torch.complex(real, imag)
# real = torch.tensor([[0, 1],[1,0]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# x = torch.complex(real, imag)


ZX = torch.tensor(Z,X)

# ZY = tf.constant(kron(Z, Y), dtype=C_DTYPE)
# real = torch.tensor([[1, 0],[0,-1]], dtype=torch.float64)
# imag = torch.tensor([[0, 0],[0,0]], dtype=torch.float64)
# z = torch.complex(real, imag)
# real = torch.tensor([[0, -0],[0,0]], dtype=torch.float64)
# imag = torch.tensor([[0, -1],[1,0]], dtype=torch.float64)
# y = torch.complex(real, imag)

ZY = torch.tensor(Z,Y)

def PhaseShift(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        torch.Tensor[complex]: diagonal part of the phase shift matrix
    """
    phi = torch.as_tensor(numpy.array(phi))
    phi = phi.type(C_DTYPE)
    return tf.as_tensor([1.0, torch.exp(1j * phi)])


def ControlledPhaseShift(phi):
    r"""Two-qubit controlled phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        torch.as_tensor[complex]: diagonal part of the controlled phase shift matrix
    """

    phi = torch.as_tensor(numpy.array(phi))
    phi = phi.type(C_DTYPE)
    return torch.as_tensor([1.0,1.0,1.0,torch.exp(1j*phi2)])


def RX(theta):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        torch.as_tensor[complex]: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    theta = torch.as_tensor(numpy.array(theta))
    theta = theta.type(C_DTYPE)
    return torch.cos(theta / 2) * I + 1j * torch.sin(-theta / 2) * X


def RY(theta):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle

    Returns:
        torch.as_tensor[complex]: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    theta = torch.as_tensor(numpy.array(theta))
    theta = theta.type(C_DTYPE)
    return torch.cos(theta / 2) * I + 1j * torch.sin(-theta / 2) * Y


def RZ(theta):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle

    Returns:
        torch.as_tensor[complex]: the diagonal part of the rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    theta = torch.as_tensor(numpy.array(theta))
    theta = theta.type(C_DTYPE)
    p = torch.exp(-0.5j * theta)
    return torch.as_tensor([p, torch.conj(p)])

#  TODO: need to find @ meaning and conversion to PyTorch

# def Rot(a, b, c):
#     r"""Arbitrary one-qubit rotation using three Euler angles.
#
#     Args:
#         a,b,c (float): rotation angles
#
#     Returns:
#         torch.as_tensor[complex]: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
#     """
#     return tf.linalg.diag(RZ(c)) @ RY(b) @ tf.linalg.diag(RZ(a))


def MultiRZ(theta, n):
    r"""Arbitrary multi Z rotation.

    Args:
        theta (float): rotation angle
        n (int): number of wires the rotation acts on

    Returns:
        torch.as_tensor[complex]: diagonal part of the MultiRZ matrix
    """
    theta = torch.as_tensor(numpy.array(theta))
    theta = theta.type(C_DTYPE)
    multi_Z_rot_eigs = torch.exp(-1j * theta / 2 * pauli_eigs(n))
    return torch.as_tensor(multi_Z_rot_eigs)


def CRX(theta):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle

    Returns:
        torch.as_tensor[complex]: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    theta = torch.as_tensor(numpy.array(theta))
    theta = theta.type(C_DTYPE)
    return (
        torch.cos(theta / 4) ** 2 * II
        - 1j * torch.sin(theta / 2) / 2 * IX
        + torch.sin(theta / 4) ** 2 * ZI
        + 1j * torch.sin(theta / 2) / 2 * ZX
    )


def CRY(theta):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
    Returns:
        torch.as_tensor[complex]: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    theta = torch.as_tensor(numpy.array(theta))
    theta = theta.type(C_DTYPE)
    return (
        torch.cos(theta / 4) ** 2 * II
        - 1j * torch.sin(theta / 2) / 2 * IY
        + torch.sin(theta / 4) ** 2 * ZI
        + 1j * torch.sin(theta / 2) / 2 * ZY
    )


def CRZ(theta):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
    Returns:
        torch.as_tensor[complex]: diagonal part of the 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    theta = torch.as_tensor(numpy.array(theta))
    theta = theta.type(C_DTYPE)
    p = torch.exp(-0.5j * theta)
    return torch.as_tensor([1.0, 1.0, p, torch.conj(p)])

# TODO:  Need to find Operation for @ in PyTorch
# def CRot(a, b, c):
#     r"""Arbitrary two-qubit controlled rotation using three Euler angles.
#
#     Args:
#         a,b,c (float): rotation angles
#     Returns:
#         torch.as_tensor[complex]: unitary 4x4 rotation matrix
#         :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
#     """
#     return tf.linalg.diag(CRZ(c)) @ (CRY(b) @ tf.linalg.diag(CRZ(a)))


def SingleExcitation(phi):
    r"""Single excitation rotation.

    Args:
        phi (float): rotation angle

    Returns:
        torch.as_tensor[complex]: Single excitation rotation matrix
    """
    phi = torch.as_tensor(numpy.array(theta))
    phi = theta.type(C_DTYPE)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    return torch.as_tensor([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def SingleExcitationPlus(phi):
    r"""Single excitation rotation with positive phase-shift outside the rotation subspace.

    Args:
        phi (float): rotation angle

    Returns:
        torch.as_tensor[complex]: Single excitation rotation matrix with positive phase-shift
    """
    phi = torch.as_tensor(numpy.array(theta))
    phi = theta.type(C_DTYPE)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(1j * phi / 2)
    return tf.as_tensor([[e, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, e]])


def SingleExcitationMinus(phi):
    r"""Single excitation rotation with negative phase-shift outside the rotation subspace.

    Args:
        phi (float): rotation angle

    Returns:
        torch.as_tensor[complex]: Single excitation rotation matrix with negative phase-shift
    """
    phi = torch.as_tensor(numpy.array(theta))
    phi = theta.type(C_DTYPE)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(-1j * phi / 2)
    return torch.as_tensor([[e, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, e]])


def DoubleExcitation(phi):
    r"""Double excitation rotation.

    Args:
        phi (float): rotation angle
    Returns:
        torch.as_tensor[complex]: Double excitation rotation matrix
    """
    phi = torch.as_tensor(numpy.array(theta))
    phi = theta.type(C_DTYPE)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
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
    return torch.as_tensor(U)


def DoubleExcitationPlus(phi):
    r"""Double excitation rotation with positive phase-shift.

    Args:
        phi (float): rotation angle
    Returns:
        torch.as_tensor[complex]: rotation matrix
    """
    phi = torch.as_tensor(numpy.array(theta))
    phi = theta.type(C_DTYPE)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(1j * phi / 2)
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
    return torch.as_tensor(U)


def DoubleExcitationMinus(phi):
    r"""Double excitation rotation with negative phase-shift.

    Args:
        phi (float): rotation angle
    Returns:
        torch.as_tensor[complex]: rotation matrix
    """
    phi = torch.as_tensor(numpy.array(theta))
    phi = theta.type(C_DTYPE)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(-1j * phi / 2)
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
    return torch.as_tensor(U)

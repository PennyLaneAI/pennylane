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
import torch
import numpy
from numpy import kron
from pennylane.utils import pauli_eigs

C_DTYPE = torch.complex128
R_DTYPE = torch.float64

I = torch.complex(torch.tensor([[1, 0],[0,1]], dtype=R_DTYPE),
                    torch.tensor([[0, 0],[0,0]], dtype=R_DTYPE))

X = torch.complex(torch.tensor([[0, 1],[1,0]], dtype=R_DTYPE),
                    torch.tensor([[0, 0],[0,0]], dtype=R_DTYPE))

Y = torch.complex(torch.tensor([[0, -0],[0,0]], dtype=R_DTYPE),
                   torch.tensor([[0, -1],[1,0]], dtype=R_DTYPE))

Z = torch.complex(torch.tensor([[1, 0],[0,-1]], dtype=R_DTYPE),
                    torch.tensor([[0, 0],[0,0]], dtype=R_DTYPE))
II = torch.eye(4)
II = II.type(C_DTYPE)
ZZ = torch.tensor(kron(Z,Z))
IX = torch.tensor(kron(I,X))
IY = torch.tensor(kron(I,Y))
IZ = torch.tensor(kron(I,Z))
ZI = torch.tensor(kron(Z,I))
ZX = torch.tensor(kron(Z,X))
ZY = torch.tensor(kron(Z,Y))

def PhaseShift(phi):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        torch.Tensor[complex]: diagonal part of the phase shift matrix
    """
    phi = torch.as_tensor(numpy.array(phi))
    phi = phi.type(C_DTYPE)
    return torch.as_tensor([1.0, torch.exp(1j * phi)])


def ControlledPhaseShift(phi):
    r"""Two-qubit controlled phase shift.

    Args:
        phi (float): phase shift angle

    Returns:
        torch.as_tensor[complex]: diagonal part of the controlled phase shift matrix
    """
    phi = torch.as_tensor(numpy.array(phi))
    phi = phi.type(C_DTYPE)
    return torch.as_tensor([1.0,1.0,1.0,torch.exp(1j*phi)])


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


def Rot(a, b, c):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles

    Returns:
        torch.as_tensor[complex]: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return torch.diag(RZ(c)) @ RY(b) @ torch.diag(RZ(a))


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

def CRot(a, b, c):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
    Returns:
        torch.as_tensor[complex]: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return torch.diag(CRZ(c)) @ (CRY(b) @ torch.diag(CRZ(a)))

def SingleExcitation(phi):
    r"""Single excitation rotation.

    Args:
        phi (float): rotation angle

    Returns:
        torch.as_tensor[complex]: Single excitation rotation matrix
    """
    phi = torch.as_tensor(numpy.array(phi))
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
    phi = torch.as_tensor(numpy.array(phi))
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
    phi = torch.as_tensor(numpy.array(phi))
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
    phi = torch.as_tensor(numpy.array(phi))
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
    phi = torch.as_tensor(numpy.array(phi))
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
    phi = torch.as_tensor(numpy.array(phi))
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

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
Utility functions and numerical implementations of quantum operations PyTorch device.
"""

import torch
import numpy as np
from pennylane.utils import pauli_eigs


C_DTYPE = torch.complex128
R_DTYPE = torch.float64


# Instantiating on the device (rather than moving) is approx. 100x faster
def op_matrix(elements):
    r"""Decorator to instantiate a tensor on a device.

    Args:
        element : torch.Tensor

    Returns:
        lambda dev : torch.Tensor elements instantiated on torch device dev
    """
    return lambda dev: torch.as_tensor(elements, dtype=C_DTYPE, device=dev)


I_array = np.array([[1, 0], [0, 1]])
X_array = np.array([[0, 1], [1, 0]])
Y_array = np.array([[0j, -1j], [1j, 0j]])
Z_array = np.array([[1, 0], [0, -1]])


I = op_matrix(I_array)
X = op_matrix(X_array)
Y = op_matrix(Y_array)
Z = op_matrix(Z_array)

II = op_matrix(np.eye(4))
ZZ = op_matrix(np.kron(Z_array, Z_array))
XX = op_matrix(np.kron(X_array, X_array))
YY = op_matrix(np.kron(Y_array, Y_array))


IX = op_matrix(np.kron(I_array, X_array))
IY = op_matrix(np.kron(I_array, Y_array))
IZ = op_matrix(np.kron(I_array, Z_array))

ZI = op_matrix(np.kron(Z_array, I_array))
ZX = op_matrix(np.kron(Z_array, X_array))
ZY = op_matrix(np.kron(Z_array, Y_array))

A = op_matrix(np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]))
B = op_matrix(np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]))
C = op_matrix(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]))

USin = op_matrix(
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
)

UCos = op_matrix(
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
)

I4_array = np.eye(16)
I4_array[3, 3] = 0
I4_array[-4, -4] = 0
I4 = op_matrix(I4_array)


def PhaseShift(phi, device=None):
    r"""One-qubit phase shift.

    Args:
        phi (float): phase shift angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: diagonal part of the phase shift matrix
    """
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    p = torch.exp(1j * phi)
    return torch.tensor([1, 0], dtype=torch.complex128, device=device) + p * torch.tensor(
        [0, 1], dtype=torch.complex128, device=device
    )


def ControlledPhaseShift(phi, device=None):
    r"""Two-qubit controlled phase shift.

    Args:
        phi (float): phase shift angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: diagonal part of the controlled phase shift matrix
    """
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    p = torch.exp(1j * phi)
    return torch.tensor([1, 1, 1, 0], dtype=torch.complex128) + p * torch.tensor(
        [0, 0, 0, 1], dtype=torch.complex128
    )


def RX(theta, device=None):
    r"""One-qubit rotation about the x axis.

    Args:
        theta (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: unitary 2x2 rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    return torch.cos(theta / 2) * I(device) + 1j * torch.sin(-theta / 2) * X(device)


def RY(theta, device=None):
    r"""One-qubit rotation about the y axis.

    Args:
        theta (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: unitary 2x2 rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)

    return torch.cos(theta / 2) * I(device) + 1j * torch.sin(-theta / 2) * Y(device)


def RZ(theta, device=None):
    r"""One-qubit rotation about the z axis.

    Args:
        theta (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: the diagonal part of the rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    p = torch.exp(-0.5j * theta)
    return p * torch.tensor([1, 0], dtype=torch.complex128, device=device) + torch.conj(
        p
    ) * torch.tensor([0, 1], dtype=torch.complex128, device=device)


def MultiRZ(theta, n, device=None):
    r"""Arbitrary multi Z rotation.

    Args:
        theta (float): rotation angle
        n (int): number of wires the rotation acts on
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: diagonal part of the MultiRZ matrix
    """

    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    eigs = torch.as_tensor(pauli_eigs(n), dtype=C_DTYPE, device=device)
    return torch.exp(-1j * theta / 2 * eigs)


def Rot(a, b, c, device=None):
    r"""Arbitrary one-qubit rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: unitary 2x2 rotation matrix ``rz(c) @ ry(b) @ rz(a)``
    """
    return torch.diag(RZ(c, device)) @ RY(b, device) @ torch.diag(RZ(a, device))


def CRX(theta, device=None):
    r"""Two-qubit controlled rotation about the x axis.

    Args:
        theta (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_x(\theta)`
    """
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    return (
        torch.cos(theta / 4) ** 2 * II(device)
        - 1j * torch.sin(theta / 2) / 2 * IX(device)
        + torch.sin(theta / 4) ** 2 * ZI(device)
        + 1j * torch.sin(theta / 2) / 2 * ZX(device)
    )


def CRY(theta, device):
    r"""Two-qubit controlled rotation about the y axis.

    Args:
        theta (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: unitary 4x4 rotation matrix :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_y(\theta)`
    """
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    return (
        torch.cos(theta / 4) ** 2 * II(device)
        - 1j * torch.sin(theta / 2) / 2 * IY(device)
        + torch.sin(theta / 4) ** 2 * ZI(device)
        + 1j * torch.sin(theta / 2) / 2 * ZY(device)
    )


def CRZ(theta, device):
    r"""Two-qubit controlled rotation about the z axis.

    Args:
        theta (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: diagonal part of the 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R_z(\theta)`
    """
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    return torch.cat([torch.as_tensor([1.0, 1.0], device=device), RZ(theta, device)], dim=0)


def CRot(a, b, c, device):
    r"""Arbitrary two-qubit controlled rotation using three Euler angles.

    Args:
        a,b,c (float): rotation angles
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: unitary 4x4 rotation matrix
        :math:`|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes R(a,b,c)`
    """
    return torch.diag(CRZ(c, device)) @ CRY(b, device) @ torch.diag(CRZ(a, device))


def IsingXX(phi, device):
    r"""Ising XX coupling gate

    .. math:: XX(\phi) = \begin{bmatrix}
        \cos(\phi / 2) & 0 & 0 & -i \sin(\phi / 2) \\
        0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
        0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
        -i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    Args:
        phi (float): rotation angle :math:`\phi`
        device: torch device on which the computation is made 'cpu' or 'cuda'
    Returns:
        torch.Tensor[complex]:: unitary 4x4 rotation matrix
    """
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    return torch.cos(phi / 2) * II(device) - 1j * torch.sin(phi / 2) * XX(device)


def IsingYY(phi, device):
    r"""Ising YY coupling gate

    .. math:: YY(\phi) = \begin{bmatrix}
        \cos(\phi / 2) & 0 & 0 & i \sin(\phi / 2) \\
        0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
        0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
        i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
        \end{bmatrix}.

    Args:
        phi (float): rotation angle :math:`\phi`
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]:: unitary 4x4 rotation matrix
    """
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    return torch.cos(phi / 2) * II(device) - 1j * torch.sin(phi / 2) * YY(device)


def IsingZZ(phi, device):
    r"""Ising ZZ coupling gate

    .. math:: ZZ(\phi) = \begin{bmatrix}
        e^{-i \phi / 2} & 0 & 0 & 0 \\
        0 & e^{i \phi / 2} & 0 & 0 \\
        0 & 0 & e^{i \phi / 2} & 0 \\
        0 & 0 & 0 & e^{-i \phi / 2}
        \end{bmatrix}.

    Args:
        phi (float): rotation :math:`\phi`
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]:: unitary 4x4 rotation matrix
    """
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    e_m = torch.exp(-1j * phi / 2)
    e = torch.exp(1j * phi / 2)
    return torch.as_tensor([[e_m, 0, 0, 0], [0, e, 0, 0], [0, 0, e, 0], [0, 0, 0, e_m]])


def SingleExcitation(phi, device):
    r"""Single excitation rotation.

    Args:
        phi (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: Single excitation rotation matrix
    """
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)

    return c * A(device) + s * B(device) + C(device)


def SingleExcitationPlus(phi, device):
    r"""Single excitation rotation with positive phase-shift outside the rotation subspace.

    Args:
        phi (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: Single excitation rotation matrix with positive phase-shift
    """
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(1j * phi / 2)

    return c * A(device) + s * B(device) + e * C(device)


def SingleExcitationMinus(phi, device):
    r"""Single excitation rotation with negative phase-shift outside the rotation subspace.

    Args:
        phi (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: Single excitation rotation matrix with negative phase-shift
    """
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(-1j * phi / 2)

    return c * A(device) + s * B(device) + e * C(device)


def DoubleExcitation(phi, device):
    r"""Double excitation rotation.

    Args:
        phi (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: Double excitation rotation matrix
    """

    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)

    return I4(device) + c * UCos(device) + s * USin(device)


def DoubleExcitationPlus(phi, device):
    r"""Double excitation rotation with positive phase-shift.

    Args:
        phi (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: rotation matrix
    """
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(1j * phi / 2)

    return e * I4(device) + c * UCos(device) + s * USin(device)


def DoubleExcitationMinus(phi, device):
    r"""Double excitation rotation with negative phase-shift.

    Args:
        phi (float): rotation angle
        device: torch device on which the computation is made 'cpu' or 'cuda'

    Returns:
        torch.Tensor[complex]: rotation matrix
    """
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    c = torch.cos(phi / 2)
    s = torch.sin(phi / 2)
    e = torch.exp(-1j * phi / 2)

    return e * I4(device) + c * UCos(device) + s * USin(device)

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

# TODO docstrings

import torch
import numpy as np

C_DTYPE = torch.complex128
R_DTYPE = torch.float64


# TODO replace by torch.diag once complex dtypes are supported
def diag(input):
    matrix = torch.eye(len(input), dtype=input.dtype, device=input.device)
    return matrix * input

# Instantiating on the device (rather than moving) is approx. 100x faster
def op_matrix(elements):
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

IX = op_matrix(np.kron(I_array, X_array))
IY = op_matrix(np.kron(I_array, Y_array))
IZ = op_matrix(np.kron(I_array, Z_array))

ZI = op_matrix(np.kron(Z_array, I_array))
ZX = op_matrix(np.kron(Z_array, X_array))
ZY = op_matrix(np.kron(Z_array, Y_array))

def PhaseShift(phi, device=None):
    phi = torch.as_tensor(phi, dtype=C_DTYPE, device=device)
    return torch.as_tensor([1.0, torch.exp(1j * phi)])


def RX(theta, device=None):
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    return torch.cos(theta / 2) * I(device) + 1j * torch.sin(-theta / 2) * X(device)


def RY(theta, device=None):
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    return torch.cos(theta / 2) * I(device) + 1j * torch.sin(-theta / 2) * Y(device)


def RZ(theta, device=None):
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    p = torch.exp(-0.5j * theta)
    return torch.as_tensor([p, torch.conj(p)], device=device)


def Rot(a, b, c, device=None):
    return diag(RZ(c, device)) @ RY(b, device) @ diag(RZ(a, device))


def CRX(theta, device=None):
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    return  (
        torch.cos(theta / 4) ** 2 * II(device)
        - 1j * torch.sin(theta / 2) / 2 * IX(device)
        + torch.sin(theta / 4) ** 2 * ZI(device)
        + 1j * torch.sin(theta / 2) / 2 * ZX(device)
    )


def CRY(theta, device):
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device) 
    return (
        torch.cos(theta / 4) ** 2 * II(device)
        - 1j * torch.sin(theta / 2) / 2 * IY(device)
        + torch.sin(theta / 4) ** 2 * ZI(device)
        + 1j * torch.sin(theta / 2) / 2 * ZY(device)
    )


def CRZ(theta, device):
    theta = torch.as_tensor(theta, dtype=C_DTYPE, device=device)
    p = torch.exp(-0.5j * theta)
    return torch.as_tensor([1.0, 1.0, p, torch.conj(p)], device=device)


def CRot(a, b, c, device):
    return diag(CRZ(c, device)) @ CRY(b, device) @ diag(CRZ(a, device))
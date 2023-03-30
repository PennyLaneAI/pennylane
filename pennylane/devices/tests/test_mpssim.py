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
"""Tests that the different measurement types work correctly on a device."""
# pylint: disable=no-self-use,pointless-statement, no-member
import pytest
import pennylane.numpy as np
import pennylane as qml
from tenpy.linalg.svd_robust import svd
from pennylane.devices.experimental.custom_device_3_numpydev.python_mps import SimpleMPS, construct_MPO, contract_MPO_MPS

def vec_to_mps(psi):
    """Returns a right-canonical mps form of an input state vector psi"""
    L = int(np.log(len(psi))/np.log(2))
    Ss, Bs = [], []
    psi = psi.reshape((2**(L-1), 2))  # (..sL-2 sL-1) sL 1
    A, SL, BL = svd(psi, full_matrices=False)      # (..sL-2 sL-1) aL | aL sL
    BL = BL.reshape((2, 2, 1))
    Bs.append(BL)
    Ss.append(Ss)
    A = A @ np.diag(SL)
    A = A.reshape((2**(L-2), -1))    # (..sL-2 (sL-1 aL)
    for i in range(1, L):
        A, Si, Bi = svd(A, full_matrices=False)       # s0 a0 | a0 (s1 a1)
        Bi = Bi.reshape((Bi.shape[0], 2, Bi.shape[1]//2))
        Bs.append(Bi)
        Ss.append(Si)
        if i != L-1:
            A = A @ np.diag(Si)
            A = A.reshape((2**(L-i-2), -1))    # 1 (s0 a0)

    return SimpleMPS(Bs[::-1], Ss[::-1])

def mps_to_vec(psi):
    """Returns a state vector given an MPS
    (of any form, does not need to be right- or left canonical)
    """
    Bs = psi.Bs
    M = Bs[-1]
    for i in range(1, 3):
        M = np.tensordot(Bs[-i-1], M, axes=[[-1], [0]])
    return M.flatten()


def test_MPO_MPS_contraction_yields_same_result():
    """Test that MPO-MPS contraction yields same result as Matrix-vector product"""
    i = 0 ; j = 2
    ops = [qml.PauliX(i) @ qml.PauliX(j), qml.PauliY(i) @ qml.PauliY(j), qml.PauliZ(i) @ qml.PauliZ(j)]
    coeffs = np.linspace(1, 3, len(ops))
    Herm = qml.op_sum(*[qml.s_prod(x, y) for x,y in zip(coeffs, ops)])
    op = qml.exp(-1j*Herm)

    basis = np.eye(2**(j-i+1))
    psi = sum([(i+1 + 1j * i) * basis[i] for i in range(len(basis))])
    psi_vec = psi / np.linalg.norm(psi)

    psi = vec_to_mps(psi_vec)
    new_psi = contract_MPO_MPS(op, psi, 10, 1e-15)
    new_psi_vec = mps_to_vec(new_psi)
    overlap = np.abs(new_psi_vec.conj().T @ qml.matrix(op, wire_order=range(3)) @ psi_vec)
    assert qml.math.isclose(overlap, 1)
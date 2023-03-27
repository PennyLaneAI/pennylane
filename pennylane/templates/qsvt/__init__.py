# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
A module which implements basic QSVT capabilities
"""
import copy

import pennylane.numpy as np
from pennylane.ops import Identity, PCPhase, BlockEncode, adjoint, s_prod, exp


def qsvt(A, phi_vect, wires, convention=None):
    """Applies the QSVT sequence to A"""
    d = len(phi_vect) - 1
    shape_a = getattr(A, "shape", None)

    dim_tilde, dim = (1, 1) if not shape_a else shape_a

    if convention == "Wx":  # Convert to Wx convention!
        phi_vect = _qsp_to_qsvt(phi_vect)
        global_phase = d % 4

        if global_phase == 1:
            exp(Identity(wires=wires), 1j * (3*np.pi / 2))

        elif global_phase == 2:
            exp(Identity(wires=wires), 1j * np.pi)

        elif global_phase == 3:
            exp(Identity(wires=wires), 1j * np.pi / 2)

    PCPhase(phi_vect[-1], dim, wires=wires)

    if d % 2 == 0:
        for k in range(1, (d // 2) + 1)[::-1]:
            BlockEncode(A, wires=wires)  # U
            PCPhase(phi_vect[2*k - 1], dim_tilde, wires=wires)  # Pi_tilde
            adjoint(BlockEncode(A, wires=wires))  # U^dag
            PCPhase(phi_vect[2*k - 2], dim, wires=wires)  # Pi

    else:
        for k in range(1, ((d - 1) // 2) + 1)[::-1]:
            BlockEncode(A, wires=wires)  # U
            PCPhase(phi_vect[2*k], dim_tilde, wires=wires)  # Pi_tilde
            adjoint(BlockEncode(A, wires=wires))  # U^dag
            PCPhase(phi_vect[2*k - 1], dim, wires=wires)  # Pi

        BlockEncode(A, wires=wires)  # U
        PCPhase(phi_vect[0], dim_tilde, wires=wires)  # Pi_tilde

    return


def _qsp_to_qsvt(phi_vect):
    new_phis = copy.copy(phi_vect)
    new_phis[0] += 3 * np.pi / 4
    new_phis[-1] -= np.pi / 4

    for i, phi in enumerate(new_phis[1:-1]):
        new_phis[i + 1] = phi + np.pi / 2
    return new_phis

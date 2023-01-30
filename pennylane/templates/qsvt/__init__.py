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
"""
A module for performing QSVT with PennyLane.
"""
import numpy as np
from pennylane.ops.op_math import adjoint
from .qsvt_ops import BlockEncoding, PiControlledPhase


def _qsvt_protocol(A, phi_vect, wires):
    """Executes the operations to perform the qsvt protocol"""
    d = len(phi_vect)
    c, r = A.shape
    u_size = c + r

    lst_operations = []

    if d % 2 == 0:
        for i in range(d//2):
            lst_operations.append(PiControlledPhase(phi_vect[2*i], (c, u_size), wires=wires))
            lst_operations.append(adjoint(BlockEncoding(A, wires=wires)))
            lst_operations.append(PiControlledPhase(phi_vect[2*i + 1], (r, u_size), wires=wires))
            lst_operations.append(BlockEncoding(A, wires=wires))
    else:
        lst_operations.append(PiControlledPhase(phi_vect[0], (r, u_size), wires=wires))
        lst_operations.append(BlockEncoding(A, wires=wires))

        for i in range(d-1 // 2):
            lst_operations.append(PiControlledPhase(phi_vect[2*i + 1], (c, u_size), wires=wires))
            lst_operations.append(adjoint(BlockEncoding(A, wires=wires)))
            lst_operations.append(PiControlledPhase(phi_vect[2*i + 2], (r, u_size), wires=wires))
            lst_operations.append(BlockEncoding(A, wires=wires))

    return lst_operations

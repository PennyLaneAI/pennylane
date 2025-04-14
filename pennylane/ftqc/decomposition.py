# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Contains functions to convert a PennyLane tape to the textbook MBQC formalism
"""

import pennylane as qml
from pennylane.math import isclose
from pennylane.ops.op_math.decompositions.single_qubit_unitary import _get_xzx_angles

from .operations import RotXZX

mbqc_gate_set = {
    qml.CNOT,
    qml.H,
    qml.S,
    qml.RZ,
    RotXZX,
    qml.X,
    qml.Y,
    qml.Z,
    qml.I,
    qml.GlobalPhase,
}


@qml.register_resources({RotXZX: 1, qml.GlobalPhase: 1})
def _rot_to_xzx(phi, theta, omega, wires, **__):
    mat = qml.Rot.compute_matrix(phi, theta, omega)
    phi, theta, lam, gamma = _get_xzx_angles(mat)

    RotXZX(lam, theta, phi, wires)
    if not isclose(gamma, 0):
        qml.GlobalPhase(-gamma)


@qml.transform
def convert_to_mbqc_gateset(tape):
    """Converts a circuit expressed in arbitrary gates to the limited gate set that we can
    convert to the textbook MBQC formalism"""
    tapes, fn = qml.transforms.decompose(
        tape, gate_set=mbqc_gate_set, alt_decomps={qml.Rot: [_rot_to_xzx]}
    )
    return tapes, fn

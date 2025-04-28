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

from pennylane import math
from pennylane.decomposition import enabled_graph, register_resources
from pennylane.ops import CNOT, RZ, GlobalPhase, H, Identity, Rot, S, X, Y, Z
from pennylane.transforms import decompose, transform

from .operations import RotXZX

mbqc_gate_set = {CNOT, H, S, RotXZX, RZ, X, Y, Z, Identity, GlobalPhase}


@register_resources({RotXZX: 1})
def _rot_to_xzx(phi, theta, omega, wires, **__):
    mat = Rot.compute_matrix(phi, theta, omega)
    lam, theta, phi = math.decomposition.xzx_rotation_angles(mat)
    RotXZX(lam, theta, phi, wires)


@transform
def convert_to_mbqc_gateset(tape):
    """Converts a circuit expressed in arbitrary gates to the limited gate set that we can
    convert to the textbook MBQC formalism"""
    if not enabled_graph():
        raise RuntimeError(
            "Using `convert_to_mbqc_gateset` requires the graph-based decomposition"
            " method. This can be toggled by calling `qml.decomposition.enable_graph()`"
        )
    tapes, fn = decompose(tape, gate_set=mbqc_gate_set, alt_decomps={Rot: [_rot_to_xzx]})
    return tapes, fn

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
"""
This file contains a number of attributes that may be held by operators,
and lists all operators satisfying those criteria.
"""

"""Operators for which composing multiple copies of the operation results in an
addition (or alternative accumulation) of parameters.

For example, ``qml.RZ`` is a composable rotation. Applying ``qml.RZ(0.1,
wires=0)`` followed by ``qml.RZ(0.2, wires=0)`` is equivalent to performing
a single rotation ``qml.RZ(0.3, wires=0)``.

"""
is_composable_rotation = [
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "CRX",
    "CRY",
    "CRZ",
    "ControlledPhaseShift",
    "Rot",
]


"""Operations that are their own inverses.
"""
is_self_inverse = ["Hadamard", "PauliX", "PauliY", "PauliZ", "CNOT", "CZ", "CY", "SWAP", "Toffoli"]


"""Operations that are the same if you exchange the order of wires.

For example, ``qml.CZ(wires=[0, 1])`` has the same effect as ``qml.CZ(wires=[1,
0])`` due to symmetry of the operation.
"""
is_symmetric_over_all_wires = [
    "CZ",
    "SWAP",
]


"""Controlled operations that are the same if you exchange the order of all but
the last (target) wire.

For example, ``qml.Toffoli(wires=[0, 1, 2])`` has the same effect as
``qml.Toffoli(wires=[1, 0, 2])``, but neither are the same as
``qml.Toffoli(wires=[0, 2, 1])``.
"""
is_symmetric_over_control_wires = ["Toffoli"]

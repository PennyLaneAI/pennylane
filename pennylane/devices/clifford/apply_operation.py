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
"""Functions to apply an operation to a state vector."""
# pylint: disable=unused-argument

from functools import singledispatch
import pennylane as qml

from pennylane import math

_operations = {
    "Identity": "I",
    "Snapshot": None,
    "BasisState": None,
    "StatePrep": None,
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "Hadamard": "H",
    "S": "S",
    "SX": "SX",
    "CNOT": "CNOT",
    "SWAP": "SWAP",
    "ISWAP": "ISWAP",
    "CY": "CY",
    "CZ": "CZ",
    "GlobalPhase": None,
}


@apply_operation.register
def apply_global_phase(op: qml.GlobalPhase, state, is_state_batched: bool = False, debugger=None):
    """Applies a :class:`~.GlobalPhase` operation by multiplying the state by ``exp(1j * op.data[0])``"""
    return qml.math.exp(-1j * qml.math.cast(op.data[0], complex)) * state

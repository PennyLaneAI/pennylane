# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains definitions of common gate sets."""

from types import MappingProxyType

from pennylane import ops

_CLIFFORD_T_ORIGINAL = frozenset(
    {
        ops.X,
        ops.Y,
        ops.Z,
        ops.H,
        ops.S,
        ops.SX,
        ops.T,
        ops.CNOT,
        ops.CY,
        ops.CZ,
        ops.SWAP,
        ops.ISWAP,
    }
)

_ADJOINT_CLIFFORD_T = frozenset({f"Adjoint({op.__name__})" for op in _CLIFFORD_T_ORIGINAL})

_CLIFFORD_T = _CLIFFORD_T_ORIGINAL | _ADJOINT_CLIFFORD_T

IDENTITY = frozenset({ops.Identity, ops.GlobalPhase})

CLIFFORD_T = _CLIFFORD_T | IDENTITY

_CLIFFORD_T_RZ = {op: 1.0 for op in _CLIFFORD_T} | {op: 0.0 for op in IDENTITY} | {ops.RZ: 100}

CLIFFORDD_T_PLUS_RZ = MappingProxyType(_CLIFFORD_T_RZ)  # readonly view

ROTATIONS_PLUS_CNOT = frozenset({ops.RX, ops.RY, ops.RZ, ops.CNOT}) | IDENTITY

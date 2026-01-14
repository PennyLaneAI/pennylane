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

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pennylane import ops

from .utils import translate_op_alias


class GateSet:
    """Stores the target gate set of a decomposition pass."""

    def __init__(self, gate_set: Iterable | Mapping, name=""):
        if not isinstance(gate_set, Mapping):
            gate_set = {op: 1.0 for op in gate_set}
        if any(v < 0 for v in gate_set.values()):
            raise ValueError("Negative weights are not supported in the gate_set.")
        self.name = name
        self._gate_set = {_to_name(op): weight for op, weight in gate_set.items()}

    def __getitem__(self, key, /):
        return self._gate_set[key]

    def __setitem__(self, key, value, /) -> None:
        raise TypeError("The GateSet is immutable.")

    def __contains__(self, op) -> bool:
        return op in self._gate_set

    def __iter__(self):
        return iter(self._gate_set)

    def __or__(self, other: GateSet, /) -> GateSet:
        return GateSet(self._gate_set | other._gate_set)

    def __repr__(self) -> str:
        return self.name if self.name else str(set(self._gate_set.keys()))


def _to_name(op):
    if isinstance(op, type):
        return op.__name__
    assert isinstance(op, str)
    return translate_op_alias(op)


_CLIFFORD_T_ORIGINAL = GateSet(
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

_ADJOINT_CLIFFORD_T = GateSet({f"Adjoint({op})" for op in _CLIFFORD_T_ORIGINAL})

_CLIFFORD_T = _CLIFFORD_T_ORIGINAL | _ADJOINT_CLIFFORD_T

_MID_MEASURE = GateSet({ops.MidMeasure})

IDENTITY = GateSet({ops.Identity: 0.0, ops.GlobalPhase: 0.0}, name="Identity")

CLIFFORD_T = _CLIFFORD_T | IDENTITY | _MID_MEASURE
CLIFFORD_T.name = "Clifford+T"

CLIFFORDD_T_PLUS_RZ = CLIFFORD_T | GateSet({ops.RZ: 100})
CLIFFORDD_T_PLUS_RZ.name = "Clifford+T+RZ"

ROTATIONS_PLUS_CNOT = GateSet({ops.RX, ops.RY, ops.RZ, ops.CNOT}) | IDENTITY | _MID_MEASURE
ROTATIONS_PLUS_CNOT.name = "Rotations+CNOT"

_MBQC_GATES = GateSet({ops.CNOT, ops.H, ops.S, "RotXZX", ops.RZ, ops.X, ops.Y, ops.Z})

MBQC_GATES = _MBQC_GATES | IDENTITY | _MID_MEASURE
MBQC_GATES.name = "MBQC"

ALL_QUBIT_OPS = GateSet(ops.qubit.__all__, name="All Qubit Gates")

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

"""This module contains definitions of the GateSet data structure."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set

from pennylane.operation import Operator

from .utils import translate_op_alias


class GateSet(Mapping):
    """Stores the target gate set of a decomposition pass."""

    def __init__(self, gate_set: Iterable | Mapping, name=""):
        if not isinstance(gate_set, Mapping):
            gate_set = {op: 1.0 for op in gate_set}
        if any(v < 0 for v in gate_set.values()):
            raise ValueError("Negative weights are not supported in the gate_set.")
        self.name = name
        self._gate_set = {_to_name(op): weight for op, weight in gate_set.items()}

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, GateSet):
            return False
        return self._gate_set == value._gate_set

    def __ne__(self, value: object, /) -> bool:
        return not self.__eq__(value)

    def __len__(self) -> int:
        return len(self._gate_set)

    def __getitem__(self, key, /):
        return self._gate_set[_to_name(key)]

    def __setitem__(self, key, value, /) -> None:
        raise TypeError("The GateSet is immutable.")

    def __contains__(self, op) -> bool:
        return _to_name(op) in self._gate_set

    def __iter__(self):
        return iter(self._gate_set)

    def __or__(self, other: Set | Mapping, /) -> GateSet:
        if not isinstance(other, (Mapping, Set)):
            return NotImplemented
        if not isinstance(other, GateSet):
            other = GateSet(other)
        return GateSet(self._gate_set | other._gate_set)

    def __sub__(self, other: Set | Mapping) -> GateSet:
        if not isinstance(other, (Mapping, Set)):
            return NotImplemented
        if not isinstance(other, GateSet):
            other = GateSet(other)
        return GateSet({k: v for k, v in self._gate_set.items() if k not in other})

    def keys(self):
        return self._gate_set.keys()

    def values(self):
        return self._gate_set.values()

    def items(self):
        return self._gate_set.items()

    def get(self, key, default=None):
        return self._gate_set.get(_to_name(key), default)

    def __repr__(self) -> str:
        if self.name:
            return self.name
        inner_str = ", ".join(list(self))
        return f"{{{inner_str}}}"


def _to_name(op):
    if isinstance(op, type):
        op = op.__name__
    if isinstance(op, Operator):
        op = op.name
    assert isinstance(op, str)
    return translate_op_alias(op)

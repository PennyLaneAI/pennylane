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

from .utils import to_name


class GateSet(Mapping):
    """Stores the target gate set of a decomposition pass.

    Args:
        gate_set (Iterable | Mapping): the contents
        name (str): a shorthand to use in the str and repr

    While the ``decompose`` transform can accept any iterable for it's ``gate_set``
    argument, the ``GateSet`` class provides some helpful tools.
    This includes a ``name`` argument for improved inspection and condensed reprs,
    immutability for improved protection when used as a global variable, and conversion
    between class and string based representations of operators.

    We can create a gateset using both :class:`~.Operator` subclasses or strings, and use
    both classes and strings to check inclusion in the gateset

    >>> from pennylane.decomposition import GateSet
    >>> gateset = GateSet({"X", qp.RX, "Adjoint(RX)"})
    >>> gateset
    GateSet({RX, PauliX, Adjoint(RX)})
    >>> qp.X in gateset
    True
    >>> "RX" in gateset
    True

    We can also provide a ``name`` for improved inspection.

    >>> gateset_name = GateSet({qp.RX, qp.RY, qp.RZ}, name="Rotations")
    >>> print(gateset_name)
    Rotations
    >>> qp.decompose(gate_set=gateset_name)
    <decompose(gate_set=Rotations)>

    Gate sets can be combined with ``|``:

    >>> gateset | {qp.RX, qp.RY, qp.RZ}
    GateSet({RX, PauliX, Adjoint(RX), RY, RZ})

    Items can be removed with ``-``:

    >>> gateset - {qp.RX, qml.RY}
    GateSet({PauliX, Adjoint(RX)})
    >>> gateset - qp.RX
    GateSet({PauliX, Adjoint(RX)})
    >>> gateset - "RX"
    GateSet({PauliX, Adjoint(RX)})

    Weights can also be provided for use in calculating costs and choosing optimal decompositions:

    >>> weighted_gateset = GateSet({qp.I: 0, qp.RX: 1, qp.CNOT: 3})

    If not provided, weights default to ``1``:

    >>> print("\n".join(f"{k}={v}" for k, v in gateset.items()))
    RX=1.0
    PauliX=1.0
    Adjoint(RX)=1.0

    """

    def __init__(self, gate_set: Iterable | Mapping, name=""):
        if not isinstance(gate_set, Mapping):
            gate_set = {op: 1.0 for op in gate_set}
        if any(v < 0 for v in gate_set.values()):
            raise ValueError("Negative weights are not supported in the gate_set.")
        self.name = name
        self._gate_set = {to_name(op): weight for op, weight in gate_set.items()}

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, GateSet):
            return False
        return self._gate_set == value._gate_set

    def __ne__(self, value: object, /) -> bool:
        return not self.__eq__(value)

    def __len__(self) -> int:
        return len(self._gate_set)

    def __getitem__(self, key, /):
        return self._gate_set[to_name(key)]

    def __setitem__(self, key, value, /) -> None:
        raise TypeError("The GateSet is immutable.")

    def __contains__(self, op) -> bool:
        return to_name(op) in self._gate_set

    def __iter__(self):
        return iter(self._gate_set)

    def __or__(self, other: Set | Mapping, /) -> GateSet:
        if not isinstance(other, (Mapping, Set)):
            return NotImplemented
        if not isinstance(other, GateSet):
            other = GateSet(other)
        return GateSet(self._gate_set | other._gate_set)

    def __sub__(self, other: Set | Mapping) -> GateSet:
        if (isinstance(other, type) and issubclass(other, Operator)) or isinstance(other, str):
            other = GateSet({other})
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
        return self._gate_set.get(to_name(key), default)

    def __str__(self) -> str:
        if self.name:
            return self.name
        inner_str = ", ".join(str(k) if v == 1 else f"{k}={v}" for k, v in self.items())
        return f"{{{inner_str}}}"

    def __repr__(self) -> str:
        gate_set_str = ", ".join(str(k) if v == 1 else f"{k}={v}" for k, v in self.items())
        name_str = f", name='{self.name}'" if self.name else ""
        return f"GateSet({{{gate_set_str}}}{name_str})"

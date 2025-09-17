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
"""SSAQubitMap class for a bidirectional map between wire labels and SSA qubits."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from xdsl.dialects import builtin
from xdsl.ir import SSAValue
from xdsl.utils.hints import isa

from ..dialects import quantum


@dataclass(frozen=True)
class AbstractWire:
    """An abstract wire."""

    qreg: quantum.QuregSSAValue | None = None
    """The SSA quantum register from which the qubit corresponding to this abstract
    wire has been extracted. This is used to differentiate between qubits allocated
    in static and dynamic registers. For qubits that are allocated dynamically
    without the use of a register, this is ``None``."""

    idx: SSAValue[builtin.I64] | None = None
    """The SSA integer value that corresponds to the index of the qubit in the quantum
    register. This is used for qubits that are allocated in the quantum register. For
    qubits that are dynamically allocated individually, this is ``None``."""

    uuid: UUID = field(default_factory=uuid4, init=False)
    """A universally unique identifier for the abstract wire. Used when qubits are
    dynamically allocated individually, since such qubits are not in a quantum
    register."""

    def __hash__(self) -> int:
        if self.qreg and self.idx:
            return hash(self.qreg) + hash(self.idx)
        return hash(self.uuid)

    def __eq__(self, other) -> bool:
        if not isinstance(other, AbstractWire):
            return False

        if self.qreg and other.qreg and self.idx and other.idx:
            return self.qreg == other.qreg and self.idx == other.idx
        return self.uuid == other.uuid


# The allowed types for the keys and values in the bidirectional map.
_valid_types = (int, AbstractWire, quantum.QubitSSAValue)


@dataclass
class SSAQubitMap:
    """Class to maintain two-way mapping between wire labels and SSA qubits."""

    wires: tuple[int, ...] | None = None
    """Tuple containing all available static wire labels. None if not provided."""

    _map: dict[
        int | AbstractWire | quantum.QubitSSAValue, int | AbstractWire | list[quantum.QubitSSAValue]
    ] = field(default_factory=dict, init=False)
    """Internal map that stores the bidirectional mapping between wire labels and
    SSA qubits. If the key is an integer or ``AbstractWire``, then the value will be
    a list of all SSA qubits that correspond to that wire label. If the key is an
    SSA qubit, then the value will be the wire (integer or ``AbstractWire``) to which
    it corresponds."""

    def _assert_valid_type(self, obj: Any) -> None:
        """Check if the given value is valid. Valid types are ``int``, ``AbstractWire``,
        or ``xdsl.ir.SSAValue`` of type ``quantum.QubitType``. If not, raise an error."""

        # We use ``xdsl.utils.hints.isa`` here instead of ``isinstance`` because that is
        # designed to allow checking the input object againt generic types, which
        # ``isinstance`` cannot handle.
        if not any(isa(obj, hint) for hint in _valid_types):
            raise AssertionError(
                f"{obj} is not a valid key or value for an SSAQubitMap. Valid keys and values "
                f"must be of one of the following types: {_valid_types}."
            )

    def _assert_valid_wire_label(self, wire: int | AbstractWire) -> None:
        """Check if the given wire label is valid. Valid wire labels are either of type
        ``AbstractWire``, or integers that belong to ``self.wires``, if provided."""
        if isinstance(wire, int) and self.wires is not None and wire not in self.wires:
            raise AssertionError(
                f"{wire} is not a valid wire label. Static wire labels must belong "
                f"to {self.wires}."
            )

    def __contains__(self, key: int | AbstractWire | quantum.QubitSSAValue) -> bool:
        """Check if the map contains a wire label or qubit."""
        return key in self._map

    def __getitem__(
        self, key: int | AbstractWire | quantum.QubitSSAValue
    ) -> int | AbstractWire | list[quantum.QubitSSAValue]:
        """Get a value from the wire/qubit maps."""
        return self._map[key]

    def __setitem__(
        self,
        key: int | AbstractWire | quantum.QubitSSAValue,
        value: quantum.QubitSSAValue | int | AbstractWire,
    ) -> None:
        """Update the wire/qubit maps."""
        self._assert_valid_type(key)
        self._assert_valid_type(value)

        if isa(key, quantum.QubitSSAValue):
            if key in self._map:
                raise KeyError("Cannot update qubits that are already in the map.")

            self._assert_valid_wire_label(value)
            self._map[key] = value
            qubits = self._map.setdefault(value, [])
            qubits.append(key)
            return

        self._assert_valid_wire_label(key)
        if value in self._map:
            raise ValueError("Cannot update qubits that are already in the map.")

        qubits = self._map.setdefault(key, [])
        qubits.append(value)
        self._map[value] = key

    def get(
        self, key: int | AbstractWire | quantum.QubitSSAValue, default: Any | None = None
    ) -> int | AbstractWire | list[quantum.QubitSSAValue] | None:
        """Return an item from the map without removing it, if it exists. Else, return
        the provided default value."""
        return self._map.get(key, default)

    def pop(
        self, key: int | AbstractWire | quantum.QubitSSAValue, default: Any | None = None
    ) -> int | AbstractWire | list[quantum.QubitSSAValue] | None:
        """Remove and return an item from the map, if it exists. Else, return the
        provided default value."""
        if key not in self._map:
            return default

        # At this point, we know that the given key has to be in the map
        if isa(key, quantum.QubitSSAValue):
            wire = self._map.pop(key)
            qubits = self._map[wire]
            _ = qubits.pop(qubits.index(key))
            if len(qubits) == 0:
                _ = self._map.pop(wire)
            return wire

        qubits = self._map.pop(key)
        for q in qubits:
            _ = self._map.pop(q)
        return qubits

    def verify(self):
        """Check if the map is valid. To be valid, the map must satisfy the following conditions:

            * The items inside the map can only be of type ``int``, ``AbstractWire``,
              or ``SSAValue[QubitType]``
            * All wire keys and wire values must be valid wire labels, i.e. belong to
              ``self.wires`` if they are static.
            * The set of wire keys is equal to the set of wire labels.
            * The set of qubit keys is equal to the set of qubit values.
            * There is no intersection between the lists of qubit values.

        If any of the above conditions are not met, an error will be raised.
        """
        qubit_keys = set()
        wire_keys = set()
        all_qubit_values = []
        wire_values = set()

        for k, v in self._map.items():
            self._assert_valid_type(k)

            if isa(k, quantum.QubitSSAValue):
                self._assert_valid_type(v)
                self._assert_valid_wire_label(v)

                qubit_keys.add(k)
                wire_values.add(v)

            else:
                self._assert_valid_wire_label(k)
                assert isinstance(v, list), f"The key {k} maps to an invalid type {v}."

                for _q in v:
                    self._assert_valid_type(_q)

                wire_keys.add(k)
                all_qubit_values.extend(v)

        assert wire_keys == wire_values, "The wire label keys do not match the wire label values."

        all_qubit_values_set = set(all_qubit_values)
        assert qubit_keys == all_qubit_values_set, "The qubit keys do not match the qubit values."
        assert len(all_qubit_values) == len(
            all_qubit_values_set
        ), "Multiple wires are being mapped to the same qubit."

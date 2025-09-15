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
"""WireQubitMap class for a bidirectional map between wire labels and SSA qubits."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from xdsl.dialects import builtin
from xdsl.ir import SSAValue

from ..dialects import quantum


@dataclass(frozen=True)
class AbstractWire:
    """An abstract wire."""

    qreg: quantum.QuregSSAValue | None = None
    """The SSA quantum register from which the qubit corresponding to this
    abstract wire has been extracted. This is used to differentiate between
    qubits allocated in static and dynamic registers. For qubits that are
    allocated dynamically without the use of a register, this is ``None``."""

    idx: SSAValue[builtin.I64] | None = None
    """The SSA integer value that corresponds to the index of the qubit
    in the quantum register. This is used for qubits that are allocated
    in the quantum register. For qubits that are dynamically allocated
    individually, this is ``None``."""

    id: UUID = field(default_factory=uuid4, init=False)
    """A universally unique identifier for the abstract wire. Used when
    qubits are dynamically allocated, since such qubits are not in the
    quantum register."""

    def __hash__(self) -> int:
        if self.qreg and self.idx:
            return hash(self.qreg) + hash(self.idx)
        return hash(self.id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, AbstractWire):
            return False

        if self.qreg and other.qreg and self.idx and other.idx:
            return self.qreg == other.qreg and self.idx == other.idx
        return self.id == other.id


@dataclass
class WireQubitMap:
    """Class to maintain two-way mapping between wire labels and SSA qubits."""

    wires: tuple[int, ...] | None = None
    """Tuple containing all available static wire labels. None if not provided."""

    from_wires: dict[int | AbstractWire, list[quantum.QubitSSAValue]] = field(
        default_factory=dict, init=False
    )
    """Map from wire labels to a list of all known qubit SSAValues to which they correspond."""

    from_qubits: dict[quantum.QubitSSAValue, int | AbstractWire] = field(
        default_factory=dict, init=False
    )
    """Map from qubit SSAValues to their corresponding wire labels."""

    def __contains__(self, key: int | AbstractWire | quantum.QubitSSAValue) -> bool:
        """Check if the map contains a wire label or qubit."""
        return key in self.from_wires or key in self.from_qubits

    def __getitem__(
        self, key: int | AbstractWire | quantum.QubitSSAValue
    ) -> int | AbstractWire | list[quantum.QubitSSAValue]:
        """Get a value from the wire/qubit maps."""
        if isinstance(key, SSAValue):
            return self.from_qubits[key]

        return self.from_wires[key]

    def __setitem__(
        self,
        key: int | AbstractWire | quantum.QubitSSAValue,
        val: quantum.QubitSSAValue | int | AbstractWire,
    ) -> None:
        """Update the wire/qubit maps."""
        if isinstance(key, SSAValue):
            if not isinstance(key.type, quantum.QubitType):
                raise KeyError(
                    "Expected key to be a QubitType SSAValue, instead got SSAValue "
                    f"with type {key.type}"
                )

            if key in self.from_qubits:
                raise ValueError("Cannot update qubits that are already in the map.")

            assert isinstance(val, (int, AbstractWire))
            if isinstance(val, int) and self.wires is not None and val not in self.wires:
                raise ValueError(f"{val} is not an available wire.")

            self.from_qubits[key] = val
            qubits = self.from_wires.setdefault(val, [])
            qubits.append(key)
            return

        if isinstance(key, (int, AbstractWire)):
            if isinstance(key, int) and self.wires is not None and key not in self.wires:
                raise KeyError(f"{key} is not an available wire.")
            if not isinstance(val, SSAValue) or not isinstance(val.type, quantum.QubitType):
                raise ValueError(f"Expected value to be a QubitType SSAValue, instead got {val}.")
            if val in self.from_qubits:
                raise ValueError("Cannot update qubits that are already in the map.")

            qubits = self.from_wires.setdefault(key, [])
            qubits.append(val)
            self.from_qubits[val] = key
            return

        raise KeyError(f"{key} is not a valid wire label or QubitType SSAValue.")

    def get(
        self, key: int | AbstractWire | quantum.QubitSSAValue, default: Any | None = None
    ) -> int | AbstractWire | list[quantum.QubitSSAValue] | None:
        """Return an item from the map without removing it, if it exists. Else, return
        the provided default value."""
        if isinstance(key, SSAValue):
            wire = self.from_qubits.get(key, default)
            return wire

        qubit = self.from_wires.get(key, default)
        return qubit

    def pop(
        self, key: int | AbstractWire | quantum.QubitSSAValue, default: Any | None = None
    ) -> int | AbstractWire | list[quantum.QubitSSAValue] | None:
        """Remove and return an item from the map, if it exists. Else, return the
        provided default value."""
        if isinstance(key, SSAValue):
            wire = self.from_qubits.pop(key, default)
            qubits = self.from_wires[wire]
            _ = qubits.pop(qubits.index(key))
            if len(qubits) == 0:
                _ = self.from_wires.pop(wire, None)
            return wire

        qubits = self.from_wires.pop(key, default)
        for q in qubits:
            _ = self.from_qubits.pop(q, None)
        return qubits

    def verify(self):
        """Check if the map is valid."""
        all_qubits_from_wires = []

        for qubits in self.from_wires.values():
            all_qubits_from_wires.extend(qubits)

        all_qubits_from_wires_set = set(all_qubits_from_wires)
        assert len(all_qubits_from_wires_set) == len(
            all_qubits_from_wires
        ), "There is more than one wire that maps to the same qubit."

        all_qubits_from_qubits_set = set(self.from_qubits.keys())
        assert (
            all_qubits_from_wires_set == all_qubits_from_qubits_set
        ), """The wire-qubit map is invalid. There is a mismatch between the
        number of qubits in both internal maps."""

        all_wires_from_wires_set = set(self.from_wires.keys())
        all_wires_from_qubits_set = set(self.from_qubits.values())
        assert (
            all_wires_from_wires_set == all_wires_from_qubits_set
        ), """The wire-qubit map is invalid. There is a mismatch between the
        number of wires in both internal maps."""

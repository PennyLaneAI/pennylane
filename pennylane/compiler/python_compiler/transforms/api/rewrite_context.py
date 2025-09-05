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
"""A context that maintains state and configuration information for rewriting
hybrid workflows."""

from collections.abc import Sequence
from functools import singledispatchmethod
from uuid import UUID, uuid4

from xdsl.dialects import arith
from xdsl.dialects import stablehlo as xstablehlo
from xdsl.dialects import tensor
from xdsl.ir import BlockArgument
from xdsl.ir import Operation as xOperation
from xdsl.ir import SSAValue

from pennylane.exceptions import CompileError

from ...dialects import mbqc, quantum


class AbstractWire:
    """An abstract wire."""

    id: UUID

    def __init__(self):
        # Create a universally unique identifier
        self.id = uuid4()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class RewriteContext:
    """Rewrite context data class.

    This class provides several abstractions for keep track of useful information
    during pattern rewriting.
    """

    wires = tuple[int] | None
    """Tuple containing all available wire labels. None if not provided."""

    nqubits: int | SSAValue | None
    """Number of allocated qubits. If not known at compile time, will be None
    or an SSAValue."""

    shots: int | SSAValue | None
    """Number of shots. If not known at compile time, will be None
    or an SSAValue."""

    wire_to_qubit_map: dict[int, quantum.QubitSSAValue]
    """Map from wire labels to the latest known qubit SSAValues to which
    they correspond."""

    qubit_to_wire_map: dict[quantum.QubitSSAValue, int]
    """Map from qubit SSAValues to their corresponding wire labels."""

    qreg: quantum.QuregSSAValue | None
    """Most up-to-date quantum register."""

    def __init__(self, wires: Sequence[int] | None = None):
        if wires is not None:
            self.wires = tuple(wires)
            self.nqubits = len(wires)
        else:
            self.wires = None
            self.nqubits = None

        self.wire_to_qubit_map = {}
        self.qubit_to_wire_map = {}
        self.qreg = None

    def update_qubit(
        self, old_qubit: quantum.QubitSSAValue, new_qubit: quantum.QubitSSAValue
    ) -> None:
        """Update a qubit."""
        wire = self.qubit_to_wire_map[old_qubit]
        self.wire_to_qubit_map[wire] = new_qubit
        self.qubit_to_wire_map[new_qubit] = wire
        self.qubit_to_wire_map.pop(old_qubit, None)

    def __getitem__(self, key: int | quantum.QubitSSAValue) -> int | quantum.QubitSSAValue | None:
        if isinstance(key, SSAValue):
            if not isinstance(key.type, quantum.QubitType):
                raise CompileError(
                    f"Expected QubitType SSAValue, instead got SSAValue with type {key.type}"
                )

            return self.qubit_to_wire_map[key]

        if self.wires is not None and key not in self.wires:
            raise CompileError(f"{key} is not an available wire.")

        return self.wire_to_qubit_map[key]

    def __setitem__(
        self, key: int | quantum.QubitSSAValue, val: quantum.QubitSSAValue | int
    ) -> None:
        if isinstance(key, SSAValue):
            if not isinstance(key.type, quantum.QubitType):
                raise CompileError(
                    "Expected key to be a QubitType SSAValue, instead got SSAValue "
                    f"with type {key.type}"
                )
            assert isinstance(val, (int, AbstractWire))

            old_wire = self.qubit_to_wire_map.pop(key, None)
            _ = self.wire_to_qubit_map.pop(old_wire, None)
            self.qubit_to_wire_map[key] = val
            self.wire_to_qubit_map[val] = key

        elif isinstance(key, (int, AbstractWire)):
            if not isinstance(val, SSAValue) or not isinstance(val.type, quantum.QubitType):
                raise CompileError(f"Expected value to be a QubitType SSAValue, instead got {val}.")
            if isinstance(key, int) and self.wires is not None and key not in self.wires:
                raise CompileError(f"{key} is not an available wire.")

            old_qubit = self.wire_to_qubit_map.pop(key, None)
            _ = self.qubit_to_wire_map.pop(old_qubit, None)
            self.wire_to_qubit_map[key] = val
            self.qubit_to_wire_map[val] = key

        raise CompileError(f"{key} is not a valid wire label or QubitType SSAValue.")

    def get_wire_from_extract_op(self, op: quantum.ExtractOp, update=True) -> int | AbstractWire:
        """Get the wire label to which a qubit extraction corresponds."""
        # TODO: Figure out if this method should be removed.
        wire = None
        if (idx_attr := getattr(op, "idx_attr", None)) is not None:
            wire = idx_attr.value.data

        else:
            extract_owner = op.idx.owner

            if isinstance(extract_owner, arith.ConstantOp):
                wire = extract_owner.properties["value"].data

            elif (
                isinstance(extract_owner, tensor.ExtractOp)
                and len(extract_owner.operand[0].type.shape) == 0
                and isinstance(extract_owner.operands[0].owner, xstablehlo.ConstantOp)
            ):
                wire = extract_owner.operands[0].owner.properties["value"].get_values()[0]
            else:
                wire = AbstractWire()

        if wire is not None and update:
            self[wire] = op.qubit
        return wire

    @singledispatchmethod
    def update_from_op(self, op: xOperation):
        """Update the wire mapping from an operation's outputs"""
        # pylint: disable=too-many-branches

        # If any operations, including operations NOT part of the Quantum dialect,
        # return quantum registers, we should update it.
        in_qubits = []
        out_qubits = []

        for result in op.results:
            if isinstance(result.type, quantum.QuregType):
                # We assume that _only one_ of the results is a QuregType
                self.qreg = result
            elif isinstance(result.type, quantum.QubitType):
                out_qubits.append(result)

        if type(op) not in quantum.Quantum.operations:
            # Qubits are only updated if they are returned by operations in
            # the Quantum dialect.
            return

        for operand in op.operands:
            if isinstance(operand.type, quantum.QubitType):
                in_qubits.append(operand)

        assert len(in_qubits) == len(out_qubits)
        for iq, oq in zip(in_qubits, out_qubits, strict=True):
            self.update_qubit(iq, oq)

    @update_from_op.register
    def _update_from_device_init(self, op: quantum.DeviceInitOp):
        """Update the context from a DeviceInitOp."""
        shots = getattr(op, "shots", None)
        if shots:
            shots_owner = shots.owner
            if isinstance(shots_owner, arith.ConstantOp):
                self.shots = shots_owner.value.data
            else:
                assert (
                    isinstance(shots_owner, tensor.ExtractOp)
                    and len(shots_owner.operands[0].type.shape) == 0
                )
                assert isinstance(shots_owner.operands[0], BlockArgument)
                self.shots = shots

    @update_from_op.register
    def _update_from_extract(self, op: quantum.ExtractOp):
        """Update the context from an ExtractOp."""
        # Update wires and qubits
        _ = self.get_wire_from_extract_op(op, update=True)

    @update_from_op.register
    def _update_from_insert(self, op: quantum.InsertOp):
        """Update the context from an InsertOp."""
        # Remove the input qubit and its corresponding wire label from the maps. We're
        # inserting a qubit into the quantum register, so it is no longer valid.
        qubit = op.qubit
        wire = self.qubit_to_wire_map.pop(qubit, None)
        _ = self.wire_to_qubit_map.pop(wire, None)

        # InsertOp returns a new quantum register
        self.qreg = op.results[0]

    @update_from_op.register
    def _update_from_alloc(self, op: quantum.AllocOp | quantum.AllocQubitOp):
        """Update the context from an AllocOp or quantum.AllocQubitOp."""
        if isinstance(op, quantum.AllocOp):
            # Update number of qubits
            if (nqubits_attr := getattr(op, "nqubits_attr", None)) is not None:
                self.nqubits = nqubits_attr.data
            else:
                assert isinstance(op.nqubits.owner, tensor.ExtractOp)
                assert isinstance(op.nqubits.owner.operands[0], BlockArgument)
                self.nqubits = op.nqubits

            # Update quantum register
            self.qreg = op.results[0]

        else:
            qubit = op.results[0]
            self[qubit] = AbstractWire()

    @update_from_op.register
    def _update_from_dealloc(self, op: quantum.DeallocOp | quantum.DeallocQubitOp):
        """Update the context from a DeallocOp or quantum.DeallocQubitOp."""
        if isinstance(op, quantum.DeallocOp):
            assert self.qreg == op.operands[0]
            self.qreg = None

        else:
            qubit = op.operands[0]
            assert qubit in self.qubit_to_wire_map
            wire = self.qubit_to_wire_map.pop(qubit)
            _ = self.wire_to_qubit_map.pop(wire, None)

    @update_from_op.register
    def _update_from_measure(self, op: quantum.MeasureOp | mbqc.MeasureInBasisOp):
        """Update the context from a MeasureOp."""
        self.update_qubit(op.in_qubit, op.out_qubit)

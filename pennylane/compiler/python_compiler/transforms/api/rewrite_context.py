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
from uuid import UUID, uuid4

from xdsl.dialects import arith
from xdsl.dialects import stablehlo as xstablehlo
from xdsl.dialects import tensor
from xdsl.ir import BlockArgument
from xdsl.ir import Operation as xOperation
from xdsl.ir import SSAValue

from pennylane.exceptions import CompileError

from ...dialects import mbqc, quantum

# Tuple of all operations that return qubits
_ops_returning_qubits = (
    quantum.CustomOp,
    quantum.AllocQubitOp,
    quantum.ExtractOp,
    quantum.GlobalPhaseOp,
    quantum.MeasureOp,
    quantum.MultiRZOp,
    quantum.QubitUnitaryOp,
    quantum.SetBasisStateOp,
    quantum.SetStateOp,
    mbqc.MeasureInBasisOp,
)

# Tuple of all operations that return "out_qubits"
_out_qubits_ops = (
    quantum.CustomOp,
    quantum.MultiRZOp,
    quantum.QubitUnitaryOp,
    quantum.SetBasisStateOp,
    quantum.SetStateOp,
)

# Tuple of all operations that return "out_ctrl_qubits"
_out_ctrl_qubits_ops = (
    quantum.CustomOp,
    quantum.GlobalPhaseOp,
    quantum.MultiRZOp,
    quantum.QubitUnitaryOp,
)

# Tuple of all operations that return "out_qubit"
_out_qubit_ops = (quantum.MeasureOp, mbqc.MeasureInBasisOp)

# Tuple of all operations that return "qubit"
_qubit_ops = (quantum.AllocQubitOp, quantum.ExtractOp)


class AbstractWire:
    """A class representing an abstract wire."""

    id: UUID

    def __init__(self):
        # Create a universally unique identifier
        self.id = uuid4()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class RewriteContext:
    """Rewrite state manager class.

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
        wire = None
        if (idx_attr := getattr(op, "idx_attr", None)) is not None:
            wire = idx_attr.value.data

        else:
            extract_owner = op.idx.owner

            if isinstance(extract_owner, arith.ConstantOp):
                wire = extract_owner.properties["value"].data

            elif (
                isinstance(extract_owner, tensor.ExtractOp)
                and len(extract_owner.operand[0].shape) == 0
                and isinstance(extract_owner.operands[0].owner, xstablehlo.ConstantOp)
            ):
                wire = extract_owner.operands[0].owner.properties["value"].get_values()[0]
            else:
                wire = AbstractWire()

        if wire is not None and update:
            self[wire] = op.qubit
        return wire

    # TODO: Use singledispatchmethod

    def update_from_op(self, op: xOperation):
        """Update the wire mapping from an operation's outputs"""
        # pylint: disable=too-many-branches
        if isinstance(op, quantum.DeviceInitOp):
            shots = getattr(op, "shots", None)
            if shots:
                shots_owner = shots.owner
                if isinstance(shots_owner, arith.ConstantOp):
                    self.shots = shots_owner.value.data
                else:
                    assert isinstance(shots_owner, tensor.ExtractOp)
                    assert isinstance(shots_owner.operands[0], BlockArgument)
                    self.shots = shots
            shots = 0
            return

        for r in op.results:
            if isinstance(r, quantum.QuregType):
                if isinstance(op, quantum.AllocOp):
                    nqubits = getattr(op, "nqubits_attr", None)
                    if nqubits:
                        self.nqubits = nqubits.data
                    else:
                        assert isinstance(op.nqubits.owner, tensor.ExtractOp)
                        assert isinstance(op.nqubits.owner.operands[0], BlockArgument)
                        self.nqubits = op.nqubits

                self.qreg = r
                # We assume that only one of the results is a QuregType
                break

        if isinstance(op, quantum.ExtractOp):
            _ = self.get_wire_from_extract_op(op, update=True)
            return

        if isinstance(op, _out_qubit_ops):
            self.update_qubit(op.in_qubit, op.out_qubit)

        if isinstance(op, _out_qubits_ops):
            for iq, oq in zip(op.in_qubits, op.out_qubits, strict=True):
                self.update_qubit(iq, oq)

        if isinstance(op, _out_ctrl_qubits_ops):
            for iq, oq in zip(op.in_ctrl_qubits, op.out_ctrl_qubits, strict=True):
                self.update_qubit(iq, oq)

        if isinstance(op, (quantum.InsertOp, quantum.DeallocQubitOp)):
            qubit = op.qubit
            wire = self.qubit_to_wire_map.pop(qubit, None)
            _ = self.wire_to_qubit_map.pop(wire, None)

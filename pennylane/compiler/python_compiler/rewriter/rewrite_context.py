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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Any
from uuid import UUID, uuid4

from xdsl.dialects import arith
from xdsl.dialects import stablehlo as xstablehlo
from xdsl.dialects import tensor
from xdsl.ir import Block, BlockArgument
from xdsl.ir import Operation as xOperation
from xdsl.ir import Region, SSAValue

from ..dialects import mbqc, quantum


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


@dataclass
class WireQubitMap:
    """Class to maintain two-way mapping between wire labels and SSA qubits."""

    wires: tuple[int] | None = field(default=None)
    """Tuple containing all available wire labels. None if not provided."""

    wire_to_qubit_map: dict[int | AbstractWire, quantum.QubitSSAValue] = {}
    """Map from wire labels to the latest known qubit SSAValues to which
    they correspond."""

    qubit_to_wire_map: dict[quantum.QubitSSAValue, int | AbstractWire] = {}
    """Map from qubit SSAValues to their corresponding wire labels."""

    def __post_init__(self):
        if self.wires is not None:
            self.wires = tuple(self.wires)

    def __contains__(self, key: int | AbstractWire | quantum.QubitSSAValue) -> bool:
        """Check if the map contains a wire label or qubit."""
        return key in self.wire_to_qubit_map or key in self.qubit_to_wire_map

    def __getitem__(
        self, key: int | AbstractWire | quantum.QubitSSAValue
    ) -> int | AbstractWire | quantum.QubitSSAValue:
        """Get a value from the wire/qubit maps."""
        if isinstance(key, SSAValue):
            if not isinstance(key.type, quantum.QubitType):
                raise TypeError(
                    f"Expected QubitType SSAValue, instead got SSAValue with type {key.type}"
                )

            return self.qubit_to_wire_map[key]

        if self.wires is not None and key not in self.wires:
            raise KeyError(f"{key} is not an available wire.")

        return self.wire_to_qubit_map[key]

    def __setitem__(
        self,
        key: int | AbstractWire | quantum.QubitSSAValue,
        val: quantum.QubitSSAValue | int | AbstractWire,
    ) -> None:
        """Update the wire/qubit maps."""
        if isinstance(key, SSAValue):
            if not isinstance(key.type, quantum.QubitType):
                raise TypeError(
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
                raise TypeError(f"Expected value to be a QubitType SSAValue, instead got {val}.")
            if isinstance(key, int) and self.wires is not None and key not in self.wires:
                raise KeyError(f"{key} is not an available wire.")

            old_qubit = self.wire_to_qubit_map.pop(key, None)
            _ = self.qubit_to_wire_map.pop(old_qubit, None)
            self.wire_to_qubit_map[key] = val
            self.qubit_to_wire_map[val] = key

        raise TypeError(f"{key} is not a valid wire label or QubitType SSAValue.")

    def pop(self, key: int | AbstractWire | quantum.QubitSSAValue, default: Any | None = None):
        """Remove and return an item from the map, if it exists. Else, return the provided default value."""
        if isinstance(key, SSAValue):
            if not isinstance(key.type, quantum.QubitType):
                raise TypeError(
                    "Expected key to be a QubitType SSAValue, instead got SSAValue "
                    f"with type {key.type}"
                )
            wire = self.qubit_to_wire_map.pop(key, default)
            _ = self.wire_to_qubit_map.pop(wire, None)
            return wire

        qubit = self.wire_to_qubit_map.pop(key, default)
        _ = self.qubit_to_wire_map.pop(qubit, None)
        return qubit

    def update_qubit(
        self, old_qubit: quantum.QubitSSAValue, new_qubit: quantum.QubitSSAValue
    ) -> None:
        """Replace a qubit in the map with a new one.

        Args:
            old_qubit (SSAValue[QubitType]): The old qubit to remove.
            new_qubit (SSAValue[QubitType]): The new qubit to add.
        """
        wire = self.qubit_to_wire_map.pop(old_qubit)
        if wire is None:
            raise KeyError(f"{old_qubit} is not in the WireQubitMap.")

        self.wire_to_qubit_map[wire] = new_qubit
        self.qubit_to_wire_map[new_qubit] = wire


@dataclass(init=False)
class RewriteContext:
    """Rewrite context data class.

    This class provides several abstractions for keep track of useful information
    during pattern rewriting.
    """

    nqubits: int | SSAValue | None = None
    """Number of allocated qubits. If not known at compile time, will be None
    or an SSAValue. Else, it will be an integer."""

    shots: int | SSAValue | None = None
    """Number of shots. If not known at compile time, will be None
    or an SSAValue. Else, it will be an integer."""

    wire_qubit_map: WireQubitMap = field(default_factory=WireQubitMap)
    """Two way map between wire labels and SSA qubits."""

    qreg: quantum.QuregSSAValue | None = None
    """Most up-to-date quantum register."""

    def __init__(self, wires: Sequence[int] | None = None):
        if wires is not None:
            self.wire_qubit_map = WireQubitMap(wires=wires)

    # TODO: Figure out if this method should be removed.
    def get_wire_from_extract_op(self, op: quantum.ExtractOp, update=True) -> int | AbstractWire:
        """Get the wire label to which a qubit extraction corresponds.

        Args:
            op (quantum.ExtractOp): ``ExtractOp`` from which we want to get the wire label.
                If the wire is dynamic, then an ``AbstractWire`` will be returned.
            update (bool): Whether or not to update the wire/qubit mapping. ``True`` by default.

        Returns:
            int | AbstractWire: Wire label corresponding to the qubit being extracted. If dynamic,
            an ``AbstractWire`` is returned.
        """
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
            self.wire_qubit_map[wire] = op.qubit
        return wire

    @singledispatchmethod
    def update_from_op(self, op: xOperation):
        """Update the context using an operation's outputs.

        This method uses the outputs of the input operation to update the various fields
        that the context maintains.

        Args:
            op (xdsl.ir.Operation): The operation whose results will be used to update
                the RewriteContext.
        """

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
            self.wire_qubit_map.update_qubit(iq, oq)

    @update_from_op.register
    def _update_from_device_init(self, op: quantum.DeviceInitOp):
        """Update the context from a ``DeviceInitOp``. The operation is used to update
        the shots."""
        # If shots are constant, they will be captured as the value of an ``arith.constant``
        #  or ``stablehlo.constant``.
        # Otherwise, they will be an argument to the function (``BlockArgument``).
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
                if isinstance(shots_owner.operands[0], BlockArgument):
                    # Shots are dynamic, i.e., a ``BlockArgument``
                    self.shots = shots

                else:
                    # Shots are static, but created using ``stablehlo.constant``
                    cst_owner = shots_owner.operands[0].owner
                    assert isinstance(cst_owner, xstablehlo.ConstantOp)
                    self.nqubits = cst_owner.properties["value"].get_values()[0]

    @update_from_op.register
    def _update_from_extract(self, op: quantum.ExtractOp):
        """Update the context from an ``ExtractOp``. The operation is used to update qubits."""
        # Update wires and qubits
        _ = self.get_wire_from_extract_op(op, update=True)

    @update_from_op.register
    def _update_from_insert(self, op: quantum.InsertOp):
        """Update the context from an ``InsertOp``. The operation is used to update qubits
        and the quantum register."""
        # Remove the input qubit and its corresponding wire label from the maps. We're
        # inserting a qubit into the quantum register, so it is no longer valid.
        qubit = op.qubit
        _ = self.wire_qubit_map.pop(qubit)

        # InsertOp returns a new quantum register
        self.qreg = op.results[0]

    @update_from_op.register
    def _update_from_alloc(self, op: quantum.AllocOp):
        """Update the context from an ``AllocOp``. The operation is used to update the number
        of qubits and the quantum register."""
        # If the number of qubits is static, they will either be captured as an IntegerAttr,
        # or as an SSAValue using ``stablehlo.constant``.
        # If number of qubits are dynamic, they will be a function argument (``BlockArgument``).
        if (nqubits_attr := getattr(op, "nqubits_attr", None)) is not None:
            self.nqubits = nqubits_attr.data
        else:
            nqubits_owner = op.nqubits.owner
            assert isinstance(nqubits_owner, tensor.ExtractOp)
            if isinstance(nqubits_owner.operands[0], BlockArgument):
                # Number of qubits is dynamic, i.e., ``BlockArgument``
                self.nqubits = op.nqubits
            else:
                # Number of qubits is constant, but created using ``stablehlo.constant``
                cst_owner = nqubits_owner.operands[0].owner
                assert isinstance(cst_owner, xstablehlo.ConstantOp)
                self.nqubits = cst_owner.properties["value"].get_values()[0]

        # Update quantum register
        self.qreg = op.results[0]

    @update_from_op.register
    def _update_from_alloc_qubit(self, op: quantum.AllocQubitOp):
        """Update the context from an ``AllocQubitOp``. The operation is used to update
        qubits."""
        qubit = op.results[0]
        self[qubit] = AbstractWire()

    @update_from_op.register
    def _update_from_dealloc(self, op: quantum.DeallocOp):
        """Update the context from a ``DeallocOp``. The operation is used to delete the
        quantum register."""
        if isinstance(op, quantum.DeallocOp):
            assert self.qreg == op.operands[0]
            self.qreg = None

    @update_from_op.register
    def _update_from_dealloc_qubit(self, op: quantum.DeallocQubitOp):
        """Update the context from a ``DeallocQubitOp``. The operation is used to delete
        the dynamically allocated qubit."""
        qubit = op.operands[0]
        assert qubit in self.wire_qubit_map
        _ = self.wire_qubit_map.pop(qubit)

    @update_from_op.register
    def _update_from_num_qubits(self, op: quantum.NumQubitsOp):
        """Update the context from a ``NumQubitsOp``. The operation is used to update
        the number of qubits."""
        self.nqubits = op.results[0]

    @update_from_op.register
    def _update_from_measure(self, op: quantum.MeasureOp | mbqc.MeasureInBasisOp):
        """Update the context from a ``MeasureOp`` or ``MeasureInBasisOp``. The operation is
        used to update qubits."""
        self.wire_qubit_map.update_qubit(op.in_qubit, op.out_qubit)

    def walk(
        self, obj: xOperation | Block | Region, reverse: bool = False, region_first: bool = False
    ) -> Iterator[xOperation]:
        """Walk over the body of the provided object.

        This method provides a similar API to ``xdsl.ir.Operation.walk`` for traversing the body of
        an operation, block, or region, while updating the ``RewriteContext`` at the same time.

        Args:
            obj (xdsl.ir.Operation | xdsl.ir.Block | xdsl.ir.Region): The object whose body to walk.
            reverse (bool): Whether to traverse the body of the given object in reverse. ``False``
                by default.
            region_first (bool): Whether to traverse the regions of an operation before the operation
                itself. ``False`` by default.
        """

        for op in obj.walk(reverse=reverse, region_first=region_first):
            yield op

            # Update the context
            self.update_from_op(op)

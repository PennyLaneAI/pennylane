# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY_STATE KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the base class for wire management."""
import uuid
from collections.abc import Iterable

from pennylane.allocation import AllocateState
from pennylane.estimator.estimate import _get_resource_decomposition
from pennylane.estimator.resource_config import ResourceConfig
from pennylane.estimator.resource_mapping import _map_to_resource_op
from pennylane.estimator.resource_operator import GateCount, ResourceOperator
from pennylane.estimator.resources_base import DefaultGateSet
from pennylane.measurements.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires


class _WireAction:
    """Base class for operations that manage wire resources."""

    def __init__(self, num_wires, state=AllocateState.ZERO, restored=False):
        self.num_wires = num_wires
        self.state = state
        self.restored = restored

    # def __eq__(self, other: "_WireAction") -> bool:
    #     return isinstance(other, self.__class__) and self.num_wires == other.num_wires

    def __mul__(self, other):
        if isinstance(other, int):
            return self.__class__(self.num_wires * other)
        raise NotImplementedError


class Allocate(_WireAction):
    def __repr__(self) -> str:
        return f"Allocate({self.num_wires}, state={self.state}, restored={self.restored})"


class Deallocate(_WireAction):
    def __init__(
        self, num_wires=None, allocated_register=None, state=AllocateState.ZERO, restored=False
    ):
        if num_wires is None and allocated_register is None:
            raise ValueError("Atleast one of `num_wires` and `allocated_register` must be provided")

        if allocated_register is not None and num_wires is not None:
            if num_wires != allocated_register.num_wires:
                raise ValueError("`num_wires` argument must match `allocated_register.num_wires`")

        if state == AllocateState.ANY:
            if restored == True:
                if allocated_register is None:
                    raise ValueError(
                        "Must provide `allocated_register` when deallocating an ANY state register with `restored=True`"
                    )
            else:
                raise ValueError(
                    "When deallocating an ANY state, `restored=False` is not supported."
                )

        if num_wires is None:
            num_wires = allocated_register.num_wires

        self.state = state
        self.restored = restored
        self.num_wires = num_wires
        self.allocated_register = allocated_register

    def __repr__(self) -> str:
        return f"Deallocate(({self.num_wires}, state={self.state}, restored={self.restored}))"


class MarkQubits:
    def __init__(self, wires):
        self.wires = Wires(wires) if wires is not None else Wires([])
        if QueuingManager.recording():
            self.queue()

    def queue(self, context=QueuingManager):
        r"""Adds the wire action object to a queue."""
        context.append(self)
        return self


class MarkAuxiliary(MarkQubits):
    def __repr__(self) -> str:
        return f"MarkAuxiliary({self.wires})"


class MarkLogical(MarkQubits):
    def __repr__(self) -> str:
        return f"MarkLogical({self.wires})"


class MarkClean(MarkQubits):
    def __repr__(self) -> str:
        return f"MarkClean({self.wires})"


def _estimate_auxiliary_wires(
    list_actions: Iterable[GateCount, Allocate, Deallocate],
    scalar: int = 1,
    gate_set: set[str] | None = None,
    config: ResourceConfig | None = None,
    num_available_any_state_aux: int = 0,
) -> Iterable:
    if scalar == 0:
        return 0, 0, 0
    if config is None:
        config = ResourceConfig()
    if gate_set is None:
        gate_set = DefaultGateSet
    any_state_aux_allocation = {}

    total = 0
    max_alloc = 0
    max_dealloc = 0

    for action in list_actions:
        if isinstance(action, GateCount):
            if action.gate.name in gate_set:
                continue

            resource_decomp = _get_resource_decomposition(action.gate, config)
            sub_max_alloc, sub_max_dealloc, sub_total = _estimate_auxiliary_wires(
                resource_decomp,
                action.count,
                gate_set,
                config,
                num_available_any_state_aux,
            )

            if sub_max_dealloc < 0:
                total += sub_max_dealloc  # sub_max_dealloc < 0
                if total < max_dealloc:
                    max_dealloc = total
            total -= sub_max_dealloc  # reset

            if sub_max_alloc > 0:
                total += sub_max_alloc
                if total > max_alloc:
                    max_alloc = total
            total -= sub_max_alloc  # reset

            total += sub_total
            continue

        elif isinstance(action, Allocate):
            if action.state == AllocateState.ANY and action.restored == True:
                diff = num_available_any_state_aux - action.num_wires

                if diff < 0:
                    total += abs(diff)
                    any_state_aux_allocation[action] = abs(diff)
                    num_available_any_state_aux = 0

                else:
                    any_state_aux_allocation[action] = 0
                    num_available_any_state_aux = diff

            else:
                total += action.num_wires

        elif isinstance(action, Deallocate):
            if action.state == AllocateState.ANY and action.restored == True:
                try:
                    associated_alloc = any_state_aux_allocation.pop(action.allocated_register)
                    total -= associated_alloc
                    num_available_any_state_aux += action.num_wires - associated_alloc

                except KeyError as e:
                    raise ValueError(
                        f"Trying to deallocate an ANY state register before it was allocated {action}"
                    ) from e

            else:
                total -= action.num_wires

        if total > max_alloc:
            max_alloc = total
        if total < max_dealloc:
            max_dealloc = total

    if len(any_state_aux_allocation) != 0:
        raise ValueError(
            f"Did NOT deallocate and restore all ANY state allocations as promised:\n{any_state_aux_allocation}"
        )

    if scalar == 1:
        return max_alloc, max_dealloc, total

    if total > 0:
        max_alloc += (scalar - 1) * total
    if total < 0:
        max_dealloc += (scalar - 1) * total
    total *= scalar

    return max_alloc, max_dealloc, total


def _process_circuit_lst(circuit_as_lst):
    circuit_wires = Wires([])
    num_generated_wires = 0
    generated_wire_labels = []

    processed_circ = []
    for op in circuit_as_lst:
        if not isinstance(op, (ResourceOperator, Operator, MeasurementProcess, MarkQubits)):
            raise ValueError(
                f"Circuit must contain only instances of 'ResourceOperator', 'Operator', 'MeasurementProcess' and 'MarkQubits', got {type(op)}"
            )

        elif isinstance(op, Operator):
            op_wires = op.wires
            cmp_rep_op = _map_to_resource_op(op).resource_rep_from_op()

            processed_circ.append((cmp_rep_op, op_wires))
            circuit_wires += op_wires

        elif isinstance(op, ResourceOperator):
            op_wires = op.wires
            cmp_rep_op = op.resource_rep_from_op()

            if op_wires is None:
                num_wires = op.num_wires
                diff = num_generated_wires - num_wires

                if diff < 0:  # generate additional wire labels
                    for _ in range(abs(diff)):
                        generated_wire_labels.append(str(uuid.uuid4()))
                    num_generated_wires += abs(diff)

                op_wires = Wires(generated_wire_labels[:num_wires])

            processed_circ.append((cmp_rep_op, op_wires))
            circuit_wires += op_wires

        elif isinstance(op, MarkQubits):
            marked_wires = op.wires
            processed_circ.append((op, marked_wires))

    for op, op_wires in processed_circ:
        if isinstance(op, MarkQubits) and (len(op_wires - circuit_wires) != 0):
            raise ValueError(
                f"Attempted to mark qubits {op_wires - circuit_wires} which don't exist in the circuit wires {circuit_wires}"
            )

    return processed_circ, circuit_wires


def estimate_wires_from_circuit(
    circuit_as_lst: Iterable[ResourceOperator, Operator, MeasurementProcess, MarkQubits],
    gate_set: set,
    config: dict,
):
    processed_circ, circuit_wires = _process_circuit_lst(circuit_as_lst)
    total_algo_qubits = len(circuit_wires)

    logical_auxiliary_wires = Wires([])
    state_circuit_wires = {w: 0 for w in circuit_wires}  # 0: clean state, 1: any state

    total = 0
    max_alloc = 0
    max_dealloc = 0

    for circuit_element, active_wires in processed_circ:
        if isinstance(circuit_element, MarkQubits):
            if isinstance(circuit_element, MarkClean):
                for w in active_wires:
                    state_circuit_wires[w] = 0

            if isinstance(circuit_element, MarkAuxiliary):
                logical_auxiliary_wires += active_wires

            if isinstance(circuit_element, MarkLogical):
                logical_auxiliary_wires -= active_wires

        else:
            for w in active_wires:
                state_circuit_wires[w] = 1

            available_logical_auxiliaries = logical_auxiliary_wires - active_wires
            num_dirty_logical_auxs = sum(
                (state_circuit_wires[w_i] for w_i in available_logical_auxiliaries)
            )
            num_clean_logical_auxs = len(available_logical_auxiliaries) - num_dirty_logical_auxs

            sub_max_alloc, sub_max_dealloc, sub_total = _estimate_auxiliary_wires(
                [GateCount(circuit_element)],
                gate_set=gate_set,
                config=config,
                num_available_any_state_aux=num_dirty_logical_auxs,
            )

            borrowable_qubits = sub_max_alloc - sub_total
            num_clean_aux_used = min(num_clean_logical_auxs, borrowable_qubits)
            sub_max_alloc -= num_clean_aux_used

            if sub_max_dealloc < 0:
                total += sub_max_dealloc  # sub_max_dealloc < 0
                if total < max_dealloc:
                    max_dealloc = total
            total -= sub_max_dealloc  # reset

            if total < 0:
                raise ValueError("Deallocated more qubits than available to allocate.")

            if sub_max_alloc > 0:
                total += sub_max_alloc
                if total > max_alloc:
                    max_alloc = total
            total -= sub_max_alloc  # reset

            total += sub_total

    if total < 0 or max_dealloc < 0:
        raise ValueError("Deallocated more qubits than available to allocate.")

    any_state = total
    zeroed = max_alloc - total
    return total_algo_qubits, any_state, zeroed

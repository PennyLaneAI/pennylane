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
"""This module contains the core logic for wire management."""

from collections.abc import Iterable

from pennylane.allocation import AllocateState
from pennylane.estimator.estimate import _get_resource_decomposition
from pennylane.estimator.resource_mapping import _map_to_resource_op
from pennylane.estimator.resource_operator import GateCount, ResourceOperator
from pennylane.estimator.resources_base import DefaultGateSet, Resources
from pennylane.estimator.wires_manager import Allocate as estimator_Allocate
from pennylane.estimator.wires_manager import Deallocate as estimator_Deallocate
from pennylane.labs.estimator_beta.resource_config import LabsResourceConfig
from pennylane.measurements.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.wires import Wires

from .base_classes import Allocate, AllocateState, Deallocate, MarkClean, MarkQubits


def _estimate_auxiliary_wires(
    list_actions: Iterable[GateCount | Allocate | Deallocate],
    scalar: int = 1,
    gate_set: set = DefaultGateSet,
    config: LabsResourceConfig | None = None,
    num_available_any_state_aux: int = 0,
    num_active_qubits: int = 0,
):  # pylint: disable=too-many-arguments,too-many-branches,too-many-statements
    """A recursive function that tracks auxiliary qubits via three quantities over the course of the workflow.
    It tracks the maximum number of qubits allocated, the maximum number of qubits deallocated and the total
    number of allocated qubits that weren't restored to the zero state by the end of the workflow.

    Args:
        list_actions (Iterable[GateCount | Allocate | Deallocate]): A quantum circuit represented by a list
            of circuit elements. The circuit elements are made up of gates with counts (``GateCount``),
            qubit allocation instructions (``Allocate``) and qubit deallocation instructions (``Deallocate``).
        scalar (int): A positive integer or zero representing how many times this quantum circuit
            (``list_actions``) is repeated.
        gate_set (set[str]): A set of names (strings) of the fundamental operators to count
            throughout the quantum workflow. If not provided, the default gate set will be used,
            i.e., ``{'Toffoli', 'T', 'CNOT', 'X', 'Y', 'Z', 'S', 'Hadamard'}``.
        config (LabsResourceConfig): configurations for the resource estimation pipeline
        num_available_any_state_aux (int): The number of external qubits, in any quantum state, that
            can be treated as auxiliary and borrowed for use within this workflow. These would potentially reduce
            the number of qubits allocated within the workflow.
        num_active_qubits (int): The total number of qubits (not auxiliary) that the operators
            in the workflow act upon.

    Returns:
        (int): A positive integer (or zero) representing the maximum number of qubits allocated (``max_alloc``).
        (int): A negative integer (or zero) representing the maximum number of qubits deallocated (``max_dealloc``).
        (int): An integer representing the total number of allocated qubits that weren't restored to the
        zero state by the end of the workflow (``total``). A positive value indicates that there were more
        allocated qubits than deallocated, a negative value indicates the opposite. A zero value indicates
        that all allocated qubits were deallocated.

    Raises:
        ValueError: if fails to deallocate and restore all ANY state allocations as required
        ValueError: if tries to deallocate an ANY state register before it was allocated
    """
    if scalar == 0:
        return 0, 0, 0

    if config is None:
        config = LabsResourceConfig()

    total = 0
    max_alloc = 0
    max_dealloc = 0
    any_state_aux_allocation = {}
    local_num_available_any_state_aux = num_available_any_state_aux - num_active_qubits

    if local_num_available_any_state_aux < 0:
        raise ValueError(
            f"`local_num_available_any_state_aux` shouldn't be negative, got {local_num_available_any_state_aux}. `num_available_any_state_aux` should always be greater than or equal to `num_active_qubits`. This could be caused by incorrect `num_wires` for resource operators."
        )

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
                num_available_any_state_aux + total,
                num_active_qubits=action.gate.num_wires,
            )

            max_alloc = max(max_alloc, total + sub_max_alloc)
            max_dealloc = min(max_dealloc, total + sub_max_dealloc)  # sub_max_dealloc < 0

            total += sub_total
            continue

        if isinstance(action, (Allocate, estimator_Allocate)):
            if isinstance(action, estimator_Allocate):
                action = Allocate(action.num_wires)

            if action.state == AllocateState.ANY and action.restored is True:
                diff = local_num_available_any_state_aux - action.num_wires

                if diff < 0:
                    total += abs(diff)
                    any_state_aux_allocation[action] = abs(diff)
                    local_num_available_any_state_aux = 0

                else:
                    any_state_aux_allocation[action] = 0
                    local_num_available_any_state_aux = diff

            else:
                total += action.num_wires

        if isinstance(action, (Deallocate, estimator_Deallocate)):
            if isinstance(action, estimator_Deallocate):
                action = Deallocate(num_wires=action.num_wires)

            if action.state == AllocateState.ANY and action.restored is True:
                try:
                    associated_alloc = any_state_aux_allocation.pop(action.allocated_register)
                    total -= associated_alloc
                    local_num_available_any_state_aux += action.num_wires - associated_alloc

                except KeyError as e:
                    raise ValueError(
                        f"Trying to deallocate an ANY state register before it was allocated {action}"
                    ) from e

            else:
                total -= action.num_wires

        max_alloc = max(max_alloc, total)
        max_dealloc = min(max_dealloc, total)

    if len(any_state_aux_allocation) != 0:
        raise ValueError(
            "Failed to uncompute and restore all `ANY state` allocations. "
            "Dirty auxiliaries must be restored to their initial states to close the operational scope. "
            f"Unresolved wires: {any_state_aux_allocation}"
        )

    if total > 0:
        max_alloc += (scalar - 1) * total
    if total < 0:
        max_dealloc += (scalar - 1) * total
    total *= scalar

    return max_alloc, max_dealloc, total


def _process_circuit_lst(
    circuit_as_lst: Iterable[ResourceOperator | Operator | MeasurementProcess | MarkQubits],
):
    r"""A private function that pre-processes the quantum tape obtained from a qfunc as part of the wire
    tracking pipeline.

    This function has three main responsibilities. Firstly, mapping and pruning all operators (``ResourceOperator``
    or ``Operator``) to their associated ``CompressedResourceOp``, ignoring any measurements
    (``MeasurementProcess``). Secondly, it extracts and stores the wires each operator acts upon, obtaining the
    set of all wires in the circuit. Finally, in case wire labels are not provided for certain operators, unique
    wires are generated for the operator and tracked as part of the circuit wires.

    Args:
        circuit_as_lst (Iterable[ResourceOperator | Operator | MeasurementProcess | MarkQubits]): A quantum circuit
            represented by a list of circuit elements (operators, measurements, etc,).

    Returns:
        tuple(list[CompressedResourceOp, MarkQubits], Wires): Returns the processed circuit and the circuit wires.
        The processed circuit is a list of tuples where each tuple contains two objects, a circuit element (either
        ``CompressedResourceOp`` or ``MarkQubits`` instances) and the wires it acts upon (``Wires``).

    Raises:
        ValueError: If incompatible type of object is encountered. Circuit must contain only instances
            of 'ResourceOperator', 'Operator', 'MeasurementProcess' and 'MarkQubits'.
        ValueError: if attempts to mark qubits that don't otherwise exist in the circuit wires
    """
    circuit_wires = Wires([])
    num_generated_wires = 0
    generated_wire_labels = []

    processed_circ = []
    for op in circuit_as_lst:
        if not isinstance(op, (ResourceOperator, Operator, MeasurementProcess, MarkQubits)):
            raise ValueError(
                f"Circuit must contain only instances of 'ResourceOperator', 'Operator', 'MeasurementProcess' and 'MarkQubits', got {type(op)}"
            )

        if isinstance(op, Operator):
            op_wires = op.wires
            cmp_rep_op = _map_to_resource_op(op).resource_rep_from_op()

            processed_circ.append((cmp_rep_op, op_wires))
            circuit_wires += op_wires

        elif isinstance(op, ResourceOperator):
            op_wires = op.wires
            cmp_rep_op = op.resource_rep_from_op()

            if op_wires is None:
                num_wires = op.num_wires
                diff = num_wires - num_generated_wires

                if diff > 0:  # generate additional wire labels
                    for i in range(diff):
                        generated_wire_labels.append(f"__generated_wire{num_generated_wires + i}__")
                    num_generated_wires += diff

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
    circuit_as_lst: Iterable[ResourceOperator | Operator | MeasurementProcess | MarkQubits],
    gate_set: set | None = None,
    config: LabsResourceConfig | None = None,
    zeroed: int = 0,
    any_state: int = 0,
):
    r"""Determine the number of auxiliary qubits needed to decompose the operators
    of a quantum circuit into a specific ``gate_set`` with a given ``config``.

    Args:
        circuit_as_lst (Iterable[ResourceOperator | Operator | MeasurementProcess | MarkQubits]): A quantum circuit
            represented by a list of circuit elements (operators, measurements, etc.).
        gate_set (set[str] | None): A set of names (strings) of the fundamental operators to count
            throughout the quantum workflow. If not provided, the default gate set will be used,
            i.e., ``{'Toffoli', 'T', 'CNOT', 'X', 'Y', 'Z', 'S', 'Hadamard'}``.
        config (LabsResourceConfig | None): configurations for the resource estimation pipeline
        zeroed (int): The number of additional auxiliary wires, prepared in the
            zero state, that can be used as part of the decomposition.
        any_state (int): The number of additional auxiliary wires, prepared in
            any state, that can be used as part of the decomposition.

    Returns:
        tuple(int, int, int): The number of qubits used as part of the decomposition. The first integer
        represents the number of qubits required to define the circuit (before decomposition). The remaining
        two integers represent the number of auxiliary qubits required as we decompose the circuit. They are
        separated according to their quantum state at the end of the workflow (``any_state``, ``zeroed``).

    Raises:
        ValueError: if more qubits were deallocated than initially allocated
    """
    if config is None:
        config = LabsResourceConfig()

    if gate_set is None:
        gate_set = DefaultGateSet

    processed_circ, circuit_wires = _process_circuit_lst(circuit_as_lst)
    total_algo_qubits = len(circuit_wires)

    state_circuit_wires = {w: 1 for w in circuit_wires}  # 1: clean state, 0: any state

    total = 0  # A running counter for the number of active (allocated but not freed) qubits
    #   --> we assume that these are in any state as they were likely used and not cleaned
    max_alloc = zeroed
    max_dealloc = 0

    for circuit_element, active_wires in processed_circ:
        if isinstance(circuit_element, MarkQubits):
            if isinstance(circuit_element, MarkClean):
                for w in active_wires:
                    state_circuit_wires[w] = 1

        else:
            for w in active_wires:
                state_circuit_wires[w] = 0

            num_clean_logical_wires = sum((state_circuit_wires[w_i] for w_i in circuit_wires))
            num_any_state_logical_wires = (
                len(circuit_wires) - num_clean_logical_wires
            )  # Note this contains the wires that circuit_element acts on

            sub_max_alloc, sub_max_dealloc, sub_total = _estimate_auxiliary_wires(
                [GateCount(circuit_element)],
                gate_set=gate_set,
                config=config,
                num_available_any_state_aux=num_any_state_logical_wires + total + any_state,
                num_active_qubits=circuit_element.num_wires,  # Should be equivalent to len(active_wires)
            )

            borrowable_qubits = sub_max_alloc - sub_total
            num_clean_aux_used = min(num_clean_logical_wires, borrowable_qubits)
            sub_max_alloc -= num_clean_aux_used

            max_alloc = max(max_alloc, total + sub_max_alloc)
            max_dealloc = min(max_dealloc, total + sub_max_dealloc)

            total += sub_total

    if max_dealloc < 0:
        raise ValueError("Deallocated more qubits than available to allocate.")

    final_any_state = any_state + total
    final_zeroed = max_alloc - total
    return total_algo_qubits, final_any_state, final_zeroed


def estimate_wires_from_resources(
    workflow: Resources,
    gate_set: set | None = None,
    config: LabsResourceConfig | None = None,
    zeroed: int = 0,
    any_state: int = 0,
):
    r"""Determine the number of auxiliary qubits needed to decompose the operators
    in a :class:`~.pennylane.estimator.resources_base.Resources` object into a specific ``gate_set`` with a given ``config``.

    Args:
        workflow (:class:`~.pennylane.estimator.resources_base.Resources`): the collection of gates and counts to be further decomposed
        gate_set (set[str] | None): A set of names (strings) of the fundamental operators to count
            throughout the quantum workflow. If not provided, the default gate set will be used,
            i.e., ``{'Toffoli', 'T', 'CNOT', 'X', 'Y', 'Z', 'S', 'Hadamard'}``.
        config (LabsResourceConfig | None): configurations for the resource estimation pipeline
        zeroed (int): The number of additional auxiliary wires, prepared in the
            zero state, that can be used as part of the decomposition.
        any_state (int): The number of additional auxiliary wires, prepared in
            any state, that can be used as part of the decomposition.

    Returns:
        tuple(int, int): The number of auxiliary qubits used as part of the decomposition. They are
        separated according to their quantum state at the end of the workflow (``any_state``, ``zeroed``).

    Raises:
        ValueError: if more qubits were deallocated than initially allocated
    """
    if config is None:
        config = LabsResourceConfig()

    if gate_set is None:
        gate_set = DefaultGateSet

    algo = workflow.algo_wires
    zeroed += workflow.zeroed_wires
    any_state += workflow.any_state_wires
    gate_counts = workflow.gate_types

    list_actions = [GateCount(gate, count) for gate, count in gate_counts.items()]

    total = 0
    max_alloc = zeroed
    max_dealloc = 0

    for action in list_actions:
        if action.gate.name in gate_set:
            continue

        resource_decomp = _get_resource_decomposition(action.gate, config)
        sub_max_alloc, sub_max_dealloc, sub_total = _estimate_auxiliary_wires(
            resource_decomp,
            action.count,
            gate_set,
            config,
            num_available_any_state_aux=algo + total + any_state,
            num_active_qubits=action.gate.num_wires,
        )

        max_alloc = max(max_alloc, total + sub_max_alloc)
        max_dealloc = min(max_dealloc, total + sub_max_dealloc)  # sub_max_dealloc < 0

        total += sub_total

    if max_dealloc < 0:
        raise ValueError("Deallocated more qubits than available to allocate.")

    final_any_state = total + any_state
    final_zeroed = max_alloc - total
    return final_any_state, final_zeroed

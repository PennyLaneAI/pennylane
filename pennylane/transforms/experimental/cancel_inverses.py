from pennylane.tape import QuantumTape
from pennylane.ops.op_math import Adjoint
from pennylane.wires import Wires

from pennylane.ops.qubit.attributes import (
    self_inverses,
    symmetric_over_all_wires,
    symmetric_over_control_wires,
)
from pennylane.transforms.optimization.optimization_utils import find_next_gate
from pennylane.transforms.experimental.transforms import transform


def _ops_equal(op1, op2):
    """Checks if two operators are equal up to class, data, hyperparameters, and wires"""
    return (
        op1.__class__ is op2.__class__
        and (op1.data == op2.data)
        and (op1.hyperparameters == op2.hyperparameters)
        and (op1.wires == op2.wires)
    )


def _are_inverses(op1, op2):
    """Checks if two operators are inverses of each other

    Args:
        op1 (~.Operator)
        op2 (~.Operator)

    Returns:
        Bool
    """
    # op1 is self-inverse and the next gate is also op1
    if op1 in self_inverses and op1.name == op2.name:
        return True

    # op1 is an `Adjoint` class and its base is equal to op2
    if isinstance(op1, Adjoint) and _ops_equal(op1.base, op2):
        return True

    # op2 is an `Adjoint` class and its base is equal to op1
    if isinstance(op2, Adjoint) and _ops_equal(op2.base, op1):
        return True

    return False


@transform
def cancel_inverses(tape):
    """Quantum function transform to remove any operations that are applied next to their
    (self-)inverses or adjoint.

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()
    operations = []
    while len(list_copy) > 0:
        current_gate = list_copy[0]
        list_copy.pop(0)

        # Find the next gate that acts on at least one of the same wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy)

        # If no such gate is found queue the operation and move on
        if next_gate_idx is None:
            operations.append(current_gate)
            continue

        # Otherwise, get the next gate
        next_gate = list_copy[next_gate_idx]

        # If either of the two flags is true, we can potentially cancel the gates
        if _are_inverses(current_gate, next_gate):
            # If the wires are the same, then we can safely remove both
            if current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx)
                continue
            # If wires are not equal, there are two things that can happen.
            # 1. There is not full overlap in the wires; we cannot cancel
            if len(Wires.shared_wires([current_gate.wires, next_gate.wires])) != len(
                current_gate.wires
            ):
                operations.append(current_gate)
                continue

            # 2. There is full overlap, but the wires are in a different order.
            # If the wires are in a different order, gates that are "symmetric"
            # over all wires (e.g., CZ), can be cancelled.
            if current_gate in symmetric_over_all_wires:
                list_copy.pop(next_gate_idx)
                continue
            # For other gates, as long as the control wires are the same, we can still
            # cancel (e.g., the Toffoli gate).
            if current_gate in symmetric_over_control_wires:
                # TODO[David Wierichs]: This assumes single-qubit targets of controlled gates
                if (
                    len(Wires.shared_wires([current_gate.wires[:-1], next_gate.wires[:-1]]))
                    == len(current_gate.wires) - 1
                ):
                    list_copy.pop(next_gate_idx)
                    continue
        # Apply gate any cases where
        # - there is no wire symmetry
        # - the control wire symmetry does not apply because the control wires are not the same
        # - neither of the flags are_self_inverses and are_inverses are true
        operations.append(current_gate)
        continue
    return [QuantumTape(operations, tape.measurements)], lambda x: x[0]

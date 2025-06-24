# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Transforms lowering gates and series of gates involving relative phases.
Transformations fist published in:

Amy, M. and Ross, N. J., “Phase-state duality in reversible circuit design”,
Physical Review A, vol. 104, no. 5, Art. no. 052602, APS, 2021. doi:10.1103/PhysRevA.104.052602.

"""
import copy
from functools import reduce

from pennylane.wires import Wires

from pennylane.operation import Operation

from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane import ops


def _find_relative_phase_toffolis(
    operations:list[Operation],
    second_target:Wires=None,
    controls:Wires=None,
    indices:list[list[Wires]]=None,
):
    """
    Searches for relative phase toffolis and finds the gates that compose them.

    Args:
        operations (list[Operation]): list of operations to search within.
        second_target (Wires): the second target wire.
        controls (Wires): the control wires.
        indices (list[list[Wires]]): list of groups of indices.

    Returns:
        list[list[int]] | list[int] | None: list (of lists for base case) of indices that point
        to gates that compose the found toffolis, or None is there is no match.

        Wires: the control wires in order.

        Wires: the first target wire.

        Wires: the second target wire.
    """
    indices = [] if indices is None else indices
    controls = [] if controls is None else controls
    first_target = []
    second_target = [] if second_target is None else second_target

    i = 0
    while i < len(operations):
        if len(controls) and second_target is not None:
            if isinstance(operations[i], ops.ControlledOp) and isinstance(operations[i].base, ops.S):
                if len(operations[i].control_wires) == 1 and operations[i].control_wires[0] in controls \
                        and operations[i].wires[-1] in controls:
                    # we fix the order of the controls now
                    controls = [operations[i].control_wires[0], operations[i].wires[-1]]
                    indices.append(i)
                elif len(operations[i].control_wires) == 2 and len(indices) == 2 \
                        and (operations[i].control_wires == Wires(controls)
                        or operations[i].control_wires[::-1] == Wires(controls)) \
                        and operations[i].wires[-1] not in controls \
                        and operations[i].wires[-1] != second_target:
                    first_target = operations[i].wires[-1]
                    indices.append(i)
            elif isinstance(operations[i], ops.MultiControlledX) and len(indices) == 3 \
                    and operations[i].wires[-1] == second_target \
                    and 3 == reduce(
                        lambda acc, wire: acc + int(wire in operations[i].control_wires),
                        operations[indices[-1]].wires,
                        0
                    ):
                indices.append(i)
            elif reduce(
                lambda acc, wire: acc + int(wire in (controls + [second_target])),
                operations[i].wires,
                0
            ):
                # we have a gate that breaks the pattern
                return None, controls, first_target, second_target
        elif isinstance(operations[i], ops.CCZ) and i < len(operations) - 3:
            # we initiate a search for each CCZ in the circuit=
            sub_indices, sub_controls, sub_first_target, sub_second_target = _find_relative_phase_toffolis(
                operations[i + 1:],
                operations[i].wires[-1],
                operations[i].control_wires,  # the control wires can be in any order
                [i],
            )
            if sub_indices is not None:
                indices.append([sub_indices[0]] + list(map(lambda index: index + i + 1, sub_indices[1:])))
                first_target.append(sub_first_target)
                second_target.append(sub_second_target)
                controls.append(sub_controls)
        i += 1

    return indices, controls, first_target, second_target

@transform
def replace_relative_phase_toffoli(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Quantum transform to replace 4-qubit relative phase toffoli gates.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]:
        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    The transform can be applied on :class:`QNode` directly.

    .. code-block:: python

        @replace_relative_phase_toffoli
        @qml.qnode(device=dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Barrier(wires=[0,1])
            qml.X(0)

            # begin relative phase 4-qubit Toffoli

            qml.CCZ(wires=[0, 1, 3])
            qml.ctrl(qml.S(wires=[1]), control=[0])
            qml.ctrl(qml.S(wires=[2]), control=[0, 1])
            qml.MultiControlledX(wires=[0, 1, 2, 3])

            # end relative phase 4-qubit Toffoli

            qml.Hadamard(wires=1)
            qml.Barrier(wires=[0,1])
            qml.X(0)
            return qml.expval(qml.Z(0))

    The relative phase 4-qubit Toffoli is then replaced before execution.

    .. details::
        :title: Usage Details

        Consider the following quantum function:

        .. code-block:: python

            def qfunc(x, y):
                qml.CCZ(wires=[0, 1, 3])
                qml.ctrl(qml.S(wires=[1]), control=[0])
                qml.ctrl(qml.S(wires=[2]), control=[0, 1])
                qml.MultiControlledX(wires=[0, 1, 2, 3])
                return qml.expval(qml.Z(0))

        The circuit before decomposition:

        >>> dev = qml.device('default.qubit', wires=4)
        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)())
            0: ─╭●─╭●─╭●─╭●─┤  <Z>
            1: ─├●─╰S─├●─├●─┤
            2: ─│─────╰S─├●─┤
            3: ─╰Z───────╰X─┤

        We can replace the relative phase 4-qubit Toffoli by running the transform:

        >>> lowered_qfunc = replace_relative_phase_toffoli(qfunc)
        >>> lowered_qnode = qml.QNode(lowered_qfunc, dev)
        >>> print(qml.draw(lowered_qnode)())

        0: ─────────────────╭●───────────╭●───────────────────────────┤  <Z>
        1: ─────────────────│─────╭●─────│─────╭●─────────────────────┤
        2: ───────╭●────────│─────│──────│─────│────────────╭●────────┤
        3: ──H──T─╰X──T†──H─╰X──T─╰X──T†─╰X──T─╰X──T†──H──T─╰X──T†──H─┤

    """
    operations = []

    for operations_indices, controls, first_target, second_target in zip(*_find_relative_phase_toffolis(tape.operations)):
        for i, gate in enumerate(tape.operations):
            if i not in operations_indices:
                operations.append(gate)
        operations = operations[:operations_indices[0]] \
            + [
                ops.Hadamard(wires=second_target),
                ops.T(wires=second_target),
                ops.CNOT(wires=[first_target, second_target]),
                ops.Adjoint(ops.T(wires=second_target)),
                ops.Hadamard(wires=second_target),
                ops.CNOT(wires=[controls[0], second_target]),
                ops.T(wires=second_target),
                ops.CNOT(wires=[controls[1], second_target]),
                ops.Adjoint(ops.T(wires=second_target)),
                ops.CNOT(wires=[controls[0], second_target]),
                ops.T(wires=second_target),
                ops.CNOT(wires=[controls[1], second_target]),
                ops.Adjoint(ops.T(wires=second_target)),
                ops.Hadamard(wires=second_target),
                ops.T(wires=second_target),
                ops.CNOT(wires=[first_target, second_target]),
                ops.Adjoint(ops.T(wires=second_target)),
                ops.Hadamard(wires=second_target),
            ] \
            + operations[operations_indices[0]:]

    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]  # pragma: no cover

    return [new_tape], null_postprocessing

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
"""Transforms for optimizing quantum circuits."""

from pennylane import apply
from pennylane.wires import Wires
from pennylane.transforms import qfunc_transform
from pennylane.operation import DiagonalOperation
from pennylane.ops.qubit import Rot

from pennylane.math import isclose

from .optimization_utils import _find_next_gate

@qfunc_transform
def commute_z_behind_controls(tape):
    """Quantum function transform to push controlled gates past diagonal gates that occur
    afterwards on the control qubit.

    Args:
        tape (.QuantumTape): A quantum tape.

    **Example**

    Consider the following quantum function.

    .. code-block:: python

        def qfunc_with_many_rots():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.T(wires=0)
            return qml.expval(qml.PauliX(0))

    In this circuit, the ``PauliZ`` and the ``CNOT`` commute; this means
    that the they can swap places, and then the ``PauliZ`` can be fused with the
    ``Hadamard`` gate if desired:

    >>> optimized_qfunc = qml.compile(pipeline=[diag_behind_controls, single_qubit_fusion])(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode)())
    0: ──Rot(3.14, 1.57, 0.785)──╭C──┤ ⟨X⟩
    1: ──────────────────────────╰X──┤
    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()

    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # Consider only two-qubit gates
        if current_gate.num_wires != 2:
            apply(current_gate)
            list_copy.pop(0)
            continue

        # Find the next gate that uses the control wire
        next_gate_idx = _find_next_gate(Wires(current_gate.wires[0]), list_copy[1:])

        # If no such gate is found (either there simply is none, or there are other gates
        # "in the way", queue the operation and move on
        if next_gate_idx is None:
            apply(current_gate)
            list_copy.pop(0)
            continue

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            # Get the next gate
            next_gate = list_copy[next_gate_idx + 1]

            # If it is not a single-qubit gate, we can't push it through, so stop.
            if len(next_gate.wires) != 1:
                break

            # Valid next gates must be diagonal, and must share the *control* wire
            if next_gate.wires[0] == current_gate.wires[0]:
                if isinstance(next_gate, DiagonalOperation):
                    list_copy.pop(next_gate_idx + 1)
                    apply(next_gate)
                # It could also be that the gate is a Rot but still diagonal if
                # the angle of the RY gate in the middle is 0
                elif isinstance(next_gate, Rot):
                    if isclose(next_gate.parameters[1], 0.0):
                        list_copy.pop(next_gate_idx + 1)
                        apply(next_gate)
                    else:
                        break
                else:
                    break

            next_gate_idx = _find_next_gate(Wires(current_gate.wires[0]), list_copy[1:])

        # After we have found all possible diagonal gates to push through the control,
        # we must still apply the original gate
        apply(current_gate)
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        apply(m)

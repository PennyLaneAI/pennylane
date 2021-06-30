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
from pennylane.ops.qubit import PauliX, RX, CNOT, CRX, Toffoli

from .optimization_utils import _find_next_gate

@qfunc_transform
def commute_x_behind_targets(tape):
    """Quantum function transform to push X rotation gates to behind the target
    gate of a controlled X rotation.

    Args:
        tape (.QuantumTape): A quantum tape.

    **Example**

    Consider the following quantum function.

    .. code-block:: python

        def qfunc_with_many_rots():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliX(0))

    In this circuit, the ``PauliX`` and the ``CNOT`` commute because the ``PauliX`` is 
    on the target of the CNOT. This means that we can push the ``PauliX`` behind
    the CNOT.

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

        # We can only do this for CNOT/CRX/Toffoli
        if not isinstance(current_gate, (CNOT, CRX, Toffoli)):
            apply(current_gate)
            list_copy.pop(0)
            continue

        # Find the next gate that uses the target wire
        next_gate_idx = _find_next_gate(Wires(current_gate.wires[-1]), list_copy[1:])

        # If no such gate is found, queue the operation and move on
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

            # Valid next gates must be an X rotation and must share the *target* wire
            if next_gate.wires[0] == current_gate.wires[-1]:
                if isinstance(next_gate, (PauliX, RX)):
                    list_copy.pop(next_gate_idx + 1)
                    apply(next_gate)
                else:
                    break

            next_gate_idx = _find_next_gate(Wires(current_gate.wires[-1]), list_copy[1:])

        # After we have pushed all possible gates through the target, apply the original gate
        apply(current_gate)
        list_copy.pop(0)

    # Queue the measurements normally
    for m in tape.measurements:
        apply(m)

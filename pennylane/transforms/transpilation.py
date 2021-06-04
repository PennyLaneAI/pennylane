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

from pennylane.ops.qubit import Hadamard, CZ
from pennylane.transforms import single_tape_transform, qfunc_transform


@qfunc_transform
def cnot_to_cz(tape):
    """Quantum function transform to apply a circuit identity that converts
    CNOT gates into CZ gates conjugated by a Hadamard on the target.

    Args:
        tape (.QuantumTape): A quantum tape.

    **Example**

    Consider the following quantum function:

    .. code-block:: python

        def bell_state():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

    Applying the transform and running the circuit results in the following:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> bell_state_with_cz = cnot_to_cz(bell_state)
    >>> qnode = qml.QNode(bell_state_with_cz, dev)
    >>> print(qml.draw(qnode)(1, 2, 3))
    0: ──H──╭Z─────┤ ⟨Z⟩
    1: ──H──╰C──H──┤

    """
    # Loop through all items in the original tape
    for op in tape.operations + tape.measurements:

        # If it's a CNOT, replace it using the circuit identity
        if op.name == "CNOT":
            wires = op.wires
            Hadamard(wires=wires[1])
            CZ(wires=[wires[1], wires[0]])
            Hadamard(wires=wires[1])

        # If it's not a CNOT, simply add it to the tape
        else:
            op.queue()

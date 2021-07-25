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
"""Transform for CNOT pair insertion method used in error mitigation."""

from pennylane import apply
from pennylane.transforms import qfunc_transform


@qfunc_transform
def cnot_pair_insertion(tape, num_pairs=0):
    """Quantum function transform that performs CNOT pair insertion.

    Anywhere we encounter a CNOT gate in the tape, we add ``num_pairs``
    additional pairs of the same CNOT.

    Args:
        qfunc (function): A quantum function.
        num_pairs (int): The number of CNOT pairs to insert at each existing CNOT.

    Returns:
        function: the transformed quantum function

    **Example**

    Suppose we have the following circuit to which we'd like to
    perform CNOT pair insertion with a single pair of CNOTs.

    .. code-block:: python

        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RZ(z, wires=2)
            qml.CNOT(wires=[2, 0])
            return qml.expval(qml.PauliZ(0))

    The number of additional CNOT pairs to include at each original
    CNOT can be specified using the ``num_pairs`` argument of the transform.

    >>> dev = qml.device('default.qubit', wires=3)
    >>> new_circuit = cnot_pair_insertion(num_pairs=1)(circuit)
    >>> qnode = qml.QNode(new_circuit, dev)
    >>> print(qml.draw(qnode)(0.5, 0.1, -0.2))
     0: ──RX(0.5)──╭C──╭C──╭C─────────────────────────────────╭X──╭X──╭X──┤ ⟨Z⟩
     1: ───────────╰X──╰X──╰X──RY(0.1)──╭C──╭C──╭C────────────│───│───│───┤
     2: ────────────────────────────────╰X──╰X──╰X──RZ(-0.2)──╰C──╰C──╰C──┤
    """
    for op in tape.operations + tape.measurements:
        apply(op)

        # Whenever a CNOT is encountered, add multiple pairs of it
        if op.name == "CNOT":
            for _ in range(2 * num_pairs):
                apply(op)

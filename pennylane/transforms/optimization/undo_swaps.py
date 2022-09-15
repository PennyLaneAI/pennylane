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

"""Transform that eliminates the swap operators by reordering the wires."""
# pylint: disable=too-many-branches
from pennylane import apply
from pennylane.transforms import qfunc_transform
from pennylane.wires import Wires
from pennylane.queuing import QueuingContext


@qfunc_transform
def undo_swaps(tape):
    """Quantum function transform to remove SWAP gates by running from right
    to left through the circuit changing the position of the qubits accordingly.

    Args:
        qfunc (function): A quantum function.

    Returns:
        function: the transformed quantum function

    **Example**

    Consider the following quantum function:

    .. code-block:: python

        def qfunc():
            qml.Hadamard(wires=0)
            qml.PauliX(wires=1)
            qml.SWAP(wires=[0,1])
            qml.SWAP(wires=[0,2])
            qml.PauliY(wires=0)
            return qml.expval(qml.PauliZ(0))

    The circuit before optimization:

    >>> dev = qml.device('default.qubit', wires=3)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)())
        0: ──H──╭SWAP──╭SWAP──Y──┤ ⟨Z⟩
        1: ──X──╰SWAP──│─────────┤
        2: ────────────╰SWAP─────┤


    We can remove the SWAP gates by running the ``undo_swap`` transform:

    >>> optimized_qfunc = undo_swaps(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode)(1, 2))
        0: ──Y──┤ ⟨Z⟩
        1: ──H──┤
        2: ──X──┤

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()
    list_copy.reverse()

    map_wires = {wire: wire for wire in tape.wires}
    gates = []

    def _change_wires(wires):
        change_wires = Wires([])
        wires = wires.toarray()
        for wire in wires:
            change_wires += map_wires[wire]
        return change_wires

    with QueuingContext.stop_recording():
        while len(list_copy) > 0:
            current_gate = list_copy[0]
            params = current_gate.parameters
            if current_gate.name != "SWAP":
                if len(params) == 0:
                    gates.append(type(current_gate)(wires=_change_wires(current_gate.wires)))
                else:
                    gates.append(
                        type(current_gate)(*params, wires=_change_wires(current_gate.wires))
                    )

            else:
                swap_wires_0, swap_wires_1 = current_gate.wires
                map_wires[swap_wires_0], map_wires[swap_wires_1] = (
                    map_wires[swap_wires_1],
                    map_wires[swap_wires_0],
                )
            list_copy.pop(0)

        gates.reverse()

        for m in tape.measurements:
            gates.append(m)

    for gate in gates:
        apply(gate)

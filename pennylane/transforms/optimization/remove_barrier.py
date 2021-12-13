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
"""Transform for removing the Barrier gate from quantum circuits."""
# pylint: disable=too-many-branches
from pennylane import apply
from pennylane.transforms import qfunc_transform


@qfunc_transform
def remove_barrier(tape):
    """Quantum function transform to remove Barrier gates.

    Args:
        qfunc (function): A quantum function.

    Returns:
        function: the transformed quantum function

    **Example**

    Consider the following quantum function:

    .. code-block:: python

        def qfunc(x, y):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.Barrier(wires=[0,1])
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

    The circuit before optimization:

    >>> dev = qml.device('default.qubit', wires=2)
    >>> qnode = qml.QNode(qfunc, dev)
    >>> print(qml.draw(qnode)(1, 2))
        0: ──H──╭||──X──┤ ⟨Z⟩
        1: ──H──╰||─────┤


    We can remove the Barrier by running the ``remove_barrier`` transform:

    >>> optimized_qfunc = remove_barrier(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode)(1, 2))
       0: ──H──X──┤ ⟨Z⟩
       1: ──H─────┤

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()

    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # Remove Barrier gate
        if current_gate.name != "Barrier":
            apply(current_gate)

        list_copy.pop(0)
        continue

    # Queue the measurements normally
    for m in tape.measurements:
        apply(m)

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
"""Transform for merging AmplitudeEmbedding gates in a quantum circuit."""
from pennylane import apply
from pennylane.transforms import qfunc_transform

from pennylane import AmplitudeEmbedding
from pennylane._device import DeviceError
from pennylane.math import kron


@qfunc_transform
def merge_amplitude_embedding(tape):
    r"""Quantum function transform to combine amplitude embedding templates that act on different qubits.

    Args:
        qfunc (function): A quantum function.

    Returns:
        function: the transformed quantum function

    **Example**

    Consider the following quantum function.

    .. code-block:: python

        def qfunc():
            qml.CNOT(wires = [0,1])
            qml.AmplitudeEmbedding([0,1], wires = 2)
            qml.AmplitudeEmbedding([0,1], wires = 3)
            return qml.state()

    The circuit before compilation will not work because of using two amplitude embedding.

    Using the transformation we can join the different amplitude embedding into a single one:

    >>> dev = qml.device('default.qubit', wires=4)
    >>> optimized_qfunc = qml.transforms.merge_amplitude_embedding(qfunc)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
    >>> print(qml.draw(optimized_qnode, show_matrices=True)())
    0: ─╭●──────────────────────┤  State
    1: ─╰X──────────────────────┤  State
    2: ─╭AmplitudeEmbedding(M0)─┤  State
    3: ─╰AmplitudeEmbedding(M0)─┤  State
    M0 =
    [0.+0.j 0.+0.j 0.+0.j 1.+0.j]

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()
    not_amplitude_embedding = []
    visited_wires = set()
    input_wires, input_vectors = [], []
    while len(list_copy) > 0:
        current_gate = list_copy[0]
        wires_set = set(current_gate.wires)

        # Check if the current gate is an AmplitudeEmbedding.
        if not isinstance(current_gate, AmplitudeEmbedding):
            not_amplitude_embedding.append(current_gate)
            list_copy.pop(0)
            visited_wires = visited_wires.union(wires_set)
            continue

        # Check the qubits have not been used.
        if len(visited_wires.intersection(wires_set)) > 0:
            raise DeviceError(
                f"Operation {current_gate.name} cannot be used after other Operation applied in the same qubit "
            )
        input_wires.append(current_gate.wires)
        input_vectors.append(current_gate.parameters[0])
        list_copy.pop(0)
        visited_wires = visited_wires.union(wires_set)

    if len(input_wires) > 0:
        final_wires = input_wires[0]
        final_vector = input_vectors[0]

        # Merge all parameters and qubits into a single one.
        for w, v in zip(input_wires[1:], input_vectors[1:]):
            final_vector = kron(final_vector, v)
            final_wires = final_wires + w

        AmplitudeEmbedding(final_vector, wires=final_wires)

    for gate in not_amplitude_embedding:
        apply(gate)

    # Queue the measurements normally
    for m in tape.measurements:
        apply(m)

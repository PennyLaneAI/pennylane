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
from typing import Sequence, Callable

from pennylane.transforms.core import transform
from pennylane.tape import QuantumTape
from pennylane import AmplitudeEmbedding
from pennylane._device import DeviceError
from pennylane.math import flatten, reshape


@transform
def merge_amplitude_embedding(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):
    r"""Quantum function transform to combine amplitude embedding templates that act on different qubits.

    Args:
        tape (QuantumTape): A quantum tape.

    Returns:
        qnode (pennylane.QNode) or qfunc or tuple[List[.QuantumTape], function]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.

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
    >>> print(qml.draw(optimized_qnode)())
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
    input_wires, input_vectors, input_batch_size = [], [], []
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
        input_batch_size.append(current_gate.batch_size)
        list_copy.pop(0)
        visited_wires = visited_wires.union(wires_set)

    if len(input_wires) > 0:
        final_wires = input_wires[0]
        final_vector = input_vectors[0]
        final_batch_size = input_batch_size[0]

        # Merge all parameters and qubits into a single one.
        for w, v, b in zip(input_wires[1:], input_vectors[1:], input_batch_size[1:]):
            final_vector = final_vector[..., :, None] * v[..., None, :]
            final_batch_size = final_batch_size or b
            final_wires = final_wires + w

            if final_batch_size:
                final_vector = reshape(final_vector, (final_batch_size, -1))
            else:
                final_vector = flatten(final_vector)

        AmplitudeEmbedding(final_vector, wires=final_wires)

    new_operations = []
    for gate in not_amplitude_embedding:
        new_operations.append(gate)

    new_tape = QuantumTape(new_operations, tape.measurements, shots=tape.shots)
    new_tape._qfunc_output = tape._qfunc_output  # pylint: disable=protected-access

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing

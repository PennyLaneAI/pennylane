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

from collections.abc import Sequence
from copy import copy
from functools import lru_cache, partial

import pennylane as qp
from pennylane import AmplitudeEmbedding
from pennylane.exceptions import DeviceError, TransformError
from pennylane.math import flatten, is_abstract, reshape
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn


@transform
def merge_amplitude_embedding(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Quantum function transform to combine amplitude embedding templates that act on different qubits.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit (QNode or quantum function).

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]: The transformed circuit as described in :func:`qp.transform <pennylane.transform>`.


    **Example**

    You can apply the transform directly on a :class:`QNode`:

    .. code-block:: python

        import pennylane as qp

        dev = qp.device('default.qubit', wires=4)

        @qp.transforms.merge_amplitude_embedding
        @qp.qnode(device=dev)
        def circuit():
            qp.CNOT(wires = [0,1])
            qp.AmplitudeEmbedding([0, 1], wires = 2)
            qp.AmplitudeEmbedding([0, 1], wires = 3)
            return qp.state()

    >>> print(qp.draw(circuit)())
    0: ─╭●───┤ ╭State
    1: ─╰X───┤ ├State
    2: ─╭|Ψ⟩─┤ ├State
    3: ─╰|Ψ⟩─┤ ╰State
    >>> circuit()
    array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
           0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    """
    new_operations = []
    visited_wires = set()
    input_wires, input_vectors, input_batch_size = [], [], []
    for current_gate in tape.operations:
        wires_set = set(current_gate.wires)

        # Check if the current gate is an AmplitudeEmbedding.
        if not isinstance(current_gate, AmplitudeEmbedding):
            new_operations.append(current_gate)
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
        visited_wires = visited_wires.union(wires_set)

    if len(input_wires) > 0:
        final_wires = input_wires[0]
        final_vector = input_vectors[0]
        final_batch_size = input_batch_size[0]

        # Merge all parameters and qubits into a single one.
        for w, v, b in zip(input_wires[1:], input_vectors[1:], input_batch_size[1:], strict=True):
            final_vector = final_vector[..., :, None] * v[..., None, :]
            final_batch_size = final_batch_size or b
            final_wires = final_wires + w

            if final_batch_size:
                final_vector = reshape(final_vector, (final_batch_size, -1))
            else:
                final_vector = flatten(final_vector)

        with QueuingManager.stop_recording():
            new_operations.insert(0, AmplitudeEmbedding(final_vector, wires=final_wires))

    new_tape = tape.copy(operations=new_operations)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing

# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains a function to extract a single tape from a QNode"""

import pennylane as qml


def construct_tape(qnode, level="user"):
    """Constructs the tape for a designated stage in the transform program.

    Args:
        qnode (QNode): the qnode we want to get the tapes and post-processing for.
        level (None, str, int, slice): Specifies which stage of the QNode's transform program to use for tape construction.

            - ``None`` or ``"device"``: Uses the entire transformation pipeline.
            - ``"top"``: Ignores transformations and returns the original tape as defined.
            - ``"user"``: Includes transformations that are manually applied by the user.
            - ``"gradient"``: Extracts the gradient-level tape.
            - ``int``: Can also accept an integer, corresponding to a number of transforms in the program.
            - ``slice``: Can also accept a ``slice`` object to select an arbitrary subset of the transform program.

    Returns:
        tape (QuantumScript): a quantum circuit.

    Raises:
        ValueError: if the ``level`` argument corresponds to more than one tape.

    .. seealso:: :func:`pennylane.workflow.get_transform_program` to inspect the contents of the transform program for a specified level.

    **Example**

    .. code-block:: python

        @qml.qnode(qml.device("default.qubit", shots=10))
        def circuit(x):
            qml.RandomLayers(qml.numpy.array([[1.0, 2.0]]), wires=(0,1))
            qml.RX(x, wires=0)
            qml.RX(-x, wires=0)
            qml.SWAP((0,1))
            qml.X(0)
            qml.X(0)
            return qml.expval(qml.X(0) + qml.Y(0))

    >>> tape = qml.workflow.construct_tape(circuit)(0.5)
    >>> tape.circuit
    [RandomLayers(tensor([[1., 2.]], requires_grad=True), wires=[0, 1]),
    RX(0.5, wires=[0]),
    RX(-0.5, wires=[0]),
    SWAP(wires=[0, 1]),
    X(0),
    X(0),
    expval(X(0) + Y(0))]

    """

    def wrapper(*args, **kwargs):

        batch, _ = qml.workflow.construct_batch(qnode, level)(*args, **kwargs)

        if len(batch) > 1:
            raise ValueError(
                "Level requested corresponds to more than one tape. Please use `qml.workflow.construct_batch` instead for this level."
            )

        return batch[0]

    return wrapper

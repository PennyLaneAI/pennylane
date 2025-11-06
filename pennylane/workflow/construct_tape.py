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
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .construct_batch import construct_batch

if TYPE_CHECKING:
    from pennylane.tape import QuantumScript

    from .qnode import QNode


def construct_tape(
    qnode: QNode, level: str | int | slice | None = "user"
) -> Callable[..., QuantumScript]:
    """Constructs the tape for a designated stage in the transform program.

    .. warning::

        Using ``level=None`` is deprecated and will be removed in a future release.
        Please use ``level='device'`` to include all transforms.

    Args:
        qnode (QNode): the qnode we want to get the tapes and post-processing for.
        level (None, str, int, slice): An indication of what transforms to apply before drawing.
            Check :func:`~.workflow.get_transform_program` for more information on the allowed values and usage details of
            this argument.

    Returns:
        tape (QuantumScript): a quantum circuit.

    Raises:
        ValueError: if the ``level`` argument corresponds to more than one tape.

    .. seealso:: :func:`pennylane.workflow.get_transform_program` to inspect the contents of the transform program for a specified level.

    **Example**

    .. code-block:: python

        @partial(qml.set_shots, shots=10)
        @qml.qnode(qml.device("default.qubit"))
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

        batch, _ = construct_batch(qnode, level)(*args, **kwargs)

        if len(batch) > 1:
            raise ValueError(
                "Level requested corresponds to more than one tape. Please use `qml.workflow.construct_batch` instead for this level."
            )

        return batch[0]

    return wrapper

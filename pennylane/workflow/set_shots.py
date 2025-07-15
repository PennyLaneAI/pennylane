# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the set_shots decorator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple

from .qnode import QNode

# Sentinel value to distinguish between "not provided" and "explicitly None"
_DECORATOR_MODE = object()

if TYPE_CHECKING:
    from pennylane.measurements import Shots


def set_shots(
    qnode: QNode | object = _DECORATOR_MODE,
    shots: Optional[Shots | int | Sequence[int | Tuple[int, int]]] = None,
) -> QNode | Callable[[QNode], QNode]:
    """Transform used to set or update a circuit's shots.

    Args:
        qnode (QNode): The QNode to be modified. If not provided, returns a decorator.
        shots (None or int or Sequence[int] or Sequence[tuple[int, int]] or pennylane.shots.Shots): The
            number of shots (or a shots vector) that the transformed circuit will execute.
            This specification will override any shots value previously associated
            with the circuit or QNode during execution.

    Returns:
        QNode or callable: The transformed QNode with updated shots, or a decorator function
        if qnode is not provided.

    There are three ways to specify shot values (see :func:`qml.measurements.Shots <pennylane.measurements.Shots>` for more details):

    * The value ``None``: analytic mode, no shots
    * A positive integer: a fixed number of shots
    * A sequence consisting of either positive integers or a tuple-pair of positive integers of the form ``(shots, copies)``

    **Examples**

    Set the number of shots as a decorator:

    .. code-block:: python

        @qml.set_shots(shots=2)
        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.sample(qml.Z(0))

    Run the circuit:

    >>> circuit()
    array([1., -1.])

    Update the shots in-line for an existing circuit:

    >>> new_circ = qml.set_shots(circuit, shots=(4, 10)) # shot vector
    >>> new_circ()
    (array([-1.,  1., -1.,  1.]), array([ 1.,  1.,  1., -1.,  1.,  1., -1., -1.,  1.,  1.]))

    """
    # When used as decorator with arguments: @set_shots(shots=...)
    # This happens when qnode parameter is not provided (decorator mode)
    if qnode is _DECORATOR_MODE:

        def decorator(qnode_func):
            return set_shots(qnode_func, shots)

        return decorator

    # When called directly with a QNode
    if isinstance(qnode, QNode):
        return qnode.update_shots(shots)

    # If qnode is not a QNode (including explicit None), raise error
    raise ValueError("set_shots can only be applied to QNodes")

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

from collections.abc import Callable
from typing import TYPE_CHECKING, overload

from .qnode import QNode

if TYPE_CHECKING:
    from pennylane.measurements import ShotsLike


# Sentinel value to detect when shots parameter is not provided
_SHOTS_NOT_PROVIDED = object()


@overload
def set_shots(qnode: QNode, shots: ShotsLike) -> QNode: ...
@overload
def set_shots(shots: ShotsLike) -> Callable[[QNode], QNode]: ...
def set_shots(*args, shots: ShotsLike = _SHOTS_NOT_PROVIDED):
    """Transform used to set or update a circuit's shots.

    Args:
        qnode (QNode): The QNode to transform. If not provided, ``set_shots`` can be used as a decorator directly.
        shots (None or int or Sequence[int] or Sequence[tuple[int, int]] or pennylane.shots.Shots): The
            number of shots (or a shots vector) that the transformed circuit will execute.

    Returns:
        QNode or callable: The transformed QNode with updated shots, or a wrapper function
        if qnode is not provided.

    There are three ways to specify shot values (see :func:`qml.measurements.Shots <pennylane.measurements.Shots>` for more details):

    * The value ``None``: analytic mode, no shots
    * A positive integer: a fixed number of shots
    * A sequence consisting of either positive integers or a tuple-pair of positive integers of the form ``(shots, copies)``

    **Examples**

    Set the number of shots as a decorator (positional argument):

    .. code-block:: python

        @qml.set_shots(500)
        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.expval(qml.Z(0))

    Set analytic mode as a decorator (positional argument):

    .. code-block:: python

        @qml.set_shots(None)
        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.expval(qml.Z(0))

    Set the number of shots as a decorator (keyword argument):

    .. code-block:: python

        @qml.set_shots(shots=2)
        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.sample(qml.Z(0))

    Set analytic mode as a decorator (keyword argument):

    .. code-block:: python

        @qml.set_shots(shots=None)
        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.expval(qml.Z(0))

    Run the circuit:

    >>> circuit()
    array([1., -1.])

    Update the shots in-line for an existing circuit:

    >>> new_circ = qml.set_shots(circuit, shots=(4, 10)) # shot vector
    >>> new_circ()
    (array([-1.,  1., -1.,  1.]), array([ 1.,  1.,  1., -1.,  1.,  1., -1., -1.,  1.,  1.]))

    Set analytic mode in-line for an existing circuit:

    >>> analytic_circ = qml.set_shots(circuit, shots=None)
    >>> analytic_circ()
    0.5403023058681398
    """
    # Keyword-only case: @set_shots(shots=500) or @set_shots(shots=None)
    if len(args) == 0 and shots is not _SHOTS_NOT_PROVIDED:
        return _set_shots_dispatch(shots)

    if len(args) == 1 and shots is not _SHOTS_NOT_PROVIDED:
        # Direct application: set_shots(qnode, shots=500) or set_shots(qnode, shots=None)
        return _apply_shots_to_qnode(args[0], shots)
    if len(args) == 1 and shots is _SHOTS_NOT_PROVIDED:
        # Positional decorator: @set_shots(500) or @set_shots(None)
        return _set_shots_dispatch(args[0])
    if len(args) == 2 and shots is _SHOTS_NOT_PROVIDED:
        return _apply_shots_to_qnode(*args)
    raise ValueError(f"Invalid arguments to set_shots: {args=}, {shots=}")


def _set_shots_dispatch(shots_value: ShotsLike) -> Callable[[QNode], QNode]:
    """Default case: @set_shots(500) - positional shots value"""

    def positional_decorator(qnode_func: QNode) -> QNode:
        return _apply_shots_to_qnode(qnode_func, shots_value)

    return positional_decorator


def _apply_shots_to_qnode(qnode: QNode, shots: ShotsLike) -> QNode:
    """Handle direct application to a QNode: set_shots(qnode, shots=500)"""
    if not isinstance(qnode, QNode):
        raise ValueError(f"set_shots can only be applied to QNodes, not {type(qnode)} provided.")
    return qnode.update_shots(shots)

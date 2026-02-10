# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the 'marker' utility for marking PennyLane objects."""

from collections.abc import Callable

from .qnode import QNode


def marker(obj: QNode | None = None, level: str | None = None) -> QNode | Callable:
    """Mark a location in a compilation pipeline for easy access with inspection utilities.

    Args:
        obj (QNode | None): The QNode containing the compilation pipeline to be marked.
            If not provided, the function is assumed to used as a decorator.
        level (str | None): The label for the level in the compilation pipeline to mark.

    Returns:
        QNode | Callable:

    Raises:
        ValueError: The 'level' argument must be provided.

    **Example:**

    .. code-block:: python

        @qml.marker("after-merge-rotations")
        @qml.transforms.merge_rotations
        @qml.marker("after-cancel-inverses")
        @qml.transforms.cancel_inverses
        @qml.marker("nothing-applied")
        @qml.qnode(qml.device("null.qubit"))
        def c():
            qml.RX(0.5, 0)
            qml.H(0)
            qml.H(0)
            qml.RX(0.5, 0)
            return qml.probs()

    >>> print(c.compile_pipeline)
    CompilePipeline(
       ├─▶ nothing-applied
      [1] cancel_inverses(),
       ├─▶ after-cancel-inverses
      [2] merge_rotations()
       └─▶ after-merge-rotations
    )
    >>> print(c.compile_pipeline.markers)
    ['nothing-applied', 'after-cancel-inverses', 'after-merge-rotations']

    These markers can then be picked up by a few of our inspectibility features.
    For example, we can verify that the Hadamard gates cancel,

    >>> print(qml.specs(c, level="after-cancel-inverses")())
    Device: null.qubit
    Device wires: None
    Shots: Shots(total=None)
    Level: after-cancel-inverses
    <BLANKLINE>
    Resource specifications:
      Total wire allocations: 1
      Total gates: 2
      Circuit depth: 2
    <BLANKLINE>
      Gate types:
        RX: 2
    <BLANKLINE>
      Measurements:
        probs(all wires): 1

    and that the rotation gates merge,

    >>> print(qml.draw(c, level="after-merge-rotations")())
    0: ──RX(1.00)─┤  Probs

    """

    if isinstance(obj, QNode) and level is not None:
        obj.compile_pipeline.add_marker(level)
        return obj

    # NOTE: In order to use as decorator: @qml.marker(level="blah")
    if isinstance(obj, str):
        level = obj
        obj = None

    if obj is None and level is not None:

        def decorator(qnode: QNode) -> QNode:
            qnode.compile_pipeline.add_marker(level)
            return qnode

        return decorator

    raise ValueError("marker requires a 'level' argument.")

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

    **Example:**

    .. code-block:: python

        @qml.marker("after-cancel-inverses")
        @qml.transforms.cancel_inverses
        @qml.marker("after-merge-rotations")
        @qml.transforms.merge_rotations
        @qml.marker("nothing-applied")
        @qml.qnode(qml.device("null.qubit"))
        def c():
            return qml.state()

    >>> print(c.compile_pipeline)
    CompilePipeline(
       ├─▶ nothing-applied
      [1] merge_rotations(),
       ├─▶ after-merge-rotations
      [2] cancel_inverses()
       └─▶ after-cancel-inverses
    )
    >>> print(c.markers)
    ['nothing-applied', 'after-merge-rotations', 'after-cancel-inverses']

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

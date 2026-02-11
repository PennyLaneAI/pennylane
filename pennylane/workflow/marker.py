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


def marker(obj: QNode | None = None, label: str | None = None) -> QNode | Callable:
    """Register a checkpoint within a compilation pipeline for inspection.

    Args:
        obj (QNode | None): The ``QNode`` containing the compilation pipeline to be marked.
            If ``None``, this function acts as a decorator for a ``QNode``.
        label (str | None): A descriptive label for this specific stage in the compilation process.
            Check :func:`~.workflow.get_transform_program` for more information on the allowed values and usage details of this argument.


    Returns:
        QNode | Callable: The marked ``QNode`` or a decorator function if ``obj`` is not provided.

    Raises:
        ValueError: The 'label' argument must be provided.

    .. seealso::
        :meth:`~.CompilePipeline.add_marker`

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

    We can then inspect our user transformations to see our markers,

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

    These markers are then recognized by a few of our inspectibility features.
    For example, we can verify that the Hadamard gates cancel using :func:`~.specs`,

    >>> print(qml.specs(c, level="after-cancel-inverses")()) # or level=1
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

    and that the rotation gates merge using :func:`~.draw`,

    >>> print(qml.draw(c, level="after-merge-rotations")()) # or level=2
    0: ──RX(1.00)─┤  Probs

    or even display our circuit before any transformations,

    >>> print(qml.draw(c, level="nothing-applied")()) # or level=0
    0: ──RX(0.50)──H──H──RX(0.50)─┤  Probs

    """

    if isinstance(obj, QNode) and label is not None:
        obj.compile_pipeline.add_marker(label)
        return obj

    # NOTE: In order to use as decorator: @qml.marker(label="blah")
    if isinstance(obj, str):
        label = obj
        obj = None

    if obj is None and label is not None:

        def decorator(qnode: QNode) -> QNode:
            qnode.compile_pipeline.add_marker(label)
            return qnode

        return decorator

    raise ValueError("marker requires a 'label' argument.")

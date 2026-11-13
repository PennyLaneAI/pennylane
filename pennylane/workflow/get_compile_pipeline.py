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
"""Contains a function for getting the compile pipeline of a given QNode."""

from __future__ import annotations

from collections.abc import Sequence

from pennylane.core.transforms import CompilePipeline

from ._levels import is_single_level, levels_to_slices, pipeline_at_slice
from .qnode import QNode


def unwrap_qjit_qnode(qjit, *, fn_name: str) -> QNode:
    """Return the QNode underlying a qjit'd workflow, raising a helpful error otherwise.

    ``fn_name`` is the name of the public function to report in the error message.
    """
    # pylint: disable=import-outside-toplevel
    # Have to import locally to prevent circular imports as well as accounting for Catalyst not being installed
    try:
        from catalyst import QJIT
    except ImportError as exc:  # pragma: no cover
        raise ImportError(f"Catalyst must be installed to use {fn_name}.") from exc

    # Unwrap the original QNode if any transforms have been applied
    if isinstance(qjit, QJIT) and isinstance(qjit.original_function, QNode):
        return qjit.original_function

    raise ValueError(f"{fn_name} can only be applied to a qjit'd QNode, instead got: {qjit}")


def get_compile_pipeline(
    qnode,
    level: str | int | slice | Sequence[str | int] = "user",
) -> CompilePipeline | list[CompilePipeline]:
    """Extract :class:`~.CompilePipeline`\\ (s) from a :func:`~.qjit`'d :class:`~.QNode`.

    .. note::

        The device preprocessing passes are represented by a single ``Device Preprocessing``
        placeholder, since the device applies its own transforms when the QNode is executed. The
        placeholder cannot be applied to a QNode.

    Args:
        qnode (:class:`~catalyst.jit.QJIT`): The qjit'd QNode to get the compile pipeline for.
        level (str | int | slice | Sequence[str | int]): The portion(s) of the
            :class:`~.CompilePipeline` to retrieve. Defaults to ``"user"``.

            ``level`` accepts the following types:

            * ``str``: One of the presets listed below, or the ``label`` of a :func:`~.marker`.
              A marker label includes all transforms up to the location of that marker.
            * ``int``: The number of transforms to include from the start of the pipeline. Must
              be non-negative, e.g. ``level=0`` is empty and ``level=2`` contains the first two
              transforms (``[0]`` and ``[1]``). The device preprocessing passes are
              denoted by a single level, placed after all user-applied transforms.
            * ``slice``: A range of transforms, using standard Python slicing with non-negative
              ``int`` bounds and a positive step, e.g. ``level=slice(1, 3)`` contains transforms
              ``[1]`` and ``[2]``. A slice without a stop, such as
              ``slice(1, None)``, includes device preprocessing passes. To exclude them, give an
              explicit stop, e.g. ``slice(1, 2)`` for a pipeline with two user-applied
              transforms. A step greater than ``1`` skips transforms, and emits a warning, as
              does a slice that selects no transforms.
            * ``Sequence``: A non-empty ``list``, ``tuple``, or similar containing ``int`` or
              ``str`` values. One compile pipeline is returned per level, in ascending order. If
              the levels are not in ascending order, or contain duplicates, they are sorted and
              deduplicated, with a warning. The ``"all"`` and ``"all-user"`` presets are not
              allowed inside a ``Sequence``, since they already correspond to multiple levels.

            The markers placed at the start or the end of the selected transforms are kept,
            except with a ``slice`` with a step greater than ``1``, which drops all markers.

            The following ``str`` presets are provided for ease of use:

            * ``"top"``: Equivalent to ``0``; an empty pipeline, before any transforms are applied.
            * ``"user"``: Only the transforms manually applied by the user.
            * ``"device"``: The user-applied transforms followed by the device preprocessing,
              i.e. the full pipeline as prepared for device execution.
            * ``"all-user"``: One pipeline for each stage of compilation with respect to the
              user-applied transforms, from ``"top"`` up to ``"user"``.
            * ``"all"``: One pipeline for each stage of compilation with respect to the
              user-applied transforms and the device preprocessing, from ``"top"`` up to
              ``"device"``.

    Returns:
        CompilePipeline | list[CompilePipeline]: The :class:`~.CompilePipeline` at the requested
        level. If ``level`` is a ``Sequence``, ``"all-user"``, or ``"all"``, a list containing
        one :class:`~.CompilePipeline` per level is returned instead.

    **Example:**

    .. code-block:: python

        from pennylane.workflow import get_compile_pipeline

        dev = qp.device("null.qubit", wires=1)

        @qp.qjit(capture=True)
        @qp.transforms.merge_rotations
        @qp.marker("checkpoint")
        @qp.transforms.cancel_inverses
        @qp.qnode(dev)
        def circuit(angle):
            qp.RX(angle, wires=0)
            qp.H(0)
            qp.H(0)
            qp.RX(angle, wires=0)
            return qp.expval(qp.Z(0))

    By default, the transforms manually applied by the user are retrieved:

    >>> print(get_compile_pipeline(circuit)) # or level="user"
    CompilePipeline(
      [0] cancel_inverses(),
       ├─▶ checkpoint
      [1] merge_rotations()
    )

    .. details::
        :title: Usage Details

        To get the full compile pipeline used during execution on the device, specify
        ``level="device"``. The device preprocessing is appended after the user-applied
        transforms, and counts as a single level, matching the ``"Device Preprocessing"``
        level reported by :func:`~.analyze`:

        >>> print(get_compile_pipeline(circuit, level="device"))
        CompilePipeline(
          [0] cancel_inverses(),
           ├─▶ checkpoint
          [1] merge_rotations(),
          [2] Device Preprocessing
        )

        We can also retrieve the compile pipeline up to a specific stage indicated by the name
        given to a :func:`~.marker`, which in our example is the ``"checkpoint"`` marker:

        >>> print(get_compile_pipeline(circuit, level="checkpoint"))
        CompilePipeline(
          [0] cancel_inverses()
           └─▶ checkpoint
        )

        If ``level`` is ``"top"`` or ``0``, an empty compile pipeline is returned:

        >>> print(get_compile_pipeline(circuit, level=0))
        CompilePipeline()
        >>> print(get_compile_pipeline(circuit, level="top"))
        CompilePipeline()

        The ``level`` can also be an integer, corresponding to the number of transforms to
        retrieve from the compile pipeline:

        >>> print(get_compile_pipeline(circuit, level=1))
        CompilePipeline(
          [0] cancel_inverses()
           └─▶ checkpoint
        )

        A ``slice`` extracts a specific range of transforms from the compile pipeline. For
        example, we can retrieve only the second transform with:

        >>> print(get_compile_pipeline(circuit, level=slice(1, 2)))
        CompilePipeline(
           ├─▶ checkpoint
          [0] merge_rotations()
        )

        Note that the ``"checkpoint"`` marker is still in the pipeline, as it marks a location
        in the pipeline rather than being attached to a particular transform.

        To retrieve several compile pipelines at once, provide a ``Sequence`` of levels. Unlike
        a ``slice``, which returns a single range of transforms, one compile pipeline is
        returned per entry, each containing all transforms up to that level:

        >>> for pipeline in get_compile_pipeline(circuit, level=[1, 2]):
        ...     print(pipeline)
        CompilePipeline(
          [0] cancel_inverses()
           └─▶ checkpoint
        )
        CompilePipeline(
          [0] cancel_inverses(),
           ├─▶ checkpoint
          [1] merge_rotations()
        )

        The ``"all-user"`` and ``"all"`` presets return the compile pipeline at every stage
        of compilation:

        >>> len(get_compile_pipeline(circuit, level="all-user"))
        3
        >>> len(get_compile_pipeline(circuit, level="all"))
        4

    """
    if isinstance(qnode, QNode):
        raise ValueError(
            "get_compile_pipeline can only be applied to a qjit'd QNode. For a QNode that is not "
            "qjit'd, use qnode.compile_pipeline for the user transforms, or "
            "qp.workflow.construct_batch to inspect the tapes at the gradient or device level."
        )
    pipeline = unwrap_qjit_qnode(qnode, fn_name="get_compile_pipeline").compile_pipeline
    pipelines = [
        pipeline_at_slice(pipeline, level_slice)
        for level_slice in levels_to_slices(level, pipeline)
    ]
    return pipelines[0] if is_single_level(level) else pipelines

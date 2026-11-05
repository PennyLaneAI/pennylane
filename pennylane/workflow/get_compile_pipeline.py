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

import warnings
from collections.abc import Sequence
from numbers import Integral

from pennylane.core.transforms import CompilePipeline
from pennylane.core.transforms.compile_pipeline import _DEVICE_PREPROCESSING

from .qnode import QNode


def _is_qjit(obj):
    """A method checking whether the object is compiled by catalyst.qjit"""
    return obj.__class__.__name__ == "QJIT" and hasattr(obj, "original_function")


def _is_int_level(level) -> bool:
    """Whether ``level`` is an integer level, such as an ``int`` or a NumPy integer, excluding
    ``bool``."""
    return isinstance(level, Integral) and not isinstance(level, bool)


def _is_single_level(level) -> bool:
    """Whether ``level`` selects a single compile pipeline, rather than a list of them."""
    if isinstance(level, str):
        return level not in ("all-user", "all")
    return isinstance(level, slice) or _is_int_level(level)


def _device_level(pipeline: CompilePipeline) -> int:
    """The level number right after the device preprocessing, which counts as a single transform
    after the user-applied transforms of ``pipeline``.

    This is also the number of transforms, including the device preprocessing."""
    return len(pipeline) + 1


def _level_to_number(level: str | int, pipeline: CompilePipeline) -> int:
    """Convert a single ``str`` or ``int`` level to its level number, which is the number of
    transforms of ``pipeline``, followed by the device preprocessing, applied at that level."""
    device_level = _device_level(pipeline)

    if isinstance(level, str):
        presets = {"top": 0, "user": len(pipeline), "device": device_level}
        if level in presets:
            return presets[level]
        if level == "gradient":
            raise ValueError("The level 'gradient' is no longer supported.")
        found_level = pipeline.get_marker_level(level)
        if found_level is None:
            raise ValueError(
                f"Marker name '{level}' not found in the compile pipeline. Builtin levels are "
                f"'top', 'user', 'device', 'all-user', and 'all'. Markers are {pipeline.markers}."
            )
        return found_level

    if _is_int_level(level):
        level = int(level)
        if not 0 <= level <= device_level:
            raise ValueError(
                f"The 'level' argument is out of bounds, got {level}. Levels range from 0 to "
                f"{device_level}, the device level."
            )
        return level

    raise ValueError(
        f"Invalid level {level!r}, expected an int, str, slice, or a Sequence of int and str."
    )


def _levels_to_numbers(
    level: str | int | Sequence[str | int], pipeline: CompilePipeline
) -> list[int]:
    """Convert ``level``, other than a ``slice``, to the sorted list of level numbers it selects.

    A ``Sequence`` of levels is sorted, with duplicates removed.
    """
    # The ``isinstance`` checks avoid elementwise comparisons, e.g. with NumPy arrays
    if isinstance(level, str) and level == "all-user":
        # From level 0 up to the user level, included
        return list(range(len(pipeline) + 1))
    if isinstance(level, str) and level == "all":
        # From level 0 up to the device level, included
        return list(range(_device_level(pipeline) + 1))
    if isinstance(level, str) or _is_int_level(level):
        return [_level_to_number(level, pipeline)]

    # ``bytes`` and ``bytearray`` are a ``Sequence`` of ``int``, but not a sequence of levels
    if not isinstance(level, Sequence) or isinstance(level, (bytes, bytearray)):
        raise ValueError(
            f"Invalid level {level!r}, expected an int, str, slice, or a Sequence of int and str."
        )
    if not level:
        raise ValueError(f"Invalid level {level!r}, which does not select any level.")

    for lvl in level:
        if not (isinstance(lvl, str) or _is_int_level(lvl)) or lvl in ("all-user", "all"):
            raise ValueError(
                f"Invalid level {lvl!r} in the level Sequence, expected an int, or a str other "
                "than 'all-user' and 'all'."
            )

    level_numbers = [_level_to_number(lvl, pipeline) for lvl in level]
    sorted_level_numbers = sorted(set(level_numbers))
    if level_numbers != sorted_level_numbers:
        warnings.warn(
            "The 'level' argument has been sorted to be in ascending order with no duplicate "
            "levels.",
            UserWarning,
        )
    return sorted_level_numbers


def _levels_to_slices(level, pipeline: CompilePipeline) -> list[slice]:
    """Convert ``level`` to a list of slices of ``pipeline``, followed by the device
    preprocessing, with one slice per compile pipeline that ``level`` selects."""
    if isinstance(level, slice):
        return [level]
    return [slice(0, number) for number in _levels_to_numbers(level, pipeline)]


def _pipeline_at_slice(pipeline: CompilePipeline, level_slice: slice) -> CompilePipeline:
    """Return the compile pipeline selected by ``level_slice`` from ``pipeline`` followed by the
    device preprocessing."""
    if any(bound is not None and bound < 0 for bound in (level_slice.start, level_slice.stop)):
        raise ValueError(
            f"Invalid level {level_slice!r}, the bounds of a slice must be non-negative."
        )
    if level_slice.step is not None and level_slice.step < 1:
        raise ValueError(
            f"Invalid level {level_slice!r}, the step of a slice must be positive, since the "
            "device preprocessing is applied after the other transforms."
        )
    if level_slice.step is not None and level_slice.step > 1:
        warnings.warn(
            f"The level {level_slice!r} has a step greater than 1, so it skips transforms of the "
            "compile pipeline.",
            UserWarning,
        )

    # Slicing the user-applied transforms directly keeps a marker placed after the last of them
    if level_slice.indices(len(pipeline) + 1)[1] <= len(pipeline):
        return pipeline[level_slice]
    return (pipeline + _DEVICE_PREPROCESSING)[level_slice]


def get_compile_pipeline(
    qnode,
    level: str | int | slice | Sequence[str | int] = "user",
) -> CompilePipeline | list[CompilePipeline]:
    """Extract :class:`~.CompilePipeline`\\ (s) from a :class:`~.QNode`.

    .. note::

        The device preprocessing is represented by a single ``Device Preprocessing`` placeholder,
        since the device applies its own transforms when the QNode is executed. The placeholder
        cannot be applied to a QNode. For a QNode that is not qjit'd, the tapes produced by the
        individual gradient and device preprocessing transforms can be inspected with
        :func:`~.workflow.construct_batch`, whose levels count each of these transforms
        separately.

    Args:
        qnode (:class:`~.QNode` | :class:`~catalyst.jit.QJIT`): The QNode, or qjit'd QNode, to
            get the compile pipeline for.
        level (str | int | slice | Sequence[str | int]): The portion(s) of the
            :class:`~.CompilePipeline` to retrieve. Defaults to ``"user"``.

            ``level`` accepts the following types:

            * ``str``: One of the presets listed below, or the ``label`` of a :func:`~.marker`.
              A marker label includes all transforms up to the location of that marker.
            * ``int``: The number of transforms to include from the start of the pipeline. Must
              be non-negative, e.g. ``level=0`` is empty and ``level=2`` contains the first two
              transforms (``[0]`` and ``[1]``). The device preprocessing counts as a single
              level, placed after all user-applied transforms.
            * ``slice``: A range of transforms, using standard Python slicing with non-negative
              bounds and a positive step, e.g. ``level=slice(1, 3)`` contains transforms ``[1]``
              and ``[2]``. A step greater than ``1`` skips transforms, and raises a warning.
            * ``Sequence``: A non-empty ``list``, ``tuple``, or similar containing ``int`` or
              ``str`` values. One compile pipeline is returned per level, in ascending order. If
              the levels are not in ascending order, or contain duplicates, they are sorted and
              deduplicated, with a warning. A ``Sequence`` rather than an arbitrary iterable is
              required so that its length and ordering can be validated up front, without
              consuming it.
              The ``"all"`` and ``"all-user"`` presets are not allowed inside a ``Sequence``,
              since they already correspond to multiple levels.

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
    if _is_qjit(qnode):
        qnode = qnode.original_function
    if not isinstance(qnode, QNode):
        raise ValueError(
            "Can only retrieve the compile pipeline of a QNode or a qjit'd QNode, instead got: "
            f"{qnode}."
        )

    pipeline = qnode.compile_pipeline
    pipelines = [
        _pipeline_at_slice(pipeline, level_slice)
        for level_slice in _levels_to_slices(level, pipeline)
    ]
    return pipelines[0] if _is_single_level(level) else pipelines

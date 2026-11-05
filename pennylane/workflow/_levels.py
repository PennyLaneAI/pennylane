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
"""Internal helpers resolving the ``level`` argument of :func:`~.workflow.get_compile_pipeline`,
:func:`~.analyze`, :func:`~pennylane.estimate` and :func:`~.track`.

.. warning::

    This module is intended for internal use only and may change or be removed in future releases.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from numbers import Integral

from pennylane.core.transforms import CompilePipeline
from pennylane.core.transforms.compile_pipeline import DEVICE_PREPROCESSING, _DevicePreprocessing

_INVALID_LEVEL_TYPES = "expected an int, str, slice, or a Sequence of int and str."


def is_int_level(level) -> bool:
    """Whether ``level`` is an integer level, such as an ``int`` or a NumPy integer, excluding
    ``bool``."""
    return isinstance(level, Integral) and not isinstance(level, bool)


def is_single_level(level) -> bool:
    """Whether ``level`` selects a single compile pipeline, rather than a list of them."""
    if isinstance(level, str):
        return level not in ("all-user", "all")
    return isinstance(level, slice) or is_int_level(level)


def unpack_single_entry(level):
    """Return the entry of a ``Sequence`` of levels with a single entry, and ``level`` otherwise."""
    if isinstance(level, Sequence) and not isinstance(level, str) and len(level) == 1:
        return level[0]
    return level


def _has_device_preprocessing(pipeline: CompilePipeline) -> bool:
    """Whether the device preprocessing is applied after ``pipeline``, which is not the case for an
    informative pipeline, since the device does not execute it."""
    return not pipeline.is_informative


def device_level(pipeline: CompilePipeline) -> int:
    """The level number right after the device preprocessing, which counts as a single transform
    after the user-applied transforms of ``pipeline``.

    This is also the number of transforms, including the device preprocessing. Without device
    preprocessing, the device level is the same as the user level."""
    return len(pipeline) + 1 if _has_device_preprocessing(pipeline) else len(pipeline)


def level_to_number(level: str | int, pipeline: CompilePipeline) -> int:
    """Convert a single ``str`` or ``int`` level to its level number, which is the number of
    transforms of ``pipeline``, followed by the device preprocessing, applied at that level."""
    dev_level = device_level(pipeline)

    if isinstance(level, str):
        presets = {"top": 0, "user": len(pipeline), "device": dev_level}
        if level in presets:
            return presets[level]
        if level in ("gradient", "all-mlir"):
            raise ValueError(f"The level '{level}' is no longer supported.")
        found_level = pipeline.get_marker_level(level)
        if found_level is None:
            raise ValueError(
                f"Marker name '{level}' not found in the compile pipeline. Builtin levels are "
                f"'top', 'user', 'device', 'all-user', and 'all'. Markers are {pipeline.markers}."
            )
        return found_level

    if is_int_level(level):
        level = int(level)
        if not 0 <= level <= dev_level:
            raise ValueError(
                f"The 'level' argument is out of bounds, got {level}. Levels range from 0 to "
                f"{dev_level}, the device level."
            )
        return level

    raise ValueError(f"Invalid level {level!r}, {_INVALID_LEVEL_TYPES}")


def _level_numbers(level: str | int | Sequence[str | int], pipeline: CompilePipeline) -> list[int]:
    """Convert ``level``, other than a ``slice``, to the numbers of the levels it selects, in the
    order of a ``Sequence``."""
    # The ``isinstance`` checks avoid elementwise comparisons, e.g. with NumPy arrays
    if isinstance(level, str) and level == "all-user":
        # From level 0 up to the user level, included
        return list(range(len(pipeline) + 1))
    if isinstance(level, str) and level == "all":
        # From level 0 up to the device level, included
        return list(range(device_level(pipeline) + 1))
    if isinstance(level, str) or is_int_level(level):
        return [level_to_number(level, pipeline)]

    # ``bytes`` and ``bytearray`` are a ``Sequence`` of ``int``, but not a sequence of levels
    if not isinstance(level, Sequence) or isinstance(level, (bytes, bytearray)):
        raise ValueError(f"Invalid level {level!r}, {_INVALID_LEVEL_TYPES}")
    if not level:
        raise ValueError(f"Invalid level {level!r}, which does not select any level.")

    for lvl in level:
        if not (isinstance(lvl, str) or is_int_level(lvl)) or lvl in ("all-user", "all"):
            raise ValueError(
                f"Invalid level {lvl!r} in the level Sequence, expected an int, or a str other "
                "than 'all-user' and 'all'."
            )

    return [level_to_number(lvl, pipeline) for lvl in level]


def levels_to_numbers(
    level: str | int | Sequence[str | int], pipeline: CompilePipeline
) -> list[int]:
    """Convert ``level``, other than a ``slice``, to the sorted list of level numbers it selects.

    A ``Sequence`` of levels is sorted, with duplicates removed.
    """
    level_numbers = _level_numbers(level, pipeline)
    sorted_level_numbers = sorted(set(level_numbers))
    if level_numbers != sorted_level_numbers:
        warnings.warn(
            "The 'level' argument has been sorted to be in ascending order with no duplicate "
            "levels.",
            UserWarning,
        )
    return sorted_level_numbers


def levels_to_slices(level, pipeline: CompilePipeline) -> list[slice]:
    """Convert ``level`` to a list of slices of ``pipeline``, followed by the device
    preprocessing, with one slice per compile pipeline that ``level`` selects."""
    if isinstance(level, slice):
        return [level]
    return [slice(0, number) for number in levels_to_numbers(level, pipeline)]


def validate_level(level, pipeline: CompilePipeline) -> None:
    """Raise an error if ``level`` is not a valid level of ``pipeline``."""
    if isinstance(level, slice):
        _validate_slice(level, pipeline)
    else:
        _level_numbers(level, pipeline)


def _validate_slice(level_slice: slice, pipeline: CompilePipeline) -> None:
    """Raise an error if ``level_slice`` is not a valid ``slice`` level of ``pipeline``."""
    start, stop, step = level_slice.start, level_slice.stop, level_slice.step
    if any(bound is not None and not is_int_level(bound) for bound in (start, stop, step)):
        raise ValueError(
            f"Invalid level {level_slice!r}, the bounds and step of a slice must be int or None."
        )
    if any(bound is not None and bound < 0 for bound in (start, stop)):
        raise ValueError(
            f"Invalid level {level_slice!r}, the bounds of a slice must be non-negative."
        )
    if start is not None and start > device_level(pipeline):
        raise ValueError(
            f"Invalid level {level_slice!r}, the start of a slice must be at most "
            f"{device_level(pipeline)}, the device level."
        )
    if step is not None and step < 1:
        raise ValueError(
            f"Invalid level {level_slice!r}, the step of a slice must be positive, since the "
            "device preprocessing is applied after the other transforms."
        )


def _slice_keeping_markers(pipeline: CompilePipeline, start: int, stop: int) -> CompilePipeline:
    """Return the transforms of ``pipeline`` from ``start`` to ``stop``, with the markers at both
    ends. Unlike ``pipeline[start:stop]``, a marker at ``stop`` is kept even if ``stop`` is not the
    end of ``pipeline``."""
    sliced = CompilePipeline(list(pipeline)[start:stop])
    for label in pipeline.markers:
        if start <= (level := pipeline.get_marker_level(label)) <= stop:
            sliced.add_marker(label, level - start)
    return sliced


def pipeline_at_slice(pipeline: CompilePipeline, level_slice: slice) -> CompilePipeline:
    """Return the compile pipeline selected by ``level_slice`` from ``pipeline`` followed by the
    device preprocessing."""
    _validate_slice(level_slice, pipeline)
    full_pipeline = (
        pipeline + DEVICE_PREPROCESSING if _has_device_preprocessing(pipeline) else pipeline
    )

    start, stop, step = level_slice.indices(len(full_pipeline))
    if step == 1:
        return _slice_keeping_markers(full_pipeline, start, stop)

    message = (
        f"The level {level_slice!r} has a step greater than 1, so it skips transforms of the "
        "compile pipeline."
    )
    if pipeline.markers:
        message += " Markers are dropped from the result."
    warnings.warn(message, UserWarning)

    # Avoid a second warning from ``CompilePipeline.__getitem__`` about the dropped markers
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Slicing a CompilePipeline that contains markers",
            category=UserWarning,
        )
        return full_pipeline[level_slice]


def split_device_preprocessing(pipeline: CompilePipeline) -> tuple[CompilePipeline, bool]:
    """Split a compile pipeline returned by :func:`pipeline_at_slice` into the transforms applied
    before the device preprocessing, and whether the device preprocessing is included."""
    # The slice has a positive step, so the device preprocessing can only be the last transform
    if pipeline and isinstance(pipeline[-1], _DevicePreprocessing):
        return _slice_keeping_markers(pipeline, 0, len(pipeline) - 1), True
    return pipeline, False

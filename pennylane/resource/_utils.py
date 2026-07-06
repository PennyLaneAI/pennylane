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
"""
Common utility functions for processing resource information

.. warning::

    This module is intended for internal use only and may change or be removed in future releases.
"""

import warnings
from collections.abc import Iterable
from functools import partial, wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pennylane.core.transforms import CompilePipeline


def unwrap_partial(fn):
    """Return the base callable and arguments bound by nested ``functools.partial`` wrappers."""
    args = ()
    kwargs = {}
    while isinstance(fn, partial):
        args = fn.args + args
        kwargs = {**(fn.keywords or {}), **kwargs}
        fn = fn.func
    return fn, args, kwargs


def apply_partial_args(fn, args, kwargs):
    """Return a callable that prepends partial-bound arguments to call-time arguments."""
    if not args and not kwargs:
        return fn

    @wraps(fn)
    def wrapper(*call_args, **call_kwargs):
        return fn(*args, *call_args, **{**kwargs, **call_kwargs})

    return wrapper


def get_last_tape_transform_level(compile_pipeline: "CompilePipeline") -> int:
    """Helper function to get the last level from a :class:`CompilePipeline` which is a tape
    transform and not an MLIR pass.

    .. note::

        The returned level includes an implicit level 0 which corresponds to the original circuit before any transforms.

    .. warning::

        This function is intended for internal use only and may change or be removed in future releases.

    Args:
        compile_pipeline: The :class:`CompilePipeline` to analyse, which may contain both user-applied tape transforms
            and MLIR passes

    Returns:
        int: The last level which is a tape transform and not an MLIR pass, or 0 if there are no tape transforms
    """
    # Find the seam where transforms end and MLIR passes begin
    # If the pass name is None, it indicates a transform which is NOT also a Catalyst pass
    for i, trans in reversed(list(enumerate(compile_pipeline))):
        if trans.pass_name is None:
            #  Add 1 to account for the implicit "Before Tape Transforms" at level=0
            return i + 1
    return 0


def preprocess_level_input(  # pylint: disable=too-many-branches
    level: str | int | slice | list[int | str],
    marker_to_level: dict[str, int],
    pipeline_len: int,
    num_tape_levels: int,
) -> list[int]:
    """Preprocesses a level input to always return a sorted list of integers.

    .. warning::

        This function is intended for internal use only and may change or be removed in future releases.

    Args:
        level (str | int | slice | iter[int | str]): The level input to preprocess
        marker_to_level (dict[str, int]): Mapping from marker names to their associated level numbers.
            Note that this should already account for any inserted lowering pass.
        pipeline_len (int): The length of the compile pipeline (number of transforms and passes)
        num_tape_levels (int): The number of tape levels in the compile pipeline (including the implicit level 0)
    Returns:
        list[int]: The preprocessed level input

    Examples:
        >>> marker_to_level = {"before": 0, "after": 1}
        >>> preprocess_level_input("before", marker_to_level, 2, 1)
        [0]
        >>> preprocess_level_input([0, "after"], marker_to_level, 2, 1)
        [0, 1]
        >>> preprocess_level_input(slice(0, 2), marker_to_level, 2, 1)
        [0, 1]
    """
    # Account for "Before MLIR passes" level
    total_levels = pipeline_len + 1

    if num_tape_levels > 1:
        # Account for an additional "Before Tape Transforms" level
        total_levels += 1

    default_level_map = {
        "all": list(range(0, total_levels)),
        "all-mlir": list(range(num_tape_levels, total_levels)),
        "user": [total_levels - 1],
    }
    if level in default_level_map:
        return keyword_map[level]

    if isinstance(level, (int, str)):
        level = [level]
    elif isinstance(level, slice):
        level = list(range(level.start or 0, level.stop, level.step or 1))
    else:
        level = list(level)

    # Convert marker names to the associated level number
    for i, lvl in enumerate(level):
        if isinstance(lvl, str):
            if lvl not in marker_to_level:
                raise ValueError(f"Marker name '{lvl}' not found in the compile pipeline.")
            level[i] = marker_to_level[lvl]
        elif isinstance(lvl, int):
            if lvl < 0 or lvl >= total_levels:
                raise ValueError(
                    "The 'level' argument to qp.specs for QJIT'd QNodes is out of bounds, "
                    f"got {lvl}."
                )
        else:
            raise ValueError(f"Invalid level '{lvl}' in level list, expected int or str.")

    level_sorted = sorted(set(level))
    if level != level_sorted:
        warnings.warn(
            "The 'level' argument to qp.specs for QJIT'd QNodes has been sorted to be in ascending "
            "order with no duplicate levels.",
            UserWarning,
        )

    return level_sorted


def make_level_name_unique(level_name: str, existing_names: Iterable[str]) -> str:
    """Helper function to make a level name unique by appending a suffix if necessary.

    .. warning::

        This function is intended for internal use and may be subject to change without deprecation.

    Args:
        level_name (str): The original level name
        existing_names (Iterable[str]): The set of existing level names to check against

    Returns:
        str: A unique level name

    Example:
        >>> existing = {"cancel-inverses", "merge-rotations", "cancel-inverses-2"}
        >>> make_level_name_unique("cancel-inverses", existing)
        'cancel-inverses-3'
    """
    unique_name = level_name
    counter = 1
    while unique_name in existing_names:
        counter += 1
        unique_name = f"{level_name}-{counter}"
    return unique_name


def get_marker_level_map(compile_pipeline: "CompilePipeline") -> dict[str, int]:
    """Helper function to get a mapping from marker names to their associated level numbers.

    .. warning::

        This function is intended for internal use and may be subject to change without deprecation.

    """
    marker_to_level: dict[str, int] = {}

    num_tape_levels = get_last_tape_transform_level(compile_pipeline)
    if num_tape_levels != 0:
        # Account for the "Before Tape Transforms" tape at level 0
        num_tape_levels += 1

    for marker in compile_pipeline.markers:
        lvl = compile_pipeline.get_marker_level(marker)
        marker_to_level[marker] = lvl

        # Account for the MLIR lowering pass if necessary
        if 0 < num_tape_levels <= lvl:
            marker_to_level[marker] += 1

    return marker_to_level

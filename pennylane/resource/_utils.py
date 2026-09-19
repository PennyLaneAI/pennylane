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


def preprocess_level_input(
    level: str | int | list[int | str],
    compile_pipeline: "CompilePipeline",
) -> list[int]:
    """Preprocesses a level input to always return a sorted list of integers.

    .. warning::

        This function is intended for internal use only and may change or be removed in future releases.

    Args:
        level (str | int | iter[int | str]): The level input to preprocess
        compile_pipeline ("CompilePipeline"): The compilation pipeline (the sequence of transforms)
    Returns:
        list[int]: The preprocessed level input

    Examples:
        >>> pipeline = qp.CompilePipeline([qp.transforms.cancel_inverses, qp.transforms.merge_rotations])
        >>> pipeline.add_marker("before", 0)
        >>> pipeline.add_marker("after", 1)
        >>> preprocess_level_input("before", pipeline)
        [0]
        >>> preprocess_level_input([0, "after"], pipeline)
        [0, 1]
        >>> preprocess_level_input("all", pipeline)
        [0, 1, 2]
    """
    if trans := [pass_ for pass_ in compile_pipeline if pass_.pass_name is None]:
        raise ValueError(
            f"Specs encountered the following tape transforms: {trans}."
            " Tape transforms are no longer supported by specs."
        )

    marker_to_level = get_marker_level_map(compile_pipeline)

    # Account for "Before MLIR passes" level
    total_levels = len(compile_pipeline) + 1

    default_level_map = {
        "top": [0],
        "user": [len(compile_pipeline)],
        "all": list(range(0, total_levels)),
    }
    if isinstance(level, str) and level in default_level_map:
        return default_level_map[level]

    # Convert single entries to a list for uniform processing
    if isinstance(level, (int, str)):
        level = [level]
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

    for marker in compile_pipeline.markers:
        lvl = compile_pipeline.get_marker_level(marker)
        marker_to_level[marker] = lvl

    return marker_to_level

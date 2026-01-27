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

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec

from pennylane.workflow import construct_execution_config, marker
from pennylane.workflow._setup_transform_program import _setup_transform_program

if TYPE_CHECKING:
    from pennylane.devices.execution_config import ExecutionConfig
    from pennylane.transforms.core import CompilePipeline
    from pennylane.workflow import QNode

P = ParamSpec("P")


def _has_terminal_expansion_pair(compile_pipeline: CompilePipeline) -> bool:
    """Checks if the compile pipeline ends with a expansion + transform pair."""
    return (
        len(compile_pipeline) > 1
        and getattr(compile_pipeline[-1], "expand_transform", None) == compile_pipeline[-2]
    )


def _find_level(program: CompilePipeline, level: str) -> int:
    """Retrieve the numerical level associated to a marker."""
    found_levels = []
    for idx, t in enumerate(program):
        if t.tape_transform == marker.tape_transform:
            found_level = t.args[0] if t.args else t.kwargs["level"]
            found_levels.append(found_level)

            if found_level == level:
                return idx
    raise ValueError(
        f"level {level} not found in compile pipeline. "
        "Builtin options are 'top', 'user', 'device', and 'gradient'."
        f" Custom levels are {found_levels}."
    )


def _resolve_level(
    level: str | int | slice,
    full_pipeline: CompilePipeline,
    num_user: int,
    config: ExecutionConfig,
) -> slice:
    """Resolve level to a slice."""

    if level == "top":
        level = slice(0, 0)
    elif level == "user":
        level = slice(0, num_user)
    elif level == "gradient":
        level = slice(0, num_user + int(hasattr(config.gradient_method, "expand_transform")))
    elif level == "device":
        # Captures everything: user + gradient + device + final
        level = slice(0, None)
    elif isinstance(level, str):
        level = slice(0, _find_level(full_pipeline, level))
    elif isinstance(level, int):
        level = slice(0, level)

    return level


def get_compile_pipeline(
    qnode: QNode,
    level: str | int | slice = "device",
) -> Callable[P, CompilePipeline]:
    """Extract a compile pipeline at a designated level.

    Args:
        qnode (QNode): The QNode to get the compile pipeline for.
        level (str, int, slice): An indication of what transforms to use from the full compile pipeline.

    """

    if not isinstance(level, (int, slice, str)):
        raise ValueError(
            f"'level={level}' of type '{type(level)}' is not supported. Please provide an integer, slice or a string as input."
        )

    @wraps(qnode)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> CompilePipeline:
        # Get full compile pipeline
        resolved_config = construct_execution_config(qnode, resolve=True)(*args, **kwargs)
        outer_pipeline, inner_pipeline = _setup_transform_program(qnode.device, resolved_config)
        full_compile_pipeline = qnode.compile_pipeline + outer_pipeline + inner_pipeline

        num_user = len(qnode.compile_pipeline)
        if qnode.compile_pipeline.has_final_transform:
            # Ignore final transforms for now, will be re-added later if needed
            num_user -= 2 if _has_terminal_expansion_pair(qnode.compile_pipeline) else 1
            if (
                level in {"gradient", "device"}
                or isinstance(level, int)
                and level
                >= num_user + int(hasattr(resolved_config.gradient_method, "expand_transform"))
            ):
                raise ValueError(
                    f"Cannot retrieve compile pipeline at requested level '{level}' due to final transforms being present."
                )

        # Slice out relevant section
        level_slice: slice = _resolve_level(level, full_compile_pipeline, num_user, resolved_config)
        resolved_pipeline = full_compile_pipeline[level_slice]

        # Add back final transforms to resolved pipeline if required
        if qnode.compile_pipeline.has_final_transform and level == "user":
            final_transform_start = (
                -2 if _has_terminal_expansion_pair(qnode.compile_pipeline) else -1
            )
            resolved_pipeline += qnode.compile_pipeline[final_transform_start:]

        return resolved_pipeline

    return wrapper

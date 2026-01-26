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
from typing import TYPE_CHECKING, Literal, ParamSpec, TypeAlias

from pennylane.workflow import construct_execution_config
from pennylane.workflow._setup_transform_program import _setup_transform_program

if TYPE_CHECKING:
    from pennylane.devices.execution_config import ExecutionConfig
    from pennylane.transforms.core import CompilePipeline
    from pennylane.workflow import QNode

P = ParamSpec("P")
PipelineLevel: TypeAlias = Literal["top", "user", "gradient", "device"] | int | slice


def _has_terminal_expansion_pair(compile_pipeline: CompilePipeline) -> bool:
    """Checks if the compile pipeline ends with a expansion + transform pair."""
    return (
        len(compile_pipeline) > 1
        and getattr(compile_pipeline[-1], "expand_transform", None) == compile_pipeline[-2]
    )


def _resolve_level(qnode: QNode, config: ExecutionConfig, level: PipelineLevel) -> slice:
    """Resolve level to a slice."""
    num_user = len(qnode.compile_pipeline)

    # Ignore final transforms for now, will be re-added later if needed
    if qnode.compile_pipeline.has_final_transform:
        # Remove pair if expansion + transform exists
        num_user -= 2 if _has_terminal_expansion_pair(qnode.compile_pipeline) else 1

    if level == "top":
        level = slice(0, 0)
    elif level == "user":
        level = slice(0, num_user)
    elif level == "gradient":
        if qnode.compile_pipeline.has_final_transform:
            raise ValueError(
                "Cannot retrieve compile pipeline if 'level=gradient' is requested and a final transform is being used."
            )
        level = slice(0, num_user + int(hasattr(config.gradient_method, "expand_transform")))
    elif level == "device":
        # Captures everything: user + gradient + device + final
        level = slice(0, None)
    elif isinstance(level, str):
        raise NotImplementedError
    elif isinstance(level, int):
        level = slice(0, level)

    return level


def get_compile_pipeline(
    qnode: QNode,
    level: PipelineLevel = "device",
) -> Callable[P, CompilePipeline]:
    """Extract a compile pipeline at a designated level.

    Args:
        qnode (QNode): The QNode to get the compile pipeline for.
        level (str, int, slice): An indication of what transforms to use from the full compile pipeline.

    """

    if not isinstance(level, (int, slice, str)):
        raise ValueError(
            f"'level={level}' is not recognized. Please provide an integer, slice or a string as input."
        )

    @wraps(qnode)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> CompilePipeline:
        # Get full compile pipeline
        config = construct_execution_config(qnode, resolve=True)(*args, **kwargs)
        outer, inner = _setup_transform_program(qnode.device, config)
        full_compile_pipeline = qnode.compile_pipeline + outer + inner

        # Slice out relevant section
        level_slice: slice = _resolve_level(qnode, config, level)
        resolved_pipeline = full_compile_pipeline[level_slice]

        # Add back final transforms to resolved pipeline
        if qnode.compile_pipeline.has_final_transform and level in {"user", "gradient"}:
            final_transform_start = (
                -2 if _has_terminal_expansion_pair(qnode.compile_pipeline) else -1
            )
            resolved_pipeline += qnode.compile_pipeline[final_transform_start:]

        return resolved_pipeline

    return wrapper

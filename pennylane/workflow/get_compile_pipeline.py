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

from pennylane.transforms.core import CompilePipeline
from pennylane.workflow import construct_execution_config

if TYPE_CHECKING:
    from pennylane.devices.execution_config import ExecutionConfig
    from pennylane.workflow import QNode

P = ParamSpec("P")
PipelineLevel: TypeAlias = Literal["top", "user", "device", "gradient"] | int | slice


def resolve_level(qnode: QNode, level: PipelineLevel) -> slice:
    """Resolve level to a slice."""
    num_user = len(qnode.compile_pipeline)

    if level == "device":
        level = slice(0, None)
    elif level == "top":
        level = slice(0, 0)
    elif level == "user":
        level = slice(0, num_user)
    elif level == "gradient":
        if qnode.compile_pipeline.has_final_transform:
            raise ValueError(
                "Cannot retrieve compile pipeline if 'level=gradient' is requested and a final transform is being used."
            )
        raise NotImplementedError
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
        config = construct_execution_config(qnode, resolve=True)(*args, **kwargs)
        level_slice: slice = resolve_level(qnode, level)
        full_compile_pipeline = qnode.compile_pipeline + qnode.device.preprocess_transforms(config)
        return full_compile_pipeline[level_slice]

    return wrapper

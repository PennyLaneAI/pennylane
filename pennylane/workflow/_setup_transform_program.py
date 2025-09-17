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
"""Contains a function for setting up the inner and outer transform programs for execution of a QNode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from cachetools import Cache, LRUCache

from pennylane.math import Interface
from pennylane.transforms.convert_to_numpy_parameters import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram, transform

from ._cache_transform import _cache_transform

if TYPE_CHECKING:
    from pennylane.devices import Device
    from pennylane.devices.execution_config import ExecutionConfig


def _prune_dynamic_transform(
    outer_transform: TransformProgram, inner_transform: TransformProgram
) -> None:
    """Ensure a single ``dynamic_one_shot`` transform is applied.

    Sometimes device preprocess contains a ``mid_circuit_measurements`` transform, which will
    be added to the inner transform program. If the user then applies a ``dynamic_one_shot``
    manually, it will duplicate the ``mid_circuit_measurements`` transform. This function ensures
    that there is only one ``dynamic_one_shot`` transform in the outer and inner transform
    programs combined.

    """

    all_transforms = outer_transform + inner_transform
    type_to_keep = 0
    if any("mid_circuit_measurements" in str(t) for t in all_transforms):
        type_to_keep = 2
    elif any("dynamic_one_shot" in str(t) for t in all_transforms):
        type_to_keep = 1

    if type_to_keep == 0:
        return

    inner_contains_one_shot = inner_transform.prune_dynamic_transform(type_to_keep)
    if inner_contains_one_shot:
        type_to_keep = 0
    outer_transform.prune_dynamic_transform(type_to_keep)


def _setup_transform_program(
    device: Device,
    resolved_execution_config: ExecutionConfig,
    cache: Cache | dict | Literal["auto"] | bool = None,
    cachesize: int = 10000,
) -> tuple[TransformProgram, TransformProgram]:
    """Sets-up the outer and inner transform programs for execution.

    Args:
        user_transform_program (TransformProgram): the user's transform program
        device (Device): a Pennylane device
        resolved_execution_config (ExecutionConfig): the resolved execution config
        cache (None, bool, dict, Cache): Whether to cache evaluations. This can result in
        a significant reduction in quantum evaluations during gradient computations. Defaults to ``None``.
        cachesize (int): The size of the cache. Defaults to 10000.

    Returns:
        tuple[TransformProgram, TransformProgram]: tuple containing the outer and inner transform programs.
    """

    device_transform_program = device.preprocess_transforms(resolved_execution_config)

    outer_transform_program = TransformProgram()
    inner_transform_program = TransformProgram()

    # Add the gradient expand to the program if necessary
    if getattr(resolved_execution_config.gradient_method, "expand_transform", False):
        outer_transform_program.add_transform(
            transform(resolved_execution_config.gradient_method.expand_transform),
            **resolved_execution_config.gradient_keyword_arguments,
        )
    if resolved_execution_config.use_device_gradient:
        outer_transform_program += device_transform_program
    else:
        inner_transform_program += device_transform_program

    # Making sure dynamic_one_shot occurs at most once between the inner and outer transform programs
    _prune_dynamic_transform(outer_transform_program, inner_transform_program)

    # If caching is desired but an explicit cache is not provided, use an ``LRUCache``.
    if cache == "auto":
        if resolved_execution_config.derivative_order == 1:
            cache = None
        else:
            cache = True
    if cache is True:
        cache = LRUCache(maxsize=cachesize)

    # Ensure that ``cache`` is not a Boolean to simplify downstream code.
    elif cache is False:
        cache = None

    # changing this set of conditions causes a bunch of tests to break.
    interface_data_supported = (
        (not resolved_execution_config.convert_to_numpy)
        or resolved_execution_config.interface is Interface.NUMPY
        or resolved_execution_config.gradient_method == "backprop"
    )
    if not interface_data_supported:
        inner_transform_program.add_transform(convert_to_numpy_parameters)
    if cache is not None:
        inner_transform_program.add_transform(_cache_transform, cache=cache)

    return outer_transform_program, inner_transform_program

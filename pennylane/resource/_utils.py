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

from collections.abc import Iterable
from functools import partial, wraps

from .resource import CircuitSpecs


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


def check_no_tape_transforms(compile_pipeline) -> None:
    """Raise an error if ``compile_pipeline`` contains tape transforms, which have no
    ``pass_name``."""
    if trans := [pass_ for pass_ in compile_pipeline if pass_.pass_name is None]:
        raise ValueError(
            f"Encountered the following tape transforms: {trans}."
            " Tape transforms are no longer supported."
        )


def build_circuit_specs(original_qnode, resources, level) -> CircuitSpecs:
    """Assemble the ``CircuitSpecs`` describing a qjit'd QNode at a given level."""
    return CircuitSpecs(
        resources=resources,
        shots=original_qnode.shots,
        device_name=original_qnode.device.name,
        num_device_wires=(
            len(original_qnode.device.wires) if original_qnode.device.wires is not None else None
        ),
        level=level,
    )


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

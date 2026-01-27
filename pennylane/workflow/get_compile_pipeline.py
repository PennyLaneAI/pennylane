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

from functools import wraps
from typing import TYPE_CHECKING, ParamSpec

from pennylane.workflow import construct_execution_config, marker
from pennylane.workflow._setup_transform_program import _setup_transform_program

if TYPE_CHECKING:
    from collections.abc import Callable

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

            - ``"device"``: Retrieves the entire compile pipeline used for device execution.
            - ``"top"``: Returns an empty compile pipeline.
            - ``"user"``: Retrieves a compile pipeline containing manually applied user transformations.
            - ``"gradient"``: Retrieves a compile pipeline that includes user transformations and any relevant gradient transformations.
            - ``str``: Can also accept a string corresponding to the name of a marker that was manually added to the compile pipeline.
            - ``int``: Can also accept an integer, corresponding to a number of transforms in the program. ``level=0`` corresponds to the start of the program.
            - ``slice``: Can also accept a ``slice`` object to select an arbitrary subset of the compile pipeline.

    Raises:
        ValueError: If a final transform is applied to the qnode with a level that goes deeper than the gradient level of the compile pipeline.

    **Example:**

    Consider this simple circuit,

    .. code-block:: python

        dev = qml.device("default.qubit")

        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        def circuit():
            qml.H(0)
            qml.RX(1, wires=0)
            qml.RX(1, wires=0)
            qml.H(0)
            return qml.expval(qml.Z(0))

    We can retrieve the compile pipeline used during execution with,

    >>> get_compile_pipeline(circuit)() # or level="device"
    CompilePipeline(cancel_inverses, merge_rotations, defer_measurements, decompose, device_resolve_dynamic_wires, validate_device_wires, validate_measurements, _conditional_broadcast_expand, no_sampling)

    or use the ``level`` argument to inspect specific stages of the pipeline.

    >>> get_compile_pipeline(circuit, level="user")()
    CompilePipeline(cancel_inverses, merge_rotations)

    .. details::
        :title: Usage Details

        Consider the circuit below which has user applied transforms, a checkpoint marker and uses the parameter-shift gradient method,

        .. code-block:: python

            dev = qml.device("default.qubit")

            @qml.transforms.merge_rotations
            @qml.marker("checkpoint")
            @qml.transforms.cancel_inverses
            @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs={"shifts": np.pi / 4})
            def circuit(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.Z(0))

        By default, without specifying a ``level`` we will get the full compile pipeline that is used during execution on this device.
        Note that this can also be retrieved by manually specifying ``level="device"``,

        >>> get_compile_pipeline(circuit)(3.14)
        CompilePipeline(cancel_inverses, marker, merge_rotations, _expand_transform_param_shift, defer_measurements, decompose, device_resolve_dynamic_wires, validate_device_wires, validate_measurements, _conditional_broadcast_expand)

        As can be seen above, this not only includes the two transforms we manually applied, but also a set of transforms used by the device in order to execute the circuit.
        The ``"user"`` level will retrieve the portion of the compile pipeline that was manually applied by the user to the qnode,

        >>> get_compile_pipeline(circuit, level="user")(3.14)
        CompilePipeline(cancel_inverses, marker, merge_rotations)

        The ``"gradient"`` level builds on top of this to then add any relevant gradient transforms,

        >>> get_compile_pipeline(circuit, level="gradient")(3.14)
        CompilePipeline(cancel_inverses, marker, merge_rotations, _expand_transform_param_shift)

        which in this case is ``_expand_transform_param_shift``, a transform that expands all trainable operations
        to a state where the parameter shift transform can operate on them.

        We can use ``qml.marker`` to further subdivide our compile pipeline into stages,

        >>> get_compile_pipeline(circuit, level="checkpoint")(3.14)
        CompilePipeline(cancel_inverses)

        If ``"top"`` or ``0`` are specified, an empty compile pipeline will be returned,

        >>> get_compile_pipeline(circuit, level=0)(3.14)
        CompilePipeline()
        >>> get_compile_pipeline(circuit, level="top")(3.14)
        CompilePipeline()

        Integer levels correspond to the number of transforms to retrieve from the compile pipeline,

        >>> get_compile_pipeline(circuit, level=3)(3.14)
        CompilePipeline(cancel_inverses, marker, merge_rotations)

        Slice levels enable you to extract a specific range of transformations in the compile pipeline. For example, we can retrieve the second to fourth transform by using a slice,

        >>> get_compile_pipeline(circuit, level=slice(1,4))(3.14)
        CompilePipeline(marker, merge_rotations, _expand_transform_param_shift)

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

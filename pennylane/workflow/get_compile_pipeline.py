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

from pennylane.core.transforms import CompilePipeline
from pennylane.workflow import construct_execution_config
from pennylane.workflow._setup_transform_program import _setup_transform_program

if TYPE_CHECKING:
    from collections.abc import Callable

    from pennylane.devices.execution_config import ExecutionConfig
    from pennylane.workflow import QNode

P = ParamSpec("P")


def _is_qjit(obj):
    """A method checking whether the object is compiled by catalyst.qjit"""
    return obj.__class__.__name__ == "QJIT" and hasattr(obj, "user_function")


def _find_level(program: CompilePipeline, level: str) -> int:
    """Retrieve the numerical level associated to a marker."""
    found_level = program.get_marker_level(level)
    if found_level is not None:
        return found_level
    raise ValueError(
        f"level {level} not found in compile pipeline. "
        "Builtin options are 'top', 'user', 'device', and 'gradient'."
        f" Custom levels are {program.markers}."
    )


def _resolve_level(
    level: str | int | slice,
    full_pipeline: CompilePipeline,
    num_user: int,
    config: ExecutionConfig | None,
) -> slice:
    """Resolve level to a slice."""

    if level == "top":
        level = slice(0, 0)
    elif level == "user":
        level = slice(0, num_user)
    elif level == "gradient":
        level = slice(0, num_user + int(hasattr(config.gradient_method, "expand_transform")))
    elif level == "device":
        # Captures everything: user + gradient + device
        level = slice(0, None)
    elif isinstance(level, str):
        level = slice(0, _find_level(full_pipeline, level))
    elif isinstance(level, int):
        level = slice(0, level)

    return level


def get_compile_pipeline(
    qnode: QNode,
    level: str | int | slice = "user",
) -> Callable[P, CompilePipeline]:
    """Extract :class:`~.CompilePipeline`\\ (s) from a :class:`~.QNode`.

    Args:
        qnode (:class:`~catalyst.jit.QJIT`): The (qjit'd) QNode to get the compile pipeline for.
        level (str | int | slice | Sequence[str | int]): The portion(s) of the
            :class:`~.CompilePipeline` to retrieve. Defaults to ``"user"``.

            ``level`` accepts the following types:

            * ``str``: One of the presets listed below, or the ``label`` of a :func:`~.marker`.
              A marker label includes all transforms up to the location of that marker.
            * ``int``: The number of transforms to include from the start of the pipeline. Must
              be non-negative, e.g. ``level=0`` is empty and ``level=2`` contains the first two
              transforms. The device preprocessing counts as a single level, placed after all
              user-applied transforms.
            * ``slice``: A range of transforms, using standard Python slicing with non-negative
              bounds, e.g. ``level=slice(1, 3)`` contains the second and third transforms.
            * ``Sequence``: A ``list``, ``tuple``, or similar containing ``int`` or ``str``
              values, sorted in ascending order with no duplicates. One compile pipeline is
              returned per entry. A ``Sequence`` rather than an arbitrary iterable is required
              so that its length and ordering can be validated up front, without consuming it.
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
      [1] cancel_inverses(),
       ├─▶ checkpoint
      [2] merge_rotations()
    )

    .. details::
        :title: Usage Details

        To get the full compile pipeline used during execution on the device, specify
        ``level="device"``. The device preprocessing is appended after the user-applied
        transforms, and counts as a single level, matching the ``"Device Preprocessing"``
        level reported by :func:`~.analyze`:

        >>> print(get_compile_pipeline(circuit, level="device"))
        CompilePipeline(
          [1] cancel_inverses(),
           ├─▶ checkpoint
          [2] merge_rotations(),
          [3] Device Preprocessing
        )

        We can also retrieve the compile pipeline up to a specific stage indicated by the name
        given to a :func:`~.marker`, which in our example is the ``"checkpoint"`` marker:

        >>> print(get_compile_pipeline(circuit, level="checkpoint"))
        CompilePipeline(
          [1] cancel_inverses()
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
          [1] cancel_inverses()
        )

        A ``slice`` extracts a specific range of transforms from the compile pipeline. For
        example, we can retrieve only the second transform with:

        >>> print(get_compile_pipeline(circuit, level=slice(1, 2)))
        CompilePipeline(
           ├─▶ checkpoint
          [1] merge_rotations()
        )

        Note that the ``"checkpoint"`` marker is still in the pipeline, as it marks a location
        in the pipeline rather than being attached to a particular transform.

        To retrieve several compile pipelines at once, provide a ``Sequence`` of levels. Unlike
        a ``slice``, which returns a single range of transforms, one compile pipeline is
        returned per entry, each containing all transforms up to that level:

        >>> for pipeline in get_compile_pipeline(circuit, level=[1, 2]):
        ...     print(pipeline)
        CompilePipeline(
          [1] cancel_inverses()
        )
        CompilePipeline(
          [1] cancel_inverses(),
           ├─▶ checkpoint
          [2] merge_rotations()
        )

        The ``"all-user"`` and ``"all"`` presets return the compile pipeline at every stage
        of compilation:

        >>> len(get_compile_pipeline(circuit, level="all-user"))
        3
        >>> len(get_compile_pipeline(circuit, level="all"))
        4

    """

    if not isinstance(level, (int, slice, str)):
        raise ValueError(
            f"'level={level}' of type '{type(level)}' is not supported. Please provide an integer, slice or a string as input."
        )

    is_qjit_qnode = False
    if _is_qjit(qnode):
        qnode = qnode.user_function
        if not hasattr(qnode, "compile_pipeline"):
            raise ValueError(
                "Can only retrieve the compilation pipeline if the QJIT'd object is a QNode."
            )
        is_qjit_qnode = True

    @wraps(qnode)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> CompilePipeline:
        full_compile_pipeline = CompilePipeline()
        full_compile_pipeline += qnode.compile_pipeline

        if level == "user":
            return full_compile_pipeline

        # NOTE: Anything past user applied transforms
        # doesn't make sense for a QJIT'd QNode.
        resolved_config = None
        if not is_qjit_qnode:
            # NOTE: Gradient + device transforms are *not* applied to qnodes that contain informative transforms
            resolved_config = construct_execution_config(qnode, resolve=True)(*args, **kwargs)
            if not qnode.compile_pipeline.is_informative:
                outer_pipeline, inner_pipeline = _setup_transform_program(
                    qnode.device, resolved_config
                )
                full_compile_pipeline += outer_pipeline + inner_pipeline

        if is_qjit_qnode and isinstance(level, str) and level in ("gradient", "device"):
            raise NotImplementedError(f"'level={level}' is not supported for QJIT'd QNodes.")

        num_user = len(qnode.compile_pipeline)
        level_slice: slice = _resolve_level(level, full_compile_pipeline, num_user, resolved_config)
        resolved_pipeline = full_compile_pipeline[level_slice]

        return resolved_pipeline

    return wrapper

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

from pennylane.transforms.core import CompilePipeline
from pennylane.workflow import construct_execution_config
from pennylane.workflow._setup_transform_program import _setup_transform_program

if TYPE_CHECKING:
    from collections.abc import Callable

    from pennylane.devices.execution_config import ExecutionConfig
    from pennylane.workflow import QNode

P = ParamSpec("P")


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
    """Retrieve the compile pipeline used during execution of a QNode at a designated level.

    Args:
        qnode (QNode): The QNode to get the compile pipeline for.
        level (str, int, slice): Specifies the stage at which to retrieve the compile pipeline.

            - ``"top"``: An empty pipeline, representing the initial stage before any transformations are applied.
            - ``"user"``: Includes only manually applied user transformations.
            - ``"gradient"``: Includes user transformations and any appended gradient-related passes.
            - ``"device"``: The full pipeline (user + gradient + device) as prepared for device execution.
            - ``str``: The name of the specific :func:`qml.marker` manually inserted into the pipeline.
            - ``int``: The number of transformations to include from the start of the pipeline (e.g. ``level=0`` is empty).
            - ``slice``: A subset of the full pipeline defined by a slice object.

    Returns:
        CompilePipeline: the compile pipeline corresponding to the requested level.

    **Example:**

    Consider this simple circuit,

    .. code-block:: python

        from pennylane.workflow import get_compile_pipeline

        dev = qml.device("default.qubit")

        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        def circuit():
            qml.RX(1, wires=0)
            qml.H(0)
            qml.H(0)
            qml.RX(1, wires=0)
            return qml.expval(qml.Z(0))

    We can retrieve the compile pipeline used during execution with,

    >>> print(get_compile_pipeline(circuit)()) # or level="device"
    CompilePipeline(
      [1] cancel_inverses(),
      [2] merge_rotations(),
      [3] defer_measurements(allow_postselect=True),
      [4] decompose(stopping_condition=..., device_wires=None, target_gates=..., name=default.qubit),
      [5] device_resolve_dynamic_wires(wires=None, allow_resets=False),
      [6] validate_device_wires(None, name=default.qubit),
      [7] validate_measurements(analytic_measurements=..., sample_measurements=..., name=default.qubit),
      [8] _conditional_broadcast_expand(),
      [9] no_sampling(name=backprop + default.qubit)
    )

    or use the ``level`` argument to inspect specific stages of the pipeline.

    >>> print(get_compile_pipeline(circuit, level="user")())
    CompilePipeline(
      [1] cancel_inverses(),
      [2] merge_rotations()
    )

    .. details::
        :title: Usage Details

        Consider the circuit below which is loaded with user applied transforms, a checkpoint marker and uses the parameter-shift gradient method,

        .. code-block:: python

            dev = qml.device("default.qubit")

            @qml.metric_tensor
            @qml.transforms.merge_rotations
            @qml.marker("checkpoint")
            @qml.transforms.cancel_inverses
            @qml.qnode(dev, diff_method="parameter-shift", gradient_kwargs={"shifts": np.pi / 4})
            def circuit(x):
                qml.RX(x, wires=0)
                qml.H(0)
                qml.H(0)
                qml.RX(x, wires=0)
                return qml.expval(qml.Z(0))

        By default, without specifying a ``level`` we will get the full compile pipeline that is used during execution on this device.
        Note that this can also be retrieved by manually specifying ``level="device"``,

        >>> print(get_compile_pipeline(circuit)(3.14)) # or level="device"
        CompilePipeline(
          [1] cancel_inverses(),
           ├─▶ checkpoint
          [2] merge_rotations(),
          [3] _expand_metric_tensor(device_wires=None),
          [4] metric_tensor(device_wires=None),
          [5] _expand_transform_param_shift(shifts=0.7853981633974483),
          [6] defer_measurements(allow_postselect=True),
          [7] decompose(stopping_condition=..., device_wires=None, target_gates=All DefaultQubit Gates, name=default.qubit),
          [8] device_resolve_dynamic_wires(wires=None, allow_resets=False),
          [9] validate_device_wires(None, name=default.qubit),
          [10] validate_measurements(analytic_measurements=..., sample_measurements=..., name=default.qubit),
          [11] _conditional_broadcast_expand()
        )

        As can be seen above, this not only includes the two transforms we manually applied, but also a set of transforms used by the device in order to execute the circuit.
        The ``"user"`` level will retrieve the portion of the compile pipeline that was manually applied by the user to the qnode,

        >>> print(get_compile_pipeline(circuit, level="user")(3.14))
        CompilePipeline(
          [1] cancel_inverses(),
           ├─▶ checkpoint
          [2] merge_rotations(),
          [3] _expand_metric_tensor(device_wires=None),
          [4] metric_tensor(device_wires=None)
        )

        The ``"gradient"`` level builds on top of this to then add any relevant gradient transforms,

        >>> print(get_compile_pipeline(circuit, level="gradient")(3.14))
        CompilePipeline(
          [1] cancel_inverses(),
           ├─▶ checkpoint
          [2] merge_rotations(),
          [3] _expand_metric_tensor(device_wires=None),
          [4] metric_tensor(device_wires=None),
          [5] _expand_transform_param_shift(shifts=0.7853981633974483)
        )

        which in this case is ``_expand_transform_param_shift``, a transform that expands all trainable operations
        to a state where the parameter shift transform can operate on them.

        We can use ``qml.marker`` to further subdivide our compile pipeline into stages,

        >>> print(get_compile_pipeline(circuit, level="checkpoint")(3.14))
        CompilePipeline(
          [1] cancel_inverses()
        )

        If ``"top"`` or ``0`` are specified, an empty compile pipeline will be returned,

        >>> print(get_compile_pipeline(circuit, level=0)(3.14))
        CompilePipeline()
        >>> print(get_compile_pipeline(circuit, level="top")(3.14))
        CompilePipeline()

        Integer levels correspond to the number of transforms to retrieve from the compile pipeline,

        >>> print(get_compile_pipeline(circuit, level=3)(3.14))
        CompilePipeline(
          [1] cancel_inverses(),
           ├─▶ checkpoint
          [2] merge_rotations(),
          [3] _expand_metric_tensor(device_wires=None)
        )

        Slice levels enable you to extract a specific range of transformations in the compile pipeline. For example, we can retrieve the second to fourth transform by using a slice,

        >>> print(get_compile_pipeline(circuit, level=slice(1,4))(3.14))
        CompilePipeline(
           ├─▶ checkpoint
          [1] merge_rotations(),
          [2] _expand_metric_tensor(device_wires=None),
          [3] metric_tensor(device_wires=None)
        )

    """

    if not isinstance(level, (int, slice, str)):
        raise ValueError(
            f"'level={level}' of type '{type(level)}' is not supported. Please provide an integer, slice or a string as input."
        )

    @wraps(qnode)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> CompilePipeline:
        resolved_config = construct_execution_config(qnode, resolve=True)(*args, **kwargs)

        full_compile_pipeline = CompilePipeline()
        full_compile_pipeline += qnode.compile_pipeline
        # NOTE: Gradient + device transforms are *not* applied to qnodes that contain informative transforms
        if not qnode.compile_pipeline.is_informative:
            outer_pipeline, inner_pipeline = _setup_transform_program(qnode.device, resolved_config)
            full_compile_pipeline += outer_pipeline + inner_pipeline

        num_user = len(qnode.compile_pipeline)
        level_slice: slice = _resolve_level(level, full_compile_pipeline, num_user, resolved_config)
        resolved_pipeline = full_compile_pipeline[level_slice]

        return resolved_pipeline

    return wrapper

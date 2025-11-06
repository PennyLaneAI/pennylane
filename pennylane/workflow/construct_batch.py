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
"""Contains a function extracting the tapes at postprocessing at any stage of a transform program."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import pennylane as qml
from pennylane.exceptions import PennyLaneDeprecationWarning

from ._setup_transform_program import _setup_transform_program
from .qnode import _make_execution_config
from .resolution import _resolve_execution_config

if TYPE_CHECKING:
    from pennylane.qnn.torch import TorchLayer
    from pennylane.tape import QuantumScriptBatch
    from pennylane.typing import PostprocessingFn

    from .qnode import QNode


def _get_full_transform_program(qnode: QNode, gradient_fn) -> qml.transforms.core.TransformProgram:
    program = qml.transforms.core.TransformProgram(qnode.transform_program)

    if getattr(gradient_fn, "expand_transform", False):
        program.add_transform(
            qml.transform(gradient_fn.expand_transform),
            **qnode.gradient_kwargs,
        )

    mcm_config = qml.devices.MCMConfig(
        postselect_mode=qnode.execute_kwargs["postselect_mode"],
        mcm_method=qnode.execute_kwargs["mcm_method"],
    )

    config = _make_execution_config(qnode, gradient_fn, mcm_config)
    config = qnode.device.setup_execution_config(config)
    return program + qnode.device.preprocess_transforms(config)


def _validate_level(
    level: Literal["top", "user", "device", "gradient"] | int | slice | None,
) -> None:
    """Check that the level specification is valid.

    Args:
        level: The level specification from user input

    Raises:
        ValueError: If the level is not recognized
    """
    if level is None:
        warnings.warn(
            "Using `level=None` is deprecated and will be removed in a future release. "
            "Please use `level='device'` to include all transforms.",
            PennyLaneDeprecationWarning,
            stacklevel=2,
        )
        return

    if isinstance(level, (int, slice)):
        return

    if isinstance(level, str):
        if level not in ("top", "user", "device", "gradient"):
            raise ValueError(
                f"level {level} not recognized. Acceptable strings are 'device', 'top', 'user', and 'gradient'."
            )
        return

    raise ValueError(
        f"level {level} not recognized. Acceptable types are None, int, str, and slice."
    )


def _get_user_transform_slice(
    level: Literal["top", "user", "device", "gradient"] | int | slice | None,
    num_user_transforms: int,
) -> slice:
    """Interpret the level specification for the initial user transform slice.

    This function handles slicing into the user transforms before any
    gradient or device transforms are applied.

    Args:
        level: The level specification from user input
        num_user_transforms: Number of user transforms

    Returns:
        slice: The slice to apply to the user transform program
    """
    if level == "top":
        return slice(0, 0)

    if level == "user":
        return slice(0, num_user_transforms)

    if level in ("device", "gradient"):
        return slice(0, None)

    if level is None or isinstance(level, int):
        return slice(0, level)

    return level


def _get_inner_transform_slice(
    level: Literal["top", "user", "device", "gradient"] | int | slice | None,
    num_user_transforms: int,
    has_gradient_expand: bool,
) -> slice:
    """Interpret the level specification for the inner transform slice.

    This function handles slicing into the remaining transforms (gradient + device)
    after user transforms have already been applied. The inner program starts
    from index 0, so we need to adjust level specifications accordingly.

    Args:
        level: The level specification from user input
        num_user_transforms: Number of user transforms (already applied)
        has_gradient_expand: Whether gradient expansion transform exists

    Returns:
        slice: The slice to apply to the remaining transform program
    """
    if level == "gradient":
        end_idx = int(has_gradient_expand)
        return slice(0, end_idx)  # Include only gradient expansion if it exists

    if level == "device":
        return slice(0, None)  # Include all remaining transforms

    if isinstance(level, int):
        # Include additional transforms up to the requested level
        # (levels <= num_user_transforms are handled by early exit)
        inner_level = level - num_user_transforms
        return slice(0, inner_level)

    if level is None:
        return slice(0, None)  # Include all remaining transforms

    # Handle slice objects - adjust for the fact that user transforms are already applied
    start = max(0, (level.start or 0) - num_user_transforms)
    stop = None if level.stop is None else max(0, level.stop - num_user_transforms)
    return slice(start, stop, level.step)


def get_transform_program(
    qnode: QNode,
    level: Literal["top", "user", "device", "gradient"] | int | slice | None = "device",
    gradient_fn="unset",
) -> qml.transforms.core.TransformProgram:
    """Extract a transform program at a designated level.

    .. warning::

        Using ``level=None`` is deprecated and will be removed in a future release.
        Please use ``level='device'`` to include all transforms.

    Args:
        qnode (QNode): the qnode to get the transform program for.
        level (None, str, int, slice): An indication of what transforms to use from the full program.

            - ``None`` or ``"device"``: Uses the entire transformation pipeline.
            - ``"top"``: Ignores transformations and returns the original tape as defined.
            - ``"user"``: Includes transformations that are manually applied by the user.
            - ``"gradient"``: Extracts the gradient-level tape.
            - ``int``: Can also accept an integer, corresponding to a number of transforms in the program. ``level=0`` corresponds to the start of the program.
            - ``slice``: Can also accept a ``slice`` object to select an arbitrary subset of the transform program.

        gradient_fn (None, str, TransformDispatcher): The processed gradient fn for the workflow.

    Returns:
        TransformProgram: the transform program corresponding to the requested level.

    .. details::
        :title: Usage Details

        The transforms are organized as:

        .. image:: ../../_static/transforms_order.png
            :align: center
            :width: 800px
            :target: javascript:void(0);

        where ``transform1`` is first applied to the ``QNode`` followed by ``transform2``.  First, user transforms are run on the tapes,
        followed by the gradient expansion, followed by the device expansion. "Final" transforms, like ``param_shift`` and ``metric_tensor``,
        always occur at the end of the program, despite being part of user transforms. Note that when requesting a level by name
        (e.g. "gradient" or "device"), the preceding levels would be applied as well.

        .. code-block:: python

            dev = qml.device('default.qubit')

            @qml.metric_tensor # final transform
            @qml.transforms.merge_rotations # transform 2
            @qml.transforms.cancel_inverses # transform 1
            @qml.qnode(dev, diff_method="parameter-shift", shifts=np.pi / 4)
            def circuit():
                return qml.expval(qml.Z(0))

        By default, we get the full transform program. This can be explicitly specified by ``level="device"``.

        >>> qml.workflow.get_transform_program(circuit)
        TransformProgram(cancel_inverses, merge_rotations, _expand_metric_tensor,
        _expand_transform_param_shift, validate_device_wires, defer_measurements,
        decompose, validate_measurements, validate_observables, metric_tensor)

        The ``"user"`` transforms are the ones manually applied to the qnode, :func:`~.cancel_inverses`,
        :func:`~.merge_rotations` and :func:`~.metric_tensor`.

        >>> qml.workflow.get_transform_program(circuit, level="user")
        TransformProgram(cancel_inverses, merge_rotations, _expand_metric_tensor, metric_tensor)

        The ``_expand_transform_param_shift`` is the ``"gradient"`` transform.
        This expands all trainable operations to a state where the parameter shift transform can operate on them. For example,
        it will decompose any parametrized templates into operators that have generators. Note how ``metric_tensor`` is still
        present at the very end of resulting program.

        >>> qml.workflow.get_transform_program(circuit, level="gradient")
        TransformProgram(cancel_inverses, merge_rotations, _expand_metric_tensor, _expand_transform_param_shift, metric_tensor)

        ``"top"`` and ``0`` both return empty transform programs.

        >>> qml.workflow.get_transform_program(circuit, level="top")
        TransformProgram()
        >>> qml.workflow.get_transform_program(circuit, level=0)
        TransformProgram()

        The ``level`` can also be any integer, corresponding to a number of transforms in the program.

        >>> qml.workflow.get_transform_program(circuit, level=2)
        TransformProgram(cancel_inverses, merge_rotations)

        ``level`` can also accept a ``slice`` object to select out any arbitrary subset of the
        transform program.  This allows you to select different starting transforms or strides.
        For example, you can skip the first transform or reverse the order:

        >>> qml.workflow.get_transform_program(circuit, level=slice(1,3))
        TransformProgram(merge_rotations, _expand_transform_param_shift)
        >>> qml.workflow.get_transform_program(circuit, level=slice(None, None, -1))
        TransformProgram(metric_tensor, validate_observables, validate_measurements,
        decompose, defer_measurements, validate_device_wires, _expand_transform_param_shift,
        _expand_metric_tensor, merge_rotations, cancel_inverses)

        You can get creative and pick a single category of transforms as follows, excluding
        any preceding transforms (and the final transform if it exists):

        >>> user_prog = qml.workflow.get_transform_program(circuit, level="user")
        >>> grad_prog = qml.workflow.get_transform_program(circuit, level="gradient")
        >>> dev_prog = qml.workflow.get_transform_program(circuit, level="device")
        >>> grad_prog[len(user_prog) - 1 : -1]
        TransformProgram(_expand_transform_param_shift)
        >>> dev_prog[len(grad_prog) - 1 : -1]
        TransformProgram(validate_device_wires, mid_circuit_measurements, decompose, validate_measurements, validate_observables)

    """
    _validate_level(level)
    if gradient_fn == "unset":
        config = qml.workflow.construct_execution_config(qnode, resolve=False)()
        # pylint: disable = protected-access
        config = qml.workflow.resolution._resolve_diff_method(
            config,
            qnode.device,
        )
        gradient_fn = config.gradient_method
    has_gradient_expand = bool(getattr(gradient_fn, "expand_transform", False))
    full_transform_program = _get_full_transform_program(qnode, gradient_fn)

    num_user = len(qnode.transform_program)
    if qnode.transform_program.has_final_transform:
        # final transform is placed after device transforms
        num_user -= 1

    readd_final_transform = False

    if level == "device":
        level = None
    elif level == "top":
        level = 0
    elif level == "user":
        readd_final_transform = True
        level = num_user
    elif level == "gradient":
        readd_final_transform = True

        level = num_user + 1 if has_gradient_expand else num_user

    if level is None or isinstance(level, int):
        level = slice(0, level)

    resolved_program = full_transform_program[level]

    if qnode.transform_program.has_final_transform and readd_final_transform:
        resolved_program += qnode.transform_program[-1:]

    return resolved_program


def construct_batch(
    qnode: QNode | TorchLayer,
    level: Literal["top", "user", "device", "gradient"] | int | slice | None = "user",
) -> Callable:
    """Construct the batch of tapes and post processing for a designated stage in the transform program.

    .. warning::

        Using ``level=None`` is deprecated and will be removed in a future release.
        Please use ``level='device'`` to include all transforms.

    Args:
        qnode (QNode): the qnode we want to get the tapes and post-processing for.
        level (None, str, int, slice): An indication of what transforms to apply before drawing.
            Check :func:`~.workflow.get_transform_program` for more information on the allowed values and usage details of
            this argument.

    Returns:
        Callable:  A function with the same call signature as the initial quantum function. This function returns
        a batch (tuple) of tapes and postprocessing function.

    .. seealso:: :func:`pennylane.workflow.get_transform_program` to inspect the contents of the transform program for a specified level.


    .. details::
        :title: Usage Details

        Suppose we have a QNode with several user transforms.

        .. code-block:: python

            from pennylane.workflow import construct_batch

            @qml.transforms.undo_swaps
            @qml.transforms.merge_rotations
            @qml.transforms.cancel_inverses
            @qml.qnode(qml.device('default.qubit'), diff_method="parameter-shift", gradient_kwargs = {"shifts": np.pi/4})
            def circuit(x):
                qml.RandomLayers(qml.numpy.array([[1.0, 2.0]]), wires=(0,1))
                qml.RX(x, wires=0)
                qml.RX(-x, wires=0)
                qml.SWAP((0,1))
                qml.X(0)
                qml.X(0)
                return qml.expval(qml.X(0) + qml.Y(0))

        We can inspect what the device will execute with:

        >>> batch, fn = construct_batch(circuit, level="device")(1.23)
        >>> batch[0].circuit
        [RY(1.0, wires=[1]),
         RX(2.0, wires=[0]),
         expval(X(0) + Y(0))]

        These tapes can be natively executed by the device. However, with non-backprop devices the parameters
        will need to be converted to NumPy with :func:`~.convert_to_numpy_parameters`.

        >>> fn(dev.execute(batch))
        (np.float64(-0.9092974268256817),)

        Or what the parameter shift gradient transform will be applied to:

        >>> batch, fn = construct_batch(circuit, level="gradient")(1.23)
        >>> batch[0].circuit
        [RY(tensor(1., requires_grad=True), wires=[1]),
         RX(tensor(2., requires_grad=True), wires=[0]),
         expval(X(0) + Y(0))]

        We can inspect what was directly captured from the qfunc with ``level=0``.

        >>> batch, fn = construct_batch(circuit, level=0)(1.23)
        >>> batch[0].circuit
        [RandomLayers(tensor([[1., 2.]], requires_grad=True), wires=[0, 1]),
         RX(1.23, wires=[0]),
         RX(-1.23, wires=[0]),
         SWAP(wires=[0, 1]),
         X(0),
         X(0),
         expval(X(0) + Y(0))]

        And iterate though stages in the transform program with different integers.
        If we request ``level=1``, the ``cancel_inverses`` transform has been applied.

        >>> batch, fn = construct_batch(circuit, level=1)(1.23)
        >>> batch[0].circuit
        [RandomLayers(tensor([[1., 2.]], requires_grad=True), wires=[0, 1]),
         RX(1.23, wires=[0]),
         RX(-1.23, wires=[0]),
         SWAP(wires=[0, 1]),
         expval(X(0) + Y(0))]

        We can also slice into a subset of the transform program.  ``slice(1, None)`` would skip the first user
        transform ``cancel_inverses``:

        >>> batch, fn = construct_batch(circuit, level=slice(1,None))(1.23)
        >>> batch[0].circuit
        [RY(tensor(1., requires_grad=True), wires=[1]),
         RX(tensor(2., requires_grad=True), wires=[0]),
         X(0),
         X(0),
         expval(X(0) + Y(0))]

    """
    _validate_level(level)
    is_torch_layer = type(qnode).__name__ == "TorchLayer"

    user_program = qnode.transform_program
    num_user_transforms = len(user_program)

    def batch_constructor(*args, **kwargs) -> tuple[QuantumScriptBatch, PostprocessingFn]:
        """Create a batch of tapes and a post processing function."""
        # Check if shots is being passed as parameter for deprecation warning
        shots = qnode._get_shots(kwargs)  # pylint: disable=protected-access

        if is_torch_layer:
            x = args[0]
            kwargs = {
                **{arg: weight.to(x) for arg, weight in qnode.qnode_weights.items()},
            }

        initial_tape = qml.tape.make_qscript(qnode.func, shots=shots)(*args, **kwargs)
        params = initial_tape.get_parameters(trainable_only=False)
        initial_tape.trainable_params = qml.math.get_trainable_indices(params)

        level_slice_initial = _get_user_transform_slice(
            level, num_user_transforms
        )  # This should be fine, since the case where `has_gradient_expand==True` only increase 1 to the end of level slice
        program = user_program[level_slice_initial]
        user_transformed_tapes, user_post_processing = program((initial_tape,))

        if level_slice_initial.stop is not None and level_slice_initial.stop <= num_user_transforms:
            # If the level slice is fully contained within user transforms, we can return early
            return user_transformed_tapes, user_post_processing
        #### User transforms finished #####
        # The new config process we would like to use.
        mcm_config = qml.devices.MCMConfig(
            postselect_mode=qnode.execute_kwargs["postselect_mode"],
            mcm_method=qnode.execute_kwargs["mcm_method"],
        )
        execution_config = _make_execution_config(qnode, qnode.diff_method, mcm_config)

        ###### Resolution of the execution config ######
        execution_config = _resolve_execution_config(
            execution_config,
            qnode.device,
            tapes=user_transformed_tapes,  # Use the user-transformed tapes
        )

        # Use _setup_transform_program like execute() does
        outer_transform_program, inner_transform_program = _setup_transform_program(
            qnode.device,
            execution_config,
            cache=qnode.execute_kwargs["cache"],
            cachesize=qnode.execute_kwargs["cachesize"],
        )
        full_transform_program = outer_transform_program + inner_transform_program

        has_gradient_expand = bool(
            getattr(execution_config.gradient_method, "expand_transform", False)
        )  # Note that it could exist as None which is still False, but can't use hasattr on it.
        level_slice_inner = _get_inner_transform_slice(
            level,
            num_user_transforms,
            has_gradient_expand,
        )
        resolved_program = full_transform_program[level_slice_inner]

        batch, remaining_post_processing = resolved_program(
            user_transformed_tapes
        )  # Use the user-transformed tapes

        def combined_post_processing(results):
            """Combine the user post-processing with the remaining post-processing."""
            intermediate_results = remaining_post_processing(results)
            return user_post_processing(intermediate_results)

        return batch, combined_post_processing

    return batch_constructor

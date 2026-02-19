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

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from pennylane import math
from pennylane.tape import make_qscript

from ._setup_transform_program import _setup_transform_program
from .qnode import _make_execution_config
from .resolution import _resolve_execution_config

if TYPE_CHECKING:
    from pennylane.qnn.torch import TorchLayer
    from pennylane.tape import QuantumScriptBatch
    from pennylane.typing import PostprocessingFn

    from .qnode import QNode


def _validate_level(
    level: str | int | slice,
) -> None:
    """Check that the level specification is valid.

    Args:
        level: The level specification from user input

    Raises:
        ValueError: If the level is not recognized
    """

    if isinstance(level, (int, slice, str)):
        return

    raise ValueError(f"level {level} not recognized. Acceptable types are int, str, and slice.")


def _find_level(program, level):
    found_level = program.get_marker_level(level)
    if found_level is not None:
        return found_level
    raise ValueError(
        f"Level {level} not found in transform program. "
        "Builtin options are 'top', 'user', 'device', and 'gradient'."
        f" Custom levels are {program.markers}."
    )


def _get_user_transform_slice(
    program,
    level: str | int | slice,
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

    if isinstance(level, str):
        return slice(0, _find_level(program, level))

    if isinstance(level, int):
        return slice(0, level)

    return level


def _get_inner_transform_slice(
    level: Literal["top", "user", "device", "gradient"] | int | slice,
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

    # Handle slice objects - adjust for the fact that user transforms are already applied
    assert isinstance(level, slice)
    start = max(0, (level.start or 0) - num_user_transforms)
    stop = None if level.stop is None else max(0, level.stop - num_user_transforms)
    return slice(start, stop, level.step)


def construct_batch(
    qnode: QNode | TorchLayer,
    level: str | int | slice = "user",
) -> Callable:
    """Construct the batch of tapes and post processing for a designated stage in the transform program.

    Args:
        qnode (QNode): the qnode we want to get the tapes and post-processing for.
        level (str, int, slice): An indication of what transforms to apply before
            drawing. Check :func:`~.workflow.get_transform_program` for more
            information on the allowed values and usage details of this argument.

    Returns:
        Callable:
            A function with the same call signature as the initial quantum function.
            This function returns a batch (tuple) of tapes and postprocessing function.

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
        [RY(1.0, wires=[1]), RX(2.0, wires=[0]), X(0), X(0), expval(X(0) + Y(0))]

    """
    _validate_level(level)
    is_torch_layer = type(qnode).__name__ == "TorchLayer"
    user_program = qnode.compile_pipeline
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

        initial_tape = make_qscript(qnode.func, shots=shots)(*args, **kwargs)
        params = initial_tape.get_parameters(trainable_only=False)
        initial_tape.trainable_params = math.get_trainable_indices(params)

        # This should be fine, since the case where `has_gradient_expand==True`
        # only increase 1 to the end of level slice
        level_slice_initial = _get_user_transform_slice(user_program, level, num_user_transforms)
        program = user_program[level_slice_initial]
        user_transformed_tapes, user_post_processing = program((initial_tape,))

        if level_slice_initial.stop is not None and level_slice_initial.stop <= num_user_transforms:
            # If the level slice is fully contained within user transforms, we can return early
            return user_transformed_tapes, user_post_processing
        #### User transforms finished #####
        # The new config process we would like to use.
        mcm_config = {
            "postselect_mode": qnode.execute_kwargs["postselect_mode"],
            "mcm_method": qnode.execute_kwargs["mcm_method"],
        }
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

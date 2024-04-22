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
"""Contains a function extracting the tapes at postprocessing at any stage of a transform program.

"""
from functools import wraps
import inspect
from typing import Union, Callable, Tuple

import pennylane as qml
from .qnode import QNode, _make_execution_config, _get_device_shots


def null_postprocessing(results):
    """A postprocessing function with null behaviour."""
    return results[0]


def expand_fn_transform(expand_fn: Callable) -> "qml.transforms.core.TransformDispatcher":
    """Construct a transform from a tape-to-tape function.

    Args:
        expand_fn (Callable): a function from a single tape to a single tape

    Returns:

        .TransformDispatcher: Returns a transform dispatcher object that that can transform any
        circuit-like object in PennyLane.

    >>> device = qml.device('default.qubit.legacy', wires=2)
    >>> my_transform = qml.transforms.core.expand_fn_transform(device.expand_fn)
    >>> my_transform
    <transform: expand_fn>
    """

    @wraps(expand_fn)
    def wrapped_expand_fn(tape, *args, **kwargs):
        return (expand_fn(tape, *args, **kwargs),), null_postprocessing

    return qml.transform(wrapped_expand_fn)


def _get_full_transform_program(qnode: QNode) -> "qml.transforms.core.TransformProgram":
    program = qml.transforms.core.TransformProgram(qnode.transform_program)
    if getattr(qnode.gradient_fn, "expand_transform", False):
        program.add_transform(
            qml.transform(qnode.gradient_fn.expand_transform),
            **qnode.gradient_kwargs,
        )
    if isinstance(qnode.device, qml.devices.Device):
        config = _make_execution_config(qnode, qnode.gradient_fn)
        return program + qnode.device.preprocess(config)[0]
    program.add_transform(qml.transform(qnode.device.batch_transform))
    program.add_transform(expand_fn_transform(qnode.device.expand_fn))
    return program


def get_transform_program(qnode: "QNode", level=None) -> "qml.transforms.core.TransformProgram":
    """Extract a transform program at a designated level.

    Args:
        qnode (QNode): the qnode to get the transform program for.
        level (None, str, int, slice): And indication of what transforms to use from the full program.

            * ``None``: use the full transform program
            * ``str``: Acceptable keys are ``"user"``, ``"device"``, ``"top"`` and ``"gradient"``
            * ``int``: How many transforms to include, starting from the front of the program
            * ``slice``: a slice to select out components of the transform program.

    Returns:
        TransformProgram: the transform program corresponding to the requested level.

    .. details::
        :title: Usage Details

        The transforms are organized as:

        .. image:: ../../_static/transforms_order.png
            :align: center
            :width: 800px
            :target: javascript:void(0);

        where ``transform1`` is first applied to the ``QNode`` followed by ``transform2``.  First user transforms are run on the tapes,
        followed by the gradient expansion, followed by the device expansion.  "Final" transforms, like ``param_shift`` and ``metric_tensor``,
        always occur at the end of the program.

        .. code-block:: python

            dev = qml.device('default.qubit')

            @qml.metric_tensor # final transform
            @qml.transforms.merge_rotations # transform 2
            @qml.transforms.cancel_inverses # transform 1
            @qml.qnode(dev, diff_method="parameter-shift", shifts=np.pi / 4)
            def circuit():
                return qml.expval(qml.Z(0))

        By default, we get the full transform program. This can be manually specified by ``level=None``.

        >>> qml.workflow.get_transform_program(circuit)
        TransformProgram(cancel_inverses, merge_rotations, _expand_metric_tensor,
        _expand_transform_param_shift, validate_device_wires, defer_measurements,
        decompose, validate_measurements, validate_observables, metric_tensor)

        The ``"user"`` transforms are the ones manually applied to the qnode, :class:`~.cancel_inverses` and
        :class:`~.merge_rotations`.

        >>> qml.workflow.get_transform_program(circuit, level="user")
        TransformProgram(cancel_inverses, merge_rotations)

        The ``_expand_transform_param_shift`` is the ``"gradient"`` transform.  This expands all trainable
        operations to a state where the parameter shift transform can operate on them. For example, it will decompose
        any parametrized templates into operators that have generators.

        >>> qml.workflow.get_transform_program(circuit, level="gradient")
        TransformProgram(cancel_inverses, merge_rotations, _expand_transform_param_shift)

        ``"device"`` includes all transforms except for a ``"final"`` transform, if it exists.  This usually
        corresponds to the circuits that will be sent to the device to execute.

        >>> qml.workflow.get_transform_program(circuit, level="device")
        TransformProgram(cancel_inverses, merge_rotations, _expand_transform_param_shift,
        validate_device_wires, defer_measurements, decompose, validate_measurements,
        validate_observables)

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

    """
    full_transform_program = _get_full_transform_program(qnode)

    num_user = len(qnode.transform_program)
    if qnode.transform_program.has_final_transform:
        # final transform is placed after device transforms
        num_user -= 1

    if level == "device":
        level = -1 if full_transform_program.has_final_transform else None
    elif level == "top":
        level = 0
    elif level == "user":
        level = num_user
    elif level == "gradient":
        if getattr(qnode.gradient_fn, "expand_transform", False):
            level = slice(0, num_user + 1)
        else:
            level = slice(0, num_user)
    elif isinstance(level, str):
        raise ValueError(
            f"level {level} not recognized. Acceptable strings are 'device', 'top', 'user', and 'gradient'."
        )
    if level is None or isinstance(level, int):
        level = slice(0, level)
    return full_transform_program[level]


def construct_batch(qnode: QNode, level: Union[None, str, int, slice] = "user") -> Callable:
    """Construct the batch of tapes and post processing for a designated stage in the transform program.

    Args:
        qnode (QNode): the qnode we want to get the tapes and post-processing for.
        level (None, str, int, slice): And indication of what transforms to use from the full program.

            * ``None``: use the full transform program
            * ``str``: Acceptable keys are ``"top"``, ``"user"``, ``"device"``, and ``"gradient"``
            * ``int``: How many transforms to include, starting from the front of the program
            * ``slice``: a slice to select out components of the transform program.

    Returns:
        Callable:  a function with the same call signature as the initial quantum function. This function returns
        a batch (tuple) of tapes and postprocessing function.

    .. seealso:: :func:`pennylane.workflow.get_transform_program` to inspect the contents of the transform program for a specified level.


    .. details::
        :title: Usage Details

        Suppose we have a QNode with several user transforms.

        .. code-block:: python

            @qml.transforms.undo_swaps
            @qml.transforms.merge_rotations
            @qml.transforms.cancel_inverses
            @qml.qnode(qml.device('default.qubit'), diff_method="parameter-shift", shifts=np.pi / 4)
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
        [RY(tensor(1., requires_grad=True), wires=[1]),
         RX(tensor(2., requires_grad=True), wires=[0]),
         expval(X(0) + Y(0))]

        These tapes can be natively executed by the device, though with non-backprop devices the parameters
        will need to be converted to numpy with :func:`~.convert_to_numpy_parameters`.

        >>> fn(dev.execute(batch))
        (tensor(-0.90929743, requires_grad=True),)

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

    # pylint: disable=protected-access
    def batch_constructor(*args, **kwargs) -> Tuple[Tuple["qml.tape.QuantumTape", Callable]]:
        """Create a batch of tapes and a post processing function."""
        if "shots" in inspect.signature(qnode.func).parameters:
            shots = _get_device_shots(qnode.device)
        else:
            shots = kwargs.pop("shots", _get_device_shots(qnode.device))

        initial_tape = qml.tape.make_qscript(qnode.func, shots=shots)(*args, **kwargs)

        qnode._update_gradient_fn(tape=initial_tape)
        program = get_transform_program(qnode, level=level)
        return program((initial_tape,))

    return batch_constructor

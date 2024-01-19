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
from typing import Union, Callable, Tuple

import pennylane as qml
from .qnode import QNode, _make_execution_config, _get_device_shots


def _get_full_transform_program(qnode: QNode) -> "qml.transforms.core.TransformProgram":
    program = qml.transforms.core.TransformProgram(qnode.transform_program)
    if getattr(qnode.gradient_fn, "expand_transform", False):
        program.add_transform(
            qml.transform(qnode.gradient_fn.expand_transform),
            **qnode.gradient_kwargs,
        )
    if isinstance(qnode.device, qml.devices.Device):
        config = _make_execution_config(qnode)
        return program + qnode.device.preprocess(config)[0]
    program.add_transform(qml.transform(qnode.device.batch_transform))
    program.add_transform(qml.transforms.core.expand_fn_transform(qnode.device.expand_fn))
    return program


def transform_program(qnode: "QNode", level=None) -> "qml.transforms.core.TransformProgram":
    """Extract a transform program at a designated level.

    Args:
        qnode (QNode): ?
        level (None, str, int, slice): And indication

            * ``None``: use the full transform program
            * ``str``: Acceptable keys are ``"user"``, ``"device"``, and ``"gradient"``
            * ``int``: How many transforms to include, starting from the front of the program
            * ``slice``: a slice to select out components of the transform program.

    Returns:
        TransformProrgram: the transform program corresponding to t


    .. code-block:: python

        dev = qml.device('default.qubit', wires=4)

        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(dev, diff_method="parameter-shift", shifts=np.pi / 4)
        def circuit():
            return qml.expval(qml.PauliZ(0))

    By default, we get the full transform program:

    >>> qml.workflow.transform_program(circuit)
    TransformProgram(cancel_inverses, merge_rotations, _expand_transform_param_shift,
    validate_device_wires, defer_measurements, decompose, validate_measurements,
    validate_observables)

    The `"user"` transforms are the one's manually applied to the qnode, :class:`~.cancel_inverses` and
    :class:`~.merge_rotations`.

    >>> qml.workflow.transform_program(circuit, level="user")
    TransformProgram(cancel_inverses, merge_rotations)

    The ``_expand_transform_param_shift`` is the ``"gradient"`` transform.  This expands all trainable
    operations to a state where the parameter shift transform can operate on them. For example, it will decompose
    any parametrized templates into operators that have generators.

    >>> qml.workflow.transform_program(circuit, level="gradient")
    TransformProgram(cancel_inverses, merge_rotations, _expand_transform_param_shift)

    ``"device"`` includes all transforms in the full transform program.  This corresponds to the circuits that will
    be sent to the device to execute.

    >>> qml.workflow.transform_program(circuit, level="device")
    TransformProgram(cancel_inverses, merge_rotations, _expand_transform_param_shift,
    validate_device_wires, defer_measurements, decompose, validate_measurements,
    validate_observables)

    >>> qml.workflow.transform_program(circuit, level="top")
    TransformProgram()

    >>> qml.workflow.transform_program(circuit, level=2)
    TransformProgram(cancel_inverses, merge_rotations)

    ``level`` can also accept a ``slice`` object to select out any arbitrary subset of the transform program.  This allows you to select
    different starting transforms or strides. Fro example, you can get the transform program with a reversed order.

    >>> qml.workflow.transform_program(circuit, level=slice(1,3))
    TransformProgram(merge_rotations, _expand_transform_param_shift)
    >>> qml.workflow.transform_program(circuit, level=slice(None, 0, -1))
    TransformProgram(validate_observables, validate_measurements, decompose,
    defer_measurements, validate_device_wires, _expand_transform_param_shift, merge_rotations)


    """
    full_transform_program = _get_full_transform_program(qnode)

    if level == "device":
        level = None
    elif level == "top":
        level = 0
    elif level == "user":
        level = len(qnode.transform_program)
    elif level == "gradient":
        if getattr(qnode.gradient_fn, "expand_transform", False):
            level = slice(0, len(qnode.transform_program) + 1)
        else:
            level = slice(0, len(qnode.transform_program))

    if level is None or isinstance(level, int):
        level = slice(0, level)
    return full_transform_program[level]


def construct_batch(qnode: QNode, level: Union[None, str, int, slice] = "user") -> Callable:
    """Construct the batch of tapes and post processing for a designated stage in the transform program.

    Args:
        qnode (QNode): ? not sure how to best define qnode here ?
        level (None, str, int, slice): And indication

            * ``None``: use the full transform program
            * ``str``: Acceptable keys are ``"user"``, ``"device"``, and ``"gradient"``
            * ``int``: How many transforms to include, starting from the front of the program
            * ``slice``: a slice to select out components of the transform program.

    Returns:
        Callable:  a function with the same call signature as the initial quantum function. This function returns
            a batch (tuple) of tapes and postprocessing function.

    Suppose we have a device with several user transforms.

    .. code-block:: python

        x = np.array([0.1, 0.2])

        dev = qml.device('default.qubit', wires=4)

        @qml.transforms.merge_rotations
        @qml.transforms.cancel_inverses
        @qml.qnode(dev, diff_method="parameter-shift", shifts=np.pi / 4)
        def circuit():
            return qml.expval(qml.PauliZ(0))

    We can inspect what the device will execute with:

    >>> batch, fn = construct_batch(circuit, expansion_strategy="device")(1.23)
    >>> batch[0].circuit
    [SWAP(wires=[0, 2]),
    SWAP(wires=[1, 2]),
    RY(tensor(1., requires_grad=True), wires=[0]),
    RX(tensor(2., requires_grad=True), wires=[1]),
    expval(PauliX(wires=[0]) + PauliY(wires=[0]))]

    These tapes can be natively executed by the device, though with non-backprop devices the parameters
    will need to be converted to numpy with :func:`~.convert_to_numpy_parameters`.

    >>> fn(dev.execute(batch))
    (tensor(0.84147098, requires_grad=True),)

    Or what the parameter shift gradient transform will be applied to:

    >>> batch, fn = construct_batch(circuit, xpansion_strategy="gradient")(1.23)
    >>> batch[0].circuit
    [Permute(wires=[0, 1, 2]),
    RY(tensor(1., requires_grad=True), wires=[0]),
    RX(tensor(2., requires_grad=True), wires=[1]),
    expval(PauliX(wires=[0]) + PauliY(wires=[0]))]

    We can inpsect what was directly captured from the qfunc with ``expansion_strategy=0``.

    >>> batch, fn = construct_batch(circuit, expansion_strategy=0)(1.23)
    >>> batch[0].circuit
    [Permute(wires=[0, 1, 2]),
    RandomLayers(tensor([[1., 2.]], requires_grad=True), wires=[0, 1]),
    RX(1.23, wires=[0]),
    RX(-1.23, wires=[0]),
    PauliX(wires=[0]),
    PauliX(wires=[0]),
    expval(PauliX(wires=[0]) + PauliY(wires=[0]))]

    And iterate though stages in the transform program with different integers.
    If we request ``expansion_strategy=1``, the ``cancel_inverses`` transform has been applied.

    >>> batch, fn = construct_batch(circuit, expansion_strategy=1)(1.23)
    >>> batch[0].circuit
    [Permute(wires=[0, 1, 2]),
    RandomLayers(tensor([[1., 2.]], requires_grad=True), wires=[0, 1]),
    RX(1.23, wires=[0]),
    RX(-1.23, wires=[0]),
    expval(PauliX(wires=[0]) + PauliY(wires=[0]))]

    We can also slice into a subset of the transform program.  ``slice(1, None)`` would skip the first user
    transform ``cancel_inverses``:

    >>> batch, fn = construct_batch(circuit, expansion_strategy=slice(1,None))(1.23, include_permute=False)
    >>> batch[0].circuit
    [RY(tensor(1., requires_grad=True), wires=[0]),
    RX(tensor(2., requires_grad=True), wires=[1]),
    PauliX(wires=[0]),
    PauliX(wires=[0]),
    expval(PauliX(wires=[0]) + PauliY(wires=[0]))]

    """
    program = transform_program(qnode, level=level)

    def batch_constructor(*args, **kwargs) -> Tuple[Tuple["qml.tape.QuantumTape", Callable]]:
        """Create a batch of tapes and a post processing function."""
        if qnode.qfunc_uses_shots_arg:
            shots = _get_device_shots(qnode.device)
        else:
            shots = kwargs.pop("shots", _get_device_shots(qnode.device))

        with qml.queuing.AnnotatedQueue() as q:
            qnode.func(*args, **kwargs)

        initial_tape = qml.tape.QuantumScript.from_queue(q, shots)
        return program((initial_tape,))

    return batch_constructor

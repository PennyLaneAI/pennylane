# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Provides transforms for adding simple noise models to quantum circuits.
"""
from collections.abc import Sequence
from typing import Type, Union

from pennylane import BasisState, QubitStateVector, apply, Device
from pennylane.operation import Channel
from pennylane.ops.channel import __qubit_channels__
from pennylane.tape import QuantumTape
from pennylane.transforms.qfunc_transforms import qfunc_transform, single_tape_transform


@single_tape_transform
def add_noise_to_tape(
    tape: QuantumTape,
    noisy_op: Type[Channel],
    noisy_op_args: Union[tuple, float],
    position: str = "all",
) -> QuantumTape:
    r"""Add noisy operations to an input tape.

    The tape will be updated to have noisy gates, specified by the ``noisy_op`` argument, added
    according to the positioning specified in the ``position`` argument.

    Args:
        tape (QuantumTape): the input tape
        noisy_op (Type[Channel]): the noisy operation to be added at positions within the tape
        noisy_op_args (tuple or float): the arguments fed to the noisy operation, or a single float
            specifying the noise strength
        position (str): Specification of where to add noise. Should be one of: ``"all"`` to add
            the noisy operation after all gates; ``"start"`` to add the noisy operation to all wires
            at the start of the circuit; ``"end"`` to add the noisy operation to all wires at the
            end of the circuit.

    Returns:
        QuantumTape: a noisy version of the input tape

    Raises:
        ValueError: if the noisy operation passed in ``noisy_op`` applies to more than one wire
        ValueError: if the requested ``position`` argument is now ``'start'``, ``'end'`` or
            ``'all'``
        ValueError: if the noisy operation passed in ``noisy_op`` is not a noisy channel
        ValueError: if more than one state preparation is present in the tape, or if the preparation
            is not at the start of the tape

    **Example:**

    Consider the following tape:

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    We can add the :class:`~.AmplitudeDamping` channel to the start of the circuit using:

    >>> from pennylane.transforms import add_noise_to_tape
    >>> noisy_tape = add_noise_to_tape(tape, qml.AmplitudeDamping, 0.05, position="end")
    >>> print(noisy_tape.draw())
     0: ──RX(0.9)──╭C──RY(0.5)──AmplitudeDamping(0.05)──╭┤ ⟨Z ⊗ Z⟩
     1: ──RY(0.4)──╰X──RX(0.6)──AmplitudeDamping(0.05)──╰┤ ⟨Z ⊗ Z⟩
    """
    if noisy_op.num_wires != 1:
        raise ValueError(
            "Adding noise to the tape is only supported for single-qubit noisy operations"
        )
    if position not in ("start", "end", "all"):
        raise ValueError("Position must be either 'start', 'end', or 'all' (default)")
    if noisy_op.__name__ not in __qubit_channels__:
        raise ValueError(
            "The noisy_op argument must be a noisy operation such as qml.AmplitudeDamping"
        )

    if not isinstance(noisy_op_args, Sequence):
        noisy_op_args = [noisy_op_args]

    preps = tuple(isinstance(o, (QubitStateVector, BasisState)) for o in tape.operations)
    valid_preps = sum(preps) == 1 and preps[0] is True or sum(preps) == 0
    if not valid_preps:
        raise ValueError("Only a single state preparation at the start of the circuit is supported")

    if sum(preps) == 1:
        apply(tape.operations[0])
        start_pos = 1
    else:
        start_pos = 0

    if position == "start":
        for w in tape.wires:
            noisy_op(*noisy_op_args, wires=w)

    for op in tape.operations[start_pos:]:
        apply(op)
        if position == "all":
            for w in op.wires:
                noisy_op(*noisy_op_args, wires=w)

    if position == "end":
        for w in tape.wires:
            noisy_op(*noisy_op_args, wires=w)

    for m in tape.measurements:
        apply(m)


add_noise = qfunc_transform(add_noise_to_tape)
add_noise.__doc__ = """Add noisy operations to an input quantum function.

    The function will be updated to have noisy gates, specified by the ``noisy_op`` argument, added
    according to the positioning specified in the ``position`` argument.

    Args:
        fn (Callable): the quantum function
        noisy_op (Type[Channel]): the noisy operation to be added at positions within the function
        noisy_op_args (tuple or float): the arguments fed to the noisy operation, or a single float
            specifying the noise strength
        position (str): Specification of where to add noise. Should be one of: ``"all"`` to add
            the noisy operation after all gates; ``"start"`` to add the noisy operation to all wires
            at the start of the circuit; ``"end"`` to add the noisy operation to all wires at the
            end of the circuit.
    
    Returns:
        Callable: a noisy version of the input function

    Raises:
        ValueError: if the noisy operation passed in ``noisy_op`` applies to more than one wire
        ValueError: if the requested ``position`` argument is now ``'start'``, ``'end'`` or
            ``'all'``
        ValueError: if the noisy operation passed in ``noisy_op`` is not a noisy channel
        ValueError: if more than one state preparation is present in the function, or if the
            preparation is not at the start of the function

    **Example:**
    
    The following QNode can be transformed to add noise to the circuit:
    
    .. code-block:: python3
    
        from pennylane.transforms import add_noise
    
        dev = qml.device("default.mixed", wires=2)
        
        @qml.qnode(dev)
        @add_noise(qml.AmplitudeDamping, 0.2, position="end")
        def f(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
    Executions of this circuit will differ from the noise-free value:
    
    >>> f(0.9, 0.4, 0.5, 0.6)
    tensor(0.754847, requires_grad=True)
    >>> print(qml.draw(f)(0.9, 0.4, 0.5, 0.6))
     0: ──RX(0.9)──╭C──RY(0.5)──AmplitudeDamping(0.2)──╭┤ ⟨Z ⊗ Z⟩ 
     1: ──RY(0.4)──╰X──RX(0.6)──AmplitudeDamping(0.2)──╰┤ ⟨Z ⊗ Z⟩ 
"""


def add_noise_to_dev(
    device: Device,
    noisy_op: Type[Channel],
    noisy_op_args: Union[tuple, float],
    position: str = "all",
):
    """Add noisy operations to an input device.

    After applying this transform, circuits executed on the device will have noisy gates added.
    The gates are specified by the ``noisy_op`` argument and positioned according to the
    ``position`` argument. The device is transformed in-place.

    This transform is only compatible with devices that support noisy operations (of type
    :class:`~.Channel`), such as ``default.mixed``.

    .. warning::

        This device transform is a beta feature. Use the :class:`~.beta.QNode` decorator to create
        compatible QNodes and use :func:`~.batch.execute` to execute quantum tapes.

    Args:
        device (Device): the device to be transformed
        noisy_op (Type[Channel]): the noisy operation to be added at positions within the function
        noisy_op_args (tuple or float): the arguments fed to the noisy operation, or a single float
            specifying the noise strength
        position (str): Specification of where to add noise. Should be one of: ``"all"`` to add
            the noisy operation after all gates; ``"start"`` to add the noisy operation to all wires
            at the start of the circuit; ``"end"`` to add the noisy operation to all wires at the
            end of the circuit.

    Raises:
        ValueError: if the noisy operation passed in ``noisy_op`` applies to more than one wire
        ValueError: if the requested ``position`` argument is now ``'start'``, ``'end'`` or
            ``'all'``
        ValueError: if the noisy operation passed in ``noisy_op`` is not a noisy channel
        ValueError: if more than one state preparation is present in the function, or if the
            preparation is not at the start of the function

    **Example:**

    Consider the following QNode:

    .. code-block:: python3

        from pennylane.beta import qnode

        dev = qml.device("default.mixed", wires=4)

        @qnode(dev)
        def f(w, x, y, z):
            qml.RX(w, wires=0)
            qml.RY(x, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=0)
            qml.RX(z, wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    Execution of the circuit on ``dev`` will be noise-free:

    >>> f(0.9, 0.4, 0.5, 0.6)
    tensor(0.86243536, requires_grad=True)

    However, noise can be easily added to the device:

    >>> qml.transforms.add_noise_to_dev(dev, qml.AmplitudeDamping, 0.2)
    >>> f(0.9, 0.4, 0.5, 0.6)
    tensor(0.72945434, requires_grad=True)
    """
    # TODO: Remove warning in docstrings once new QNode replaces the old
    original_expand_fn = device.expand_fn

    def new_expand_fn(circuit, max_expansion=10):
        new_tape = add_noise_to_tape(circuit, noisy_op, noisy_op_args, position)
        return original_expand_fn(new_tape, max_expansion=max_expansion)

    device.expand_fn = new_expand_fn

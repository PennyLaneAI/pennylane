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

from pennylane import BasisState, Device, QubitStateVector, apply
from pennylane.operation import Channel
from pennylane.ops.channel import __qubit_channels__
from pennylane.tape import QuantumTape
from pennylane.transforms.qfunc_transforms import qfunc_transform, single_tape_transform


@qfunc_transform
@single_tape_transform
def add_noise(
    circuit: Union[callable, QuantumTape],
    noisy_op: Type[Channel],
    noisy_op_args: Union[tuple, float],
    position: str = "all",
) -> Union[callable, QuantumTape]:
    """Add noisy operations to an input circuit.

    The circuit will be updated to have noisy gates, specified by the ``noisy_op`` argument, added
    according to the positioning specified in the ``position`` argument.

    Args:
        circuit (callable or QuantumTape): the input circuit
        noisy_op (Type[Channel]): the noisy operation to be added at positions within the circuit
        noisy_op_args (tuple or float): the arguments fed to the noisy operation, or a single float
            specifying the noise strength
        position (str): Specification of where to add noise. Should be one of: ``"all"`` to add
            the noisy operation after all gates; ``"start"`` to add the noisy operation to all wires
            at the start of the circuit; ``"end"`` to add the noisy operation to all wires at the
            end of the circuit.

    Returns:
        callable or QuantumTape: a noisy version of the input circuit

    Raises:
        ValueError: if the noisy operation passed in ``noisy_op`` applies to more than one wire
        ValueError: if the requested ``position`` argument is now ``'start'``, ``'end'`` or
            ``'all'``
        ValueError: if the noisy operation passed in ``noisy_op`` is not a noisy channel
        ValueError: if more than one state preparation is present in the circuit, or if the
            preparation is not at the start of the circuit

    .. UsageDetails::

        **Transforming tapes:**

        Consider the following tape:

        .. code-block:: python3

            with qml.tape.QuantumTape() as tape:
                qml.RX(0.9, wires=0)
                qml.RY(0.4, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(0.5, wires=0)
                qml.RX(0.6, wires=1)
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        We can add the :class:`~.AmplitudeDamping` channel to the end of the circuit using:

        >>> from pennylane.transforms import add_noise
        >>> noisy_tape = add_noise.tape_fn(tape, qml.AmplitudeDamping, 0.05, position="end")
        >>> print(noisy_tape.draw())
         0: ──RX(0.9)──╭C──RY(0.5)──AmplitudeDamping(0.05)──╭┤ ⟨Z ⊗ Z⟩
         1: ──RY(0.4)──╰X──RX(0.6)──AmplitudeDamping(0.05)──╰┤ ⟨Z ⊗ Z⟩

        **Transforming QNodes:**

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
    if noisy_op.num_wires != 1:
        raise ValueError(
            "Adding noise to the circuit is only supported for single-qubit noisy operations"
        )
    if position not in ("start", "end", "all"):
        raise ValueError("Position must be either 'start', 'end', or 'all' (default)")
    if noisy_op.__name__ not in __qubit_channels__:
        raise ValueError(
            "The noisy_op argument must be a noisy operation such as qml.AmplitudeDamping"
        )

    if not isinstance(noisy_op_args, Sequence):
        noisy_op_args = [noisy_op_args]

    preps = tuple(isinstance(o, (QubitStateVector, BasisState)) for o in circuit.operations)
    valid_preps = sum(preps) == 1 and preps[0] is True or sum(preps) == 0
    if not valid_preps:
        raise ValueError("Only a single state preparation at the start of the circuit is supported")

    if sum(preps) == 1:
        apply(circuit.operations[0])
        start_pos = 1
    else:
        start_pos = 0

    if position == "start":
        for w in circuit.wires:
            noisy_op(*noisy_op_args, wires=w)

    for op in circuit.operations[start_pos:]:
        apply(op)
        if position == "all":
            for w in op.wires:
                noisy_op(*noisy_op_args, wires=w)

    if position == "end":
        for w in circuit.wires:
            noisy_op(*noisy_op_args, wires=w)

    for m in circuit.measurements:
        apply(m)


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

        This device transform is a beta feature. Use the :class:`pennylane.beta.QNode` decorator to
        create compatible QNodes and use :func:`~.execute` to execute quantum tapes.

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

        dev = qml.device("default.mixed", wires=2)

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
        new_tape = add_noise.tape_fn(circuit, noisy_op, noisy_op_args, position)
        return original_expand_fn(new_tape, max_expansion=max_expansion)

    device.expand_fn = new_expand_fn

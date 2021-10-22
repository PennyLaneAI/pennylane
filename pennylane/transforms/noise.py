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
from collections import Sequence
from typing import Union, Type

from pennylane import apply
from pennylane.operation import Channel
from pennylane.tape import QuantumTape
from pennylane.transforms.qfunc_transforms import single_tape_transform, qfunc_transform
from pennylane.ops.channel import __qubit_channels__
from pennylane import QubitStateVector, BasisState


@single_tape_transform
def add_noise_to_tape(tape: QuantumTape, noisy_op: Type[Channel], noisy_op_args: Union[tuple, float], position: str = "all") -> QuantumTape:
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
    >>> noisy_tape = add_noise_to_tape(tape, qml.AmplitudeDamping, 0.05, position="start")
    >>> print(noisy_tape.draw())
     0: ──AmplitudeDamping(0.05)──RX(0.9)──╭C──RY(0.5)──╭┤ ⟨Z ⊗ Z⟩
     1: ──AmplitudeDamping(0.05)──RY(0.4)──╰X──RX(0.6)──╰┤ ⟨Z ⊗ Z⟩
    """
    if noisy_op.num_wires != 1:
        raise ValueError("Adding noise to the tape is only supported for single-qubit noisy operations")
    if position not in ("start", "end", "all"):
        raise ValueError("Position must be either 'start', 'end', or 'all' (default)")
    if noisy_op.__name__ not in __qubit_channels__:
        raise ValueError("The noisy_op argument must be a noisy operation such as qml.AmplitudeDamping")

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

    for i, op in enumerate(tape.operations[start_pos:]):
        apply(op)
        if position == "all":
            for w in op.wires:
                noisy_op(*noisy_op_args, wires=w)

    if position == "end":
        for w in tape.wires:
            noisy_op(*noisy_op_args, wires=w)

    for m in tape.measurements:
        apply(m)


add_noise_to_qfunc = qfunc_transform(add_noise_to_tape)
add_noise_to_qfunc.__doc__ = """Add noisy operations to an input quantum function.

    The function will be updated to have noisy gates, specified by the ``noisy_op`` argument, added
    according to the positioning specified in the ``position`` argument.

    Args:
        fn (Callable): the quantum function
        noisy_op (Type[Channel]): the noisy operation to be added at positions within the tape
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

    **Example:**
    
    The following QNode can be transformed to add noise to the circuit:
    
    .. code-block:: python3
    
        from pennylane.transforms import add_noise_to_qfunc
    
        dev = qml.device("default.mixed", wires=2)
        
        @qml.qnode(dev)
        @add_noise_to_qfunc(qml.AmplitudeDamping, 0.2, position="end")
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
"""

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
"""TODO"""
from functools import wraps
from collections.abc import Sequence

from pennylane.operation import Operation


def to_mitiq(fn):
    from pennylane.tape import QuantumTape

    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = list(args)

        circuit_kwarg = kwargs.get("circuit", None)
        qp_kwarg = kwargs.get("qp", None)

        if circuit_kwarg is not None:
            tape_key = "circuit"
        elif qp_kwarg is not None:
            tape_key = "qp"
        else:
            tape_key = None

        tape = circuit_kwarg or qp_kwarg or args[0]
        dev = kwargs.get("executor", None) or args[1]

        tape_no_measurements = _remove_measurements(tape)

        def new_executor(updated_tape):
            with QuantumTape as updated_tape_with_measurements:
                for op in updated_tape.operations:
                    op.queue()

                for meas in tape.measurements:
                    meas.queue()

            return dev.execute(updated_tape_with_measurements)[0]

        if tape_key is not None:
            kwargs[tape_key] = tape_no_measurements
        else:
            args[0] = tape_no_measurements

        if kwargs.get("executor", None) is not None:
            kwargs["executor"] = new_executor
        else:
            args[1] = new_executor()

        return fn(*args, **kwargs)

    return wrapper


def _remove_measurements(tape):
    """Removes the measurements of a given tape

    Args:
        tape (QuantumTape): input quantum tape which may include measurements

    Returns:
        QuantumTape: the input tape with the measurements removed
    """
    from pennylane.tape import QuantumTape

    with QuantumTape() as new_tape:
        for op in tape.operations:
            op.queue()
    return new_tape


def add_noise_to_tape(tape, noisy_op: Operation, noisy_op_args, position: str = "all"):
    """Add noisy operations to an input tape.

    Args:
        tape (QuantumTape): the input tape
        noisy_op (Operation): the noisy operation to be added at positions within the tape
        noisy_op_args (tuple or float): the arguments fed to the noisy operation or a single float
            specifying the operation strength
        position (str): Specification of where to add noise. Should be one of: ``"all"`` to add
            the noisy operation after all gates; ``"start"`` to add the noisy operation to all wires
            at the start of the circuit; ``"end"`` to add the noisy operation to all wires at the
            end of the circuit.

    Returns:
        QuantumTape: a noisy version of the input tape

    **Example:**

    Consider the following tape:

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.2, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.5, wires=0)
            qml.RX(0.6, wires=1)

    We can add the :class:`~.AmplitudeDamping` channel to each gate using:

    >>> noisy_tape = add_noise_to_tape(tape, qml.AmplitudeDamping, 0.2, position="all")
    >>> print(noisy_tape.draw())
     0: ──RX(0.2)──AmplitudeDamping(0.2)──╭C──AmplitudeDamping(0.2)──RZ(0.5)──AmplitudeDamping(0.2)──┤
     1: ──RY(0.4)──AmplitudeDamping(0.2)──╰X──AmplitudeDamping(0.2)──RX(0.6)──AmplitudeDamping(0.2)──┤
    """
    from pennylane.tape import QuantumTape
    from pennylane.ops.channel import __qubit_channels__

    if noisy_op.num_wires > 1:
        raise ValueError("Adding noise to the tape is only supported for single-qubit noisy operations")
    if position not in ("start", "end", "all"):
        raise ValueError("Position must be either 'start', 'end', or 'all' (default)")
    if noisy_op.name not in __qubit_channels__:
        raise ValueError("The noisy_op argument must be a noisy operation such as qml.AmplitudeDamping")

    if not isinstance(noisy_op_args, Sequence):
        noisy_op_args = [noisy_op_args]

    with QuantumTape() as noisy_tape:
        if position == "start":
            for w in tape.wires:
                noisy_op(*noisy_op_args, wires=w)

        for op in tape.operations:
            op.queue()

            if position == "all":
                for w in op.wires:
                    noisy_op(*noisy_op_args, wires=w)

        if position == "end":
            for w in tape.wires:
                noisy_op(*noisy_op_args, wires=w)

    return noisy_tape

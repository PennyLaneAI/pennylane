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


def mitiq_interface(fn):
    """A wrapper for functionality from the `mitiq <https://github.com/unitaryfund/mitiq>`__ library
    to support mitigation in PennyLane.

    This function wraps the
    `execute_with_zne() <https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.zne.zne.execute_with_zne>`__
    and
    `execute_with_pec() <https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.pec.pec.execute_with_pec>`__
    functions from ``mitiq``. As a result, the ``circuit`` argument can be a PennyLane tape which
    includes measurements and the ``executor`` argument can be a PennyLane device.

    Args:
        fn (Callable): an ``execute_with_*`` function from the ``mitiq`` library

    Returns:
        function: the wrapped function that interfaces with PennyLane

    **Example:**

    We consider a noisy device by adding noise to ``default.mixed`` using the
    :func:`~.add_noise_to_device` transform:

    >>> dev = qml.device("default.mixed", wires=2)
    >>> qml.mitigate.add_noise_to_device(dev, qml.AmplitudeDamping, 0.2, position="all")

    Our objective is to mitigate noise from the following circuit:

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.5, wires=0)
            qml.RX(0.6, wires=1)

    This can be achieved using the ``mitiq`` library:

    >>> from mitiq import zne
    >>> from mitiq.zne.scaling import fold_global
    >>> qml.mitigate.mitiq_interface(execute_with_zne)(tape, dev, scale_noise=fold_global)
    0.7196657937904828
    """
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
            with QuantumTape() as updated_tape_with_measurements:
                for op in updated_tape.operations:
                    op.queue()

                for meas in tape.measurements:
                    meas.queue()

            return updated_tape_with_measurements.execute(dev)[0]

        if tape_key is not None:
            kwargs[tape_key] = tape_no_measurements
        else:
            args[0] = tape_no_measurements

        if kwargs.get("executor", None) is not None:
            kwargs["executor"] = new_executor
        else:
            args[1] = new_executor

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

    The tape will be updated to have noisy gates, specified by the ``noisy_op`` argument, added
    according to the positioning specified in the ``position`` argument.

    To add noise to all tapes executed by a device, consider using the :func:`add_noise_to_device`
    transform.

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
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.5, wires=0)
            qml.RX(0.6, wires=1)

    We can add the :class:`~.AmplitudeDamping` channel to each gate using:

    >>> noisy_tape = add_noise_to_tape(tape, qml.AmplitudeDamping, 0.2, position="all")
    >>> print(noisy_tape.draw())
     0: ──RX(0.9)──AmplitudeDamping(0.2)──╭C──AmplitudeDamping(0.2)──RZ(0.5)──AmplitudeDamping(0.2)──┤
     1: ──RY(0.4)──AmplitudeDamping(0.2)──╰X──AmplitudeDamping(0.2)──RX(0.6)──AmplitudeDamping(0.2)──┤
    """
    from pennylane.tape import QuantumTape
    from pennylane.ops.channel import __qubit_channels__

    if noisy_op.num_wires > 1:
        raise ValueError("Adding noise to the tape is only supported for single-qubit noisy operations")
    if position not in ("start", "end", "all"):
        raise ValueError("Position must be either 'start', 'end', or 'all' (default)")
    if noisy_op.__name__ not in __qubit_channels__:
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

        for m in tape.measurements:
            m.queue()

    return noisy_tape


def add_noise_to_device(dev, noisy_op: Operation, noisy_op_args, position: str = "all"):
    """Add noise to a device.

    Each circuit executed by the device will have noisy gates, specified by the ``noisy_op``
    argument, added according to the positioning specified in the ``position`` argument. The device
    is modified in place.

    To add noise to a specific tape, consider using the :func:`add_noise_to_tape` transform.

    Args:
        dev (Device): device to be modified in place to add noise
        noisy_op (Operation): the noisy operation to be added at positions within the tape
        noisy_op_args (tuple or float): the arguments fed to the noisy operation or a single float
            specifying the operation strength
        position (str): Specification of where to add noise. Should be one of: ``"all"`` to add
            the noisy operation after all gates; ``"start"`` to add the noisy operation to all wires
            at the start of the circuit; ``"end"`` to add the noisy operation to all wires at the
            end of the circuit.

    **Example:**

    Consider the following device:

    >>> dev = qml.device("default.mixed", wires=2)

    Also consider the tape:

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.5, wires=0)
            qml.RX(0.6, wires=1)

    We can execute the tape on the device using

    >>> tape.execute(dev)
    [0.62160997]

    Noise can be added to the device using

    >>> qml.mitigate.add_noise_to_device(dev, qml.AmplitudeDamping, s, position="all")

    The resulting execution now gives a different result:

    >>> tape.execute(dev)
    [0.97578304]
    """
    original_execute = dev.execute

    def execute(circuit, **kwargs):
        noisy_circuit = add_noise_to_tape(circuit, noisy_op, noisy_op_args, position)
        return original_execute(noisy_circuit, **kwargs)

    dev.execute = execute

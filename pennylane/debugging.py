# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains functions for debugging quantum programs on simulator devices.
"""
import copy
import pdb
import sys
from contextlib import contextmanager

import pennylane as qml
from pennylane import DeviceError


class _Debugger:
    """A debugging context manager.

    Without an active debugging context, devices will not save their internal state when
    encoutering Snapshot operations. The debugger also serves as storage for the device states.

    Args:
        dev (Device): device to attach the debugger to
    """

    def __init__(self, dev):
        # old device API: check if Snapshot is supported
        if isinstance(dev, qml.devices.LegacyDevice) and "Snapshot" not in dev.operations:
            raise DeviceError("Device does not support snapshots.")

        # new device API: check if it's the simulator device
        if isinstance(dev, qml.devices.Device) and not isinstance(
            dev, (qml.devices.DefaultQubit, qml.devices.DefaultClifford)
        ):
            raise DeviceError("Device does not support snapshots.")

        self.snapshots = {}
        self.active = False
        self.device = dev
        dev._debugger = self

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.active = False
        self.device._debugger = None


def snapshots(qnode):
    r"""Create a function that retrieves snapshot results from a QNode.

    Args:
        qnode (.QNode): the input QNode to be simulated

    Returns:
        A function that has the same argument signature as ``qnode`` and returns a dictionary.
        When called, the function will execute the QNode on the registered device and retrieve
        the saved snapshots obtained via the :class:`~.pennylane.Snapshot` operation. Additionally,
        the snapshot dictionary always contains the execution results of the QNode, so the use of
        the tag "execution_results" should be avoided to prevent conflicting key names.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=None)
        def circuit():
            qml.Snapshot(measurement=qml.expval(qml.Z(0))
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.X(0))

    >>> qml.snapshots(circuit)()
    {0: 1.0,
    'very_important_state': array([0.70710678+0.j, 0.        +0.j, 0.70710678+0.j, 0.        +0.j]),
    2: array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j]),
    'execution_results': 0.0}
    """

    def get_snapshots(*args, **kwargs):
        old_interface = qnode.interface
        if old_interface == "auto":
            qnode.interface = qml.math.get_interface(*args, *list(kwargs.values()))

        with _Debugger(qnode.device) as dbg:
            # pylint: disable=protected-access
            if qnode._original_device:
                qnode._original_device._debugger = qnode.device._debugger
            results = qnode(*args, **kwargs)
            # Reset interface
            if old_interface == "auto":
                qnode.interface = "auto"
        dbg.snapshots["execution_results"] = results
        return dbg.snapshots

    return get_snapshots


class PLDB(pdb.Pdb):
    """Custom debugging class integrated with Pdb.

    This class is responsible for storing and updating a global device to be
    used for executing quantum circuits while in debugging context. The core
    debugger functionality is inherited from the native Python debugger (Pdb).

    This class is not directly user-facing, but is interfaced with the
    ``qml.breakpoint()`` function and ``pldb_device_manager`` context manager.
    The former is responsible for launching the debugger prompt and the latter
    is responsible with extracting and storing the ``qnode.device``.

    The device information is used for validation checks and to execute measurements.
    """

    __active_dev = None

    def __init__(self, *args, **kwargs):
        """Initialize the debugger, and set custom prompt string."""
        super().__init__(*args, **kwargs)
        self.prompt = "[pldb]: "

    @classmethod
    def valid_context(cls):
        """Determine if the debugger is called in a valid context.

        Raises:
            RuntimeError: breakpoint is called outside of a qnode execution
            TypeError: breakpoints not supported on this device
        """

        if not qml.queuing.QueuingManager.recording() or not cls.has_active_dev():
            raise RuntimeError("Can't call breakpoint outside of a qnode execution")

        if cls.get_active_device().name not in ("default.qubit", "lightning.qubit"):
            raise TypeError("Breakpoints not supported on this device")

    @classmethod
    def add_device(cls, dev):
        """Update the global active device.

        Args:
            dev (Union[Device, "qml.devices.Device"]): the active device
        """
        cls.__active_dev = dev

    @classmethod
    def get_active_device(cls):
        """Return the active device.

        Raises:
            RuntimeError: No active device to get

        Returns:
            Union[Device, "qml.devices.Device"]: The active device
        """
        if not cls.has_active_dev():
            raise RuntimeError("No active device to get")

        return cls.__active_dev

    @classmethod
    def has_active_dev(cls):
        """Determine if there is currently an active device.

        Returns:
            bool: True if there is an active device
        """
        return bool(cls.__active_dev)

    @classmethod
    def reset_active_dev(cls):
        """Reset the global active device variable to None."""
        cls.__active_dev = None

    @classmethod
    def _execute(cls, batch_tapes):
        """Execute the batch of tapes on the active device"""
        dev = cls.get_active_device()

        valid_batch = batch_tapes
        if dev.wires:
            valid_batch = qml.devices.preprocess.validate_device_wires(
                batch_tapes, wires=dev.wires
            )[0]

        program, new_config = dev.preprocess()
        new_batch, fn = program(valid_batch)

        # TODO: remove [0] index once compatible with transforms
        return fn(dev.execute(new_batch, new_config))[0]


@contextmanager
def pldb_device_manager(device):
    """Context manager to automatically set and reset active
    device on the PLDB debugger.

    Args:
        device (Union[Device, "qml.devices.Device"]): the active device instance
    """
    try:
        PLDB.add_device(device)
        yield
    finally:
        PLDB.reset_active_dev()


def breakpoint():
    """A function which freezes execution and launches the PennyLane debugger (PLDB).

    This function marks a location in a quantum function (QNode). When it is encountered during
    execution of the quantum circuit, an interactive debugging prompt is launched to step
    throught the circuit execution using Pdb like commands (:code:`list`, :code:`next`,
    :code:`continue`, :code:`quit`).

    **Example**

    Consider the following python script containing the quantum circuit with breakpoints.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.breakpoint()

            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)

            qml.breakpoint()

            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(0))

        circuit(1.23)

    Running the above python script opens up the interactive :code:`[pldb]:` prompt in the terminal.
    The prompt specifies the path to the script along with the next line to be executed after the breakpoint.

    .. code-block:: console

        > /Users/your/path/to/script.py(8)circuit()
        -> qml.RX(x, wires=0)
        [pldb]:

    We can interact with the prompt using the commands: :code:`list` , :code:`next`,
    :code:`continue`, and :code:`quit`. Additionally, we can also access any variables defined in the function.

    .. code-block:: console

        [pldb]: x
        1.23

    The :code:`list` command will print a section of code around the breakpoint, highlighting the next line
    to be executed.

    .. code-block:: console

        [pldb]: list
          3
          4  	@qml.qnode(dev)
          5  	def circuit(x):
          6  	    qml.breakpoint()
          7
          8  ->	    qml.RX(x, wires=0)
          9  	    qml.Hadamard(wires=1)
         10
         11  	    qml.breakpoint()
         12
         13  	    qml.CNOT(wires=[0, 1])
        [pldb]:

    The :code:`next` command will execute the next line of code, and print the new line to be executed.

    .. code-block:: console

        [pldb]: next
        > /Users/your/path/to/script.py(9)circuit()
        -> qml.Hadamard(wires=1)
        [pldb]:

    The :code:`continue` command will resume code execution until another breakpoint is reached. It will
    then print the new line to be executed. Finally, :code:`quit` will resume execution of the file and
    terminate the debugging prompt.

    .. code-block:: console

        [pldb]: continue
        > /Users/your/path/to/script.py(13)circuit()
        -> qml.CNOT(wires=[0, 1])
        [pldb]: quit

    """
    PLDB.valid_context()  # Ensure its being executed in a valid context

    debugger = PLDB(skip=["pennylane.*"])  # skip internals when stepping through trace
    debugger.set_trace(sys._getframe().f_back)  # pylint: disable=protected-access


def state():
    """Compute the quantum state at the current point in the quantum circuit.

    Returns:
        Array(complex): quantum state of the circuit.

    **Example**

    While in a "debugging context", we can query the state as we would at the end of a circuit.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)

            qml.breakpoint()

            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(0))

        circuit(1.23)

    Running the above python script opens up the interactive :code:`[pldb]:` prompt in the terminal.
    We can query the state:

    .. code-block:: console

        [pldb]: longlist
          4  	@qml.qnode(dev)
          5  	def circuit(x):
          6  	    qml.RX(x, wires=0)
          7  	    qml.Hadamard(wires=1)
          8
          9  	    qml.breakpoint()
         10
         11  ->	    qml.CNOT(wires=[0, 1])
         12  	    return qml.expval(qml.Z(0))
        [pldb]: qml.debugging.state()
        array([0.57754604+0.j        , 0.57754604+0.j        ,
        0.        -0.40797128j, 0.        -0.40797128j])

    """
    with qml.queuing.QueuingManager.stop_recording():
        m = qml.state()

    return _measure(m)


def expval(op):
    """Compute the expectation value of an observable at the
    current state of the quantum circuit.

    Args:
        op (Operator): the observable to compute the expectation value for.

    Returns:
        complex: expectation value of the operator

    **Example**

    While in a "debugging context", we can query the expectation value of an observable
    as we would at the end of a circuit.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)

            qml.breakpoint()

            qml.CNOT(wires=[0, 1])
            return qml.state()

        circuit(1.23)

    Running the above python script opens up the interactive :code:`[pldb]:` prompt in the terminal.
    We can query the expectation value:

    .. code-block:: console

        [pldb]: longlist
          4  	@qml.qnode(dev)
          5  	def circuit(x):
          6  	    qml.RX(x, wires=0)
          7  	    qml.Hadamard(wires=1)
          8
          9  	    qml.breakpoint()
         10
         11  ->	    qml.CNOT(wires=[0, 1])
         12  	    return qml.state()
        [pldb]: qml.debugging.expval(qml.Z(0))
        0.33423772712450256
    """

    qml.queuing.QueuingManager.active_context().remove(op)  # ensure we didn't accidentally queue op

    with qml.queuing.QueuingManager.stop_recording():
        m = qml.expval(op)

    return _measure(m)


def probs(wires=None, op=None):
    """Compute the probability distribution for the state at the current
    point in the quantum circuit.

    Args:
        wires (Union[Iterable, int, str, list]): the wires the operation acts on
        op (Union[Observable, MeasurementValue]): observable (with a ``diagonalizing_gates``
            attribute) that rotates the computational basis, or a  ``MeasurementValue``
            corresponding to mid-circuit measurements.

    Returns:
        Array(float): the probability distribution of the bitstrings for the wires.

    **Example**

    While in a "debugging context", we can query the probability distribution
    as we would at the end of a circuit.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)

            qml.breakpoint()

            qml.CNOT(wires=[0, 1])
            return qml.state()

        circuit(1.23)

    Running the above python script opens up the interactive :code:`[pldb]:` prompt in the terminal.
    We can query the probability distribution:

    .. code-block:: console

        [pldb]: longlist
          4  	@qml.qnode(dev)
          5  	def circuit(x):
          6  	    qml.RX(x, wires=0)
          7  	    qml.Hadamard(wires=1)
          8
          9  	    qml.breakpoint()
         10
         11  ->	    qml.CNOT(wires=[0, 1])
         12  	    return qml.state()
        [pldb]: qml.debugging.probs()
        array([0.33355943, 0.33355943, 0.16644057, 0.16644057])
    """
    if op:
        qml.queuing.QueuingManager.active_context().remove(
            op
        )  # ensure we didn't accidentally queue op

    with qml.queuing.QueuingManager.stop_recording():
        m = qml.probs(wires, op)

    return _measure(m)


def _measure(measurement):
    """Perform the measurement.

    Args:
        measurement (MeasurementProcess): the type of measurement to be performed

    Returns:
        tuple(complex): results from the measurement
    """
    active_queue = qml.queuing.QueuingManager.active_context()
    copied_queue = copy.deepcopy(active_queue)

    copied_queue.append(measurement)
    qtape = qml.tape.QuantumScript.from_queue(copied_queue)
    return PLDB._execute((qtape,))  # pylint: disable=protected-access


def tape():
    """Access the tape of the quantum circuit.

    The tape can then be drawn to visualize the gates that have
    been applied from the quantum circuit so far.

    Returns:
        QuantumTape: the quantum tape representing the circuit.

    **Example**

    While in a "debugging context", we can access the :code:`QuantumTape` representing the
    operations we have applied so far:

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])

            qml.breakpoint()

            return qml.expval(qml.Z(0))

        circuit(1.23)

    Running the above python script opens up the interactive :code:`[pldb]:` prompt in the terminal.
    We can access the tape and draw it as follows:

    .. code-block:: console

        [pldb]: t = qml.debugging.tape()
        [pldb]: print(t.draw())
        0: ──RX─╭●─┤
        1: ──H──╰X─┤

    """
    active_queue = qml.queuing.QueuingManager.active_context()
    return qml.tape.QuantumScript.from_queue(active_queue)

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
"""
This module contains functionality for the PennyLane Debugger (PLDB) to support
interactive debugging of quantum circuits.
"""
import copy
import pdb
import sys
from contextlib import contextmanager

from pennylane.devices.preprocess import validate_device_wires
from pennylane.measurements import expval, probs, state
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript


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
        self.prompt = "[pldb] "

    @classmethod
    def valid_context(cls):
        """Determine if the debugger is called in a valid context.

        Raises:
            RuntimeError: breakpoint is called outside of a qnode execution
            TypeError: breakpoints not supported on this device
        """

        if not QueuingManager.recording() or not cls.has_active_dev():
            raise RuntimeError("Can't call breakpoint outside of a qnode execution.")

        if cls.get_active_device().name not in ("default.qubit", "lightning.qubit"):
            raise TypeError("Breakpoints not supported on this device.")

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
            valid_batch = validate_device_wires(batch_tapes, wires=dev.wires)[0]

        program, new_config = dev.preprocess()
        new_batch, fn = program(valid_batch)

        # TODO: remove [0] index once compatible with transforms
        return fn(dev.execute(new_batch, new_config))[0]


@contextmanager
def pldb_device_manager(device):
    """Context manager to automatically set and reset active
    device on the Pennylane Debugger (PLDB).

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

    This function marks a location in a quantum circuit (QNode). When it is encountered during
    execution of the quantum circuit, an interactive debugging prompt is launched to step
    through the circuit execution. Since it is based on the `Python Debugger <https://docs.python.org/3/library/pdb.html>`_ (PDB), commands like
    (:code:`list`, :code:`next`, :code:`continue`, :code:`quit`) can be used to navigate the code.

    .. seealso:: :doc:`/code/qml_debugging`

    **Example**

    Consider the following python script containing the quantum circuit with breakpoints.

    .. code-block:: python3
        :linenos:

        import pennylane as qml

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

    Running the above python script opens up the interactive :code:`[pldb]` prompt in the terminal.
    The prompt specifies the path to the script along with the next line to be executed after the breakpoint.

    .. code-block:: console

        > /Users/your/path/to/script.py(9)circuit()
        -> qml.RX(x, wires=0)
        [pldb]

    We can interact with the prompt using the commands: :code:`list` , :code:`next`,
    :code:`continue`, and :code:`quit`. Additionally, we can also access any variables defined in the function.

    .. code-block:: console

        [pldb] x
        1.23

    The :code:`list` command will print a section of code around the breakpoint, highlighting the next line
    to be executed.

    .. code-block:: console

        [pldb] list
        5     @qml.qnode(dev)
        6     def circuit(x):
        7         qml.breakpoint()
        8
        9  ->     qml.RX(x, wires=0)
        10         qml.Hadamard(wires=1)
        11
        12         qml.breakpoint()
        13
        14         qml.CNOT(wires=[0, 1])
        15         return qml.expval(qml.Z(0))
        [pldb]

    The :code:`next` command will execute the next line of code, and print the new line to be executed.

    .. code-block:: console

        [pldb] next
        > /Users/your/path/to/script.py(10)circuit()
        -> qml.Hadamard(wires=1)
        [pldb]

    The :code:`continue` command will resume code execution until another breakpoint is reached. It will
    then print the new line to be executed. Finally, :code:`quit` will resume execution of the file and
    terminate the debugging prompt.

    .. code-block:: console

        [pldb] continue
        > /Users/your/path/to/script.py(14)circuit()
        -> qml.CNOT(wires=[0, 1])
        [pldb] quit

    """
    PLDB.valid_context()  # Ensure its being executed in a valid context

    debugger = PLDB(skip=["pennylane.*"])  # skip internals when stepping through trace
    debugger.set_trace(sys._getframe().f_back)  # pylint: disable=protected-access


def debug_state():
    """Compute the quantum state at the current point in the quantum circuit.

    Debugging measurements do not alter the state, it remains the same until the
    next operation in the circuit.

    Returns:
        Array(complex): quantum state of the circuit

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

    Running the above python script opens up the interactive :code:`[pldb]` prompt in the terminal.
    We can query the state:

    .. code-block:: console

        [pldb] longlist
          4  	@qml.qnode(dev)
          5  	def circuit(x):
          6  	    qml.RX(x, wires=0)
          7  	    qml.Hadamard(wires=1)
          8
          9  	    qml.breakpoint()
         10
         11  ->	    qml.CNOT(wires=[0, 1])
         12  	    return qml.expval(qml.Z(0))
        [pldb] qml.debug_state()
        array([0.57754604+0.j        , 0.57754604+0.j        ,
        0.        -0.40797128j, 0.        -0.40797128j])

    """
    with QueuingManager.stop_recording():
        m = state()

    return _measure(m)


def debug_expval(op):
    """Compute the expectation value of an observable at the
    current point in the quantum circuit.

    Debugging measurements do not alter the state, it remains the same until the
    next operation in the circuit.

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

    Running the above python script opens up the interactive :code:`[pldb]` prompt in the terminal.
    We can query the expectation value:

    .. code-block:: console

        [pldb] longlist
          4  	@qml.qnode(dev)
          5  	def circuit(x):
          6  	    qml.RX(x, wires=0)
          7  	    qml.Hadamard(wires=1)
          8
          9  	    qml.breakpoint()
         10
         11  ->	    qml.CNOT(wires=[0, 1])
         12  	    return qml.state()
        [pldb] qml.debug_expval(qml.Z(0))
        0.33423772712450256
    """

    QueuingManager.active_context().remove(op)  # ensure we didn't accidentally queue op

    with QueuingManager.stop_recording():
        m = expval(op)

    return _measure(m)


def debug_probs(wires=None, op=None):
    """Compute the probability distribution for the state at the current
    point in the quantum circuit.

    Debugging measurements do not alter the state, it remains the same until the
    next operation in the circuit.

    Args:
        wires (Union[Iterable, int, str, list]): the wires the operation acts on
        op (Union[Operator, MeasurementValue]): an observable (with a ``diagonalizing_gates``
            attribute) that rotates the computational basis, or a  ``MeasurementValue``
            corresponding to mid-circuit measurements.

    Returns:
        Array(float): the probability distribution of the bitstrings for the wires

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

    Running the above python script opens up the interactive :code:`[pldb]` prompt in the terminal.
    We can query the probability distribution:

    .. code-block:: console

        [pldb] longlist
          4  	@qml.qnode(dev)
          5  	def circuit(x):
          6  	    qml.RX(x, wires=0)
          7  	    qml.Hadamard(wires=1)
          8
          9  	    qml.breakpoint()
         10
         11  ->	    qml.CNOT(wires=[0, 1])
         12  	    return qml.state()
        [pldb] qml.debug_probs()
        array([0.33355943, 0.33355943, 0.16644057, 0.16644057])

    """
    if op:
        QueuingManager.active_context().remove(op)  # ensure we didn't accidentally queue op

    with QueuingManager.stop_recording():
        m = probs(wires, op)

    return _measure(m)


def _measure(measurement):
    """Perform the measurement.

    Args:
        measurement (MeasurementProcess): the type of measurement to be performed

    Returns:
        tuple(complex): results from the measurement
    """
    active_queue = QueuingManager.active_context()
    copied_queue = copy.deepcopy(active_queue)

    copied_queue.append(measurement)
    qtape = QuantumScript.from_queue(copied_queue)
    return PLDB._execute((qtape,))  # pylint: disable=protected-access


def debug_tape():
    """Access the tape of the quantum circuit.

    The tape can then be used to access all properties stored in :class:`~pennylane.tape.QuantumTape`.
    This can be used to visualize the gates that have
    been applied from the quantum circuit so far or otherwise process the operations.

    Returns:
        QuantumTape: the quantum tape representing the circuit

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

    Running the above python script opens up the interactive :code:`[pldb]` prompt in the terminal.
    We can access the tape and draw it as follows:

    .. code-block:: console

        [pldb] t = qml.debug_tape()
        [pldb] print(t.draw())
        0: ──RX─╭●─┤
        1: ──H──╰X─┤

    """
    active_queue = QueuingManager.active_context()
    return QuantumScript.from_queue(active_queue)

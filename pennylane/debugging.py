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
This module contains functionality for debugging quantum programs on simulator devices.
"""
import copy
import pdb
import sys
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Callable, Sequence

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.typing import Result, ResultBatch


def _is_snapshot_compatible(dev):
    # The `_debugger` attribute is a good enough proxy for snapshot compatibility
    return hasattr(dev, "_debugger")


class _Debugger:
    """A debugging context manager.

    Without an active debugging context, devices will not save their internal state when
    encoutering Snapshot operations. The debugger also serves as storage for the device states.

    Args:
        dev (Device): device to attach the debugger to
    """

    def __init__(self, dev):
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


@transform
def snapshots(tape_: QuantumTape) -> tuple[Sequence[QuantumTape], Callable[[ResultBatch], Result]]:
    r"""This transform processes the :func:`Snapshot <pennylane.Snapshot>` instances depending on the compatibility of the execution device.
    For supported devices, the snapshots' measurements are computed as the execution progresses.
    Otherwise, the :func:`QuantumTape <pennylane.tape.QuantumTape>` gets split into several, one for each snapshot, with each aggregating
    all the operations up to that specific snapshot.

    Args:
        tape_ (QNode or QuantumTape or Callable): a quantum circuit.

    Returns:
        dictionary (dict) or qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:
        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    If tape splitting is carried out, the transform will be conservative about the wires that it includes in each tape.
    So, if all operations preceding a snapshot in a 3-qubit circuit has been applied to only one wire,
    the tape would only be looking at this wire. This can be overriden by the configuration of the execution device
    and its nature.

    Regardless of the transform's behaviour, the output is a dictionary where each key is either
    the tag supplied to the snapshot or its index in order of appearance, in addition to an
    ``"execution_results"`` entry that returns the final output of the quantum circuit. The post-processing
    function is responsible for aggregating the results into this dictionary.

    When the transform is applied to a QNode, the ``shots`` configuration is inherited from the device.
    Therefore, the snapshot measurements must be supported by the device's nature. An exception of this are
    the :func:`qml.state <pennylane.state>` measurements on finite-shot simulators.

    .. warning::

        For devices that do not support snapshots (e.g QPUs, external plug-in simulators), be mindful of
        additional costs that you might incur due to the 1 separate execution/snapshot behaviour.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=None)
        def circuit():
            qml.Snapshot(measurement=qml.expval(qml.Z(0)))
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.X(0))

    >>> qml.snapshots(circuit)()
    {0: 1.0,
    1: array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j]),
    'execution_results': 0.0}

    Here one can see how a device that does not natively support snapshots executes two different circuits:

    .. code-block:: python3

        @qml.snapshots
        @qml.qnode(qml.device("lightning.qubit", shots=100, wires=2), diff_method="parameter-shift")
        def circuit():
            qml.Hadamard(wires=0),
            qml.Snapshot(qml.counts())
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliZ(0))

        with circuit.device.tracker:
            out = circuit()

    >>> circuit.device.tracker.totals
    {'batches': 1, 'simulations': 3, 'executions': 3}

    >>> out
    {0: {'00': tensor(51, requires_grad=True), '10': tensor(49, requires_grad=True)}, 'execution_results': tensor(0., requires_grad=True)}

    Here you can see the default behaviour of the transform for unsupported devices and you can see how the amount of wires included
    in each resulting tape is minimal:

    .. code-block:: python3

        ops = [
            qml.Snapshot(),
            qml.Hadamard(wires=0),
            qml.Snapshot("very_important_state"),
            qml.CNOT(wires=[0, 1]),
            qml.Snapshot(),
        ]

        measurements = [qml.expval(qml.PauliX(0))]

        tape = qml.tape.QuantumTape(ops, measurements)

        tapes, collect_results_into_dict = qml.snapshots(tape)

    >>> print(tapes)
    [<QuantumTape: wires=[], params=0>, <QuantumTape: wires=[0], params=0>, <QuantumTape: wires=[0, 1], params=0>, <QuantumTape: wires=[0, 1], params=0>]
    """

    qml.devices.preprocess.validate_measurements(tape_)

    new_tapes = []
    accumulated_ops = []
    snapshot_tags = []

    for op in tape_.operations:
        if isinstance(op, qml.Snapshot):
            snapshot_tags.append(op.tag or len(new_tapes))
            meas_op = op.hyperparameters["measurement"]

            new_tapes.append(
                type(tape_)(ops=accumulated_ops, measurements=[meas_op], shots=tape_.shots)
            )
        else:
            accumulated_ops.append(op)

    # Create an additional final tape if a return measurement exists
    if tape_.measurements:
        snapshot_tags.append("execution_results")
        new_tapes.append(type(tape_)(ops=accumulated_ops, measurements=tape_.measurements))

    def postprocessing_fn(results, snapshot_tags):
        return dict(zip(snapshot_tags, results))

    if len(new_tapes) > 1:
        warnings.warn(
            "Snapshots are not supported for the given device. Therefore, a tape will be "
            f"created for each snapshot, resulting in a total of {len(new_tapes)} executions.",
            UserWarning,
        )

    return new_tapes, partial(postprocessing_fn, snapshot_tags=snapshot_tags)


@snapshots.custom_qnode_transform
def snapshots_qnode(self, qnode, targs, tkwargs):
    """A custom QNode wrapper for the snapshot transform :func:`~.snapshots`.
    Depending on whether the QNode's device supports snapshots, an efficient execution
    would be used. Otherwise, the QNode's tape would be split into several around the
    present snapshots and execute each individually.
    """

    def get_snapshots(*args, **kwargs):
        # Need to construct to generate the tape and be able to validate
        qnode.construct(args, kwargs)
        qml.devices.preprocess.validate_measurements(qnode.tape)

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
            if qnode._original_device:
                qnode.device._debugger = None

        dbg.snapshots["execution_results"] = results
        return dbg.snapshots

    if _is_snapshot_compatible(qnode.device):
        return get_snapshots

    return self.default_qnode_transform(qnode, targs, tkwargs)


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
        """Execute tape on the active device"""
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
    """Launch the custom PennyLane debugger."""
    PLDB.valid_context()  # Ensure its being executed in a valid context

    debugger = PLDB(skip=["pennylane.*"])  # skip internals when stepping through trace
    debugger.set_trace(sys._getframe().f_back)  # pylint: disable=protected-access


def state():
    """Compute the state of the quantum circuit.

    Returns:
        Array(complex): quantum state of the circuit.
    """
    with qml.queuing.QueuingManager.stop_recording():
        m = qml.state()

    return _measure(m)


def expval(op):
    """Compute the expectation value of an observable.

    Args:
        op (Operator): the observable to compute the expectation value for

    Returns:
        complex: expectation value of the operator
    """

    qml.queuing.QueuingManager.active_context().remove(op)  # ensure we didn't accidentally queue op

    with qml.queuing.QueuingManager.stop_recording():
        m = qml.expval(op)

    return _measure(m)


def probs(wires=None, op=None):
    """Compute the probability distribution for the state.
    Args:
        wires (Union[Iterable, int, str, list]): the wires the operation acts on
        op (Union[Observable, MeasurementValue]): observable (with a ``diagonalizing_gates``
            attribute) that rotates the computational basis, or a ``MeasurementValue``
            corresponding to mid-circuit measurements.

    Returns:
        Array(float): the probability distribution of the bitstrings for the wires
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
    """Access the quantum tape of the circuit.

    Returns:
        QuantumScript: the quantum tape representing the circuit
    """
    active_queue = qml.queuing.QueuingManager.active_context()
    return qml.tape.QuantumScript.from_queue(active_queue)

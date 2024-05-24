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
import warnings
from functools import partial
from typing import Callable, Sequence, Tuple

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.transforms import transform


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
def snapshots(tape: QuantumTape) -> Tuple[Sequence[QuantumTape], Callable]:
    r"""Transforms a QNode into several tapes by aggregating all operations up to a `qml.Snapshot`
    operation into their own execution tape.

    The output is a dictionary where each key is either the tag supplied to the snapshot or its
    index in order of appearance, in additition to an "execution_results" that returns the final output
    of the quantum circuit.
    Args:
        tape (QNode or QuantumTape or Callable): a quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    .. warning::

        For devices that do not support snapshots (e.g QPUs, external plug-in's simulators), be mindful of
        additional costs that you might incur due to the 1 separate execution/snapshot behaviour.

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
    new_tapes = []
    accumulated_ops = []
    snapshot_tags = []

    for op in tape.operations:
        if isinstance(op, qml.Snapshot):
            snapshot_tags.append(op.tag or len(new_tapes))
            shots = op.hyperparameters["shots"]

            if shots == -1:
                shots = tape.shots

            shots = (
                qml.measurements.Shots(shots)
                if not isinstance(shots, qml.measurements.Shots)
                else shots
            )

            meas_op = op.hyperparameters["measurement"]
            new_tapes.append(type(tape)(ops=accumulated_ops, measurements=[meas_op], shots=shots))
        else:
            accumulated_ops.append(op)

    # Create an additional final tape if a return measurement exists
    if tape.measurements:
        snapshot_tags.append("execution_results")
        new_tapes.append(type(tape)(ops=accumulated_ops, measurements=tape.measurements))

    def postprocessing_fn(results, snapshot_tags):
        return dict(zip(snapshot_tags, results))

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

    if _is_snapshot_compatible(qnode.device):
        return get_snapshots

    return self.default_qnode_transform(qnode, targs, tkwargs)

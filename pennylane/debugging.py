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
        if isinstance(dev, qml.Device) and "Snapshot" not in dev.operations:
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

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
r"""
This module contains functionality for debugging quantum programs on simulator devices.
"""
from pennylane import DeviceError


def snapshots(qnode):
    r"""Create a function that retrieves snapshot results from a QNode.

    Args:
        qnode (.QNode): the input QNode to be simulated

    Returns:
        A function that has the same argument signature as ``qnode``. When called,
        the function will execute the QNode on the registered device and retrieve
        the saved snapshots obtained via the ``qml.Snapshot`` operation.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)
        @qml.qnode(dev)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.probs([0, 1])

    >>> qml.snapshots(circuit)()
    {0: array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]),
    'very_important_state': array([0.70710678+0.j, 0.        +0.j, 0.70710678+0.j, 0.        +0.j]),
    2: array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j])}
    """

    def get_snapshots(*args, **kwargs):
        qnode(*args, **kwargs)
        try:
            return qnode.device.snapshots
        except AttributeError:
            # pylint: disable=raise-missing-from
            raise DeviceError("Device does not support snapshots.")

    return get_snapshots

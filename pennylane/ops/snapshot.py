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
This module contains the Snapshot (pseudo) operation that is common to both
cv and qubit computing paradigms in PennyLane.
"""
from pennylane.operation import AnyWires, Operation


# pylint: disable=unused-argument
class Snapshot(Operation):
    r"""
    The Snapshot operation saves the internal simulator state at specific
    execution steps of a quantum function. As such, it is a pseudo operation
    with no effect on the quantum state.

    **Details:**

    * Number of wires: AllWires
    * Number of parameters: 0

    Args:
        tag (str or None): An optional custom tag for the snapshot, used to index it
                           in the snapshots dictionary.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface=None)
        def circuit():
            qml.Snapshot()
            qml.Hadamard(wires=0)
            qml.Snapshot("very_important_state")
            qml.CNOT(wires=[0, 1])
            qml.Snapshot()
            return qml.expval(qml.PauliX(0))

    >>> qml.snapshots(circuit)()
    {0: array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]),
    'very_important_state': array([0.70710678+0.j, 0.+0.j, 0.70710678+0.j, 0.+0.j]),
    2: array([0.70710678+0.j, 0.+0.j, 0.+0.j, 0.70710678+0.j]),
    'execution_results': array(0.)}

    .. seealso:: :func:`~.snapshots`
    """
    num_wires = AnyWires
    num_params = 0
    grad_method = None

    def __init__(self, tag=None, do_queue=True):
        self.tag = tag
        super().__init__(wires=[], do_queue=do_queue)

    def label(self, decimals=None, base_label=None, cache=None):
        return "|S|"

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        return []

    def _controlled(self, _):
        return Snapshot(tag=self.tag)

    def adjoint(self):
        return Snapshot(tag=self.tag)

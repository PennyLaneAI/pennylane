# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the :class:`OperationRecorder`.
"""
from pennylane.queuing import QueuingManager

from .tape import QuantumTape


class OperationRecorder(QuantumTape):
    """A template and quantum function inspector,
    allowing easy introspection of operators that have been
    applied without requiring a QNode.

    **Example**:

    The OperationRecorder is a context manager. Executing templates
    or quantum functions stores applied operators in the
    recorder, which can then be printed.

    >>> shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=1, n_wires=2)
    >>> weights = np.random.random(shape)
    >>>
    >>> with OperationRecorder() as rec:
    >>>    qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])


    Alternatively, the :attr:`~.OperationRecorder.queue` attribute can be used
    to directly access the applied :class:`~.Operation` and :class:`~.Observable`
    objects.
    """

    def __init__(self):
        super().__init__()
        self.ops = None
        self.obs = None

    def _process_queue(self):
        super()._process_queue()

        for obj, info in self._queue.items():
            QueuingManager.append(obj, **info)

        # remove the operation recorder from the queuing
        # context
        QueuingManager.remove(self)

        new_tape = self.expand(depth=5, stop_at=lambda obj: not isinstance(obj, QuantumTape))
        self.ops = new_tape.operations
        self.obs = new_tape.observables

    def __str__(self):
        output = ""
        output += "Operations\n"
        output += "==========\n"
        for op in self.ops:
            output += repr(op) + "\n"

        output += "\n"
        output += "Observables\n"
        output += "===========\n"
        for op in self.obs:
            output += repr(op) + "\n"

        return output

    @property
    def queue(self):
        return self.ops + self.obs

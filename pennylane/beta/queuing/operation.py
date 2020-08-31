# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
r"""
This module contains the abstract base classes for defining PennyLane
operations and observables.
"""
import pennylane as qml
from pennylane.operation import Tensor

from .queuing import QueuingContext


class BetaTensor(Tensor):
    """Container class representing tensor products of observables.

    To create a tensor, simply initiate it like so:

    >>> T = Tensor(qml.PauliX(0), qml.Hermitian(A, [1, 2]))

    You can also create a tensor from other Tensors:

    >>> T = Tensor(T, qml.PauliZ(4))

    The ``@`` symbol can be used as a tensor product operation:

    >>> T = qml.PauliX(0) @ qml.Hadamard(2)
    """

    # pylint: disable=abstract-method,too-few-public-methods

    def __init__(self, *args):
        super().__init__(*args)
        self.queue()

    def queue(self):
        """Queues the Tensor instance and updates the ownership related info if applicable."""
        QueuingContext.append(self, owns=tuple(self.obs))

        try:
            for o in self.obs:
                QueuingContext.update_info(o, owner=self)
        except NotImplementedError:
            pass


# ========================================================
# Monkeypatching methods
# ========================================================
# The following functions monkeypatch and un-monkeypatch
# the PennyLane operations to work with the new QueuingContext.

ORIGINAL_QUEUE = qml.operation.Operation.queue
ORIGINAL_INV = qml.operation.Operation.inv


def queue(self):
    """Monkeypatched queuing method."""
    QueuingContext.append(self)
    return self


def inv(self):
    """Monkeypatched inverse method.

    This operation acts as a 'radio button', swapping the current
    boolean property of the 'inverse' annotation on the object in the queue.
    """
    current_inv = QueuingContext.get_info(self).get("inverse", False)
    QueuingContext.update_info(self, inverse=not current_inv)
    return self


def expand(self):
    """Monkeypatched expand method for operations.

    Returns:
        .QuantumTape: Returns a quantum tape that contains the
        operations decomposition, or if not implemented, simply
        the operation itself.
    """
    tape = qml.beta.tapes.QuantumTape()

    with tape:
        try:
            self.decomposition(*self.data, wires=self.wires)
        except NotImplementedError:
            self.__class__(*self.data, wires=self.wires)

    if not self.data:
        # original operation has no trainable parameters
        tape.trainable_params = {}

    return tape


def tensor_init(self, *args):
    """Monkeypatched tensor init method.

    The current Tensor class does not perform queueing, so here we modify
    the init to force queueing. Note that we cannot use super() here,
    as it is not supported during monkeypatching.
    """
    self._eigvals_cache = None
    self.obs = []

    for o in args:
        if isinstance(o, qml.operation.Tensor):
            self.obs.extend(o.obs)
        elif isinstance(o, qml.operation.Observable):
            self.obs.append(o)
        else:
            raise ValueError("Can only perform tensor products between observables.")

    self.queue()


def tensor_queue(self):
    """Monkeypatched tensor queuing method."""
    QueuingContext.append(self, owns=tuple(self.obs))
    for o in self.obs:
        QueuingContext.update_info(o, owner=self)
    return self


def monkeypatch_operations():
    """Monkeypatch the operations to work with the new QueuingContext."""
    qml.operation.Operation.queue = queue
    qml.operation.Observable.queue = queue
    qml.operation.Operation.inv = inv
    qml.operation.Operation.expand = expand
    qml.operation.Tensor.__init__ = tensor_init
    qml.operation.Tensor.queue = tensor_queue


def unmonkeypatch_operations():
    """Remove the monkeypatching."""
    qml.operation.Operation.queue = ORIGINAL_QUEUE
    qml.operation.Observable.queue = ORIGINAL_QUEUE
    qml.operation.Operation.inv = ORIGINAL_INV
    qml.operation.Operation.expand = lambda self: None
    qml.operation.Tensor.queue = lambda self: None

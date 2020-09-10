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
The following functions in this module monkeypatch and un-monkeypatch
the PennyLane operations to work with the new AnnotatedQueue.
"""
from unittest import mock

import pennylane as qml

from .queuing import QueuingContext


def operation_queue(self):
    """Monkeypatched :meth:`~.Operation.queue` method, allowing
    operations to queue themselves to the beta :class:`~.QueuingContext`.
    """
    QueuingContext.append(self)
    return self


def operation_inv(self):
    """Monkeypatched :meth:`~.Operation.inv` method.

    Rather than updating some internal attribute, this monkeypatched
    method instead annotates the queue with the current 'inverse'
    status.

    This operation acts as a 'radio button', swapping the current
    boolean property of the 'inverse' annotation on the object in the queue.
    """
    current_inv = QueuingContext.get_info(self).get("inverse", False)
    QueuingContext.update_info(self, inverse=not current_inv)
    return self


def operation_expand(self):
    """Monkeypatched :meth:`~.Operation.expand` method for operations.

    Currently, this monkeypatched expand method simply mirrors the
    existing :meth:`~.Operation.decomposition` method; however with
    two main differences:

    * It returns a tape containing the decomposed operations, rather
      than a list.

    * If a decomposition is not available, it simply returns itself.

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
    """Monkeypatched :meth:`~.Tensor.__init__` method, allowing
    Tensors to queue themselves to the beta :class:`~.QueuingContext`,
    and to annotate the queue specifying previously queued objects they
    'own'.
    """
    self._eigvals_cache = None
    self.obs = []

    for o in args:
        if isinstance(o, qml.operation.Tensor):
            self.obs.extend(o.obs)

        elif isinstance(o, qml.operation.Observable):
            self.obs.append(o)

        QueuingContext.update_info(o, owner=self)

    QueuingContext.append(self, owns=tuple(args))


def tensor_matmul(self, other):
    """Monkeypatched :meth:`~.Tensor.__matmul__` method, to ensure
    that tensors created via left matrix multiplication are correctly
    added to the queue, and their components are annotated.
    """
    if isinstance(other, qml.operation.Tensor):
        self.obs.extend(other.obs)

    elif isinstance(other, qml.operation.Observable):
        self.obs.append(other)

    owning_info = QueuingContext.get_info(self)["owns"] + (other,)

    # update the annotated queue information
    QueuingContext.update_info(self, owns=owning_info)
    QueuingContext.update_info(other, owner=self)

    return self


def tensor_rmatmul(self, other):
    """Monkeypatched :meth:`~.Tensor.__rmatmul__` method, to ensure
    that tensors created via right matrix multiplication are correctly
    added to the queue, and their components are annotated.
    """
    self.obs[:0] = [other]
    QueuingContext.update_info(other, owner=self)
    return self


def mock_operations():
    """Create mock operations, observables and tensors that are monkeypatched
    to work with the new QueuingContext.

    Creating mocked methods, rather than directly monkeypatching/overwriting the methods,
    allows us to later remove/undo the monkeypatching once no longer needed.

    Returns:
        list[MagicMock]: list containing the mocked operations.
    """
    # Monkeypatch the 'expand' method of operations directly.
    # This is required since it does not already exist, and so can't be mocked.
    qml.operation.Operation.expand = operation_expand

    mocks = []

    # create mock operation methods
    mocks += [mock.patch.object(qml.operation.Operation, "queue", operation_queue)]
    mocks += [mock.patch.object(qml.operation.Operation, "inv", operation_inv)]

    # create mock observable methods
    mocks += [mock.patch.object(qml.operation.Observable, "queue", operation_queue)]

    # create mock tensor methods
    mocks += [mock.patch.object(qml.operation.Tensor, "__init__", tensor_init)]
    mocks += [mock.patch.object(qml.operation.Tensor, "__matmul__", tensor_matmul)]
    mocks += [mock.patch.object(qml.operation.Tensor, "__rmatmul__", tensor_rmatmul)]

    # Mock the operations so that they no longer perform validation
    # on argument types and domain. This is required to avoid the operations
    # complaining when unknown types (such as TensorFlow and Torch tensors) are
    # used as arguments.
    mocks += [mock.patch.object(qml.operation.Operator, "do_check_domain", False)]

    return mocks

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

# --------------------
# Beta related imports
# --------------------

from pennylane.operation import Observable, Tensor

class BetaTensor(Tensor):
    """Container class representing tensor products of observables.

    To create a tensor, simply initiate it like so:

    >>> T = Tensor(qml.PauliX(0), qml.Hermitian(A, [1, 2]))

    You can also create a tensor from other Tensors:

    >>> T = Tensor(T, qml.PauliZ(4))

    The ``@`` symbol can be used as a tensor product operation:

    >>> T = qml.PauliX(0) @ qml.Hadamard(2)
    """

    # pylint: disable=abstract-method

    def __init__(self, *args):
        super().__init__(*args)
        self.queue()

    def queue(self):
        qml.QueuingContext.append(self, owns=tuple(self.obs))

        if qml.QueuingContext.active_context().__class__.__name__ == "AnnotatedQueue":
            for o in self.obs:
                qml.QueuingContext.update_info(o, owner=self)

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            self.obs.extend(other.obs)
            return self

        if isinstance(other, Observable):
            self.obs.append(other)
            return self

        raise ValueError("Can only perform tensor products between observables.")

    def __rmatmul__(self, other):
        if isinstance(other, Observable):
            self.obs[:0] = [other]
            return self

        raise ValueError("Can only perform tensor products between observables.")

    __imatmul__ = __matmul__

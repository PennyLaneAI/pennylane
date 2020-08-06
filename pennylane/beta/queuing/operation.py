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
import abc
import itertools
import functools
import numbers
from collections.abc import Sequence
from enum import Enum, IntEnum
from pennylane.wires import Wires

import numpy as np
from numpy.linalg import multi_dot

import pennylane as qml

from pennylane.utils import pauli_eigs
from pennylane.variable import Variable

# --------------------
# Beta related imports
# --------------------
#import pennylane.beta.queuing.ops as ops

from pennylane.operation import Observable, Tensor

class BetaObservable(Observable):

    def __matmul__(self, other):
        if isinstance(other, BetaTensor):
            return other.__rmatmul__(self)

        if isinstance(other, BetaObservable):
            return BetaTensor(self, other)

        raise ValueError("Can only perform tensor products between observables.")

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
    return_type = None
    tensor = True
    par_domain = None

    def __init__(self, *args):  # pylint: disable=super-init-not-called
        super().__init__(*args)
        self.queue()

    def queue(self):
        qml.QueuingContext.append(self, owns=tuple(self.obs))

        for o in self.obs:
            try:
                qml.QueuingContext.update_info(o, owner=self)
            except AttributeError:
                pass

        return self

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

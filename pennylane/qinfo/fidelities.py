# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differentiable quantum fidelity"""

from collections.abc import Iterable
import pennylane as qml

from autograd.numpy.numpy_boxes import ArrayBox


def fidelity(qnode0, qnode1, wires0, wires1):
    """Compute the Fidelity entropy from two :class:`.QNode` returning a :func:`~.state`."""

    if wires0 != wires1:
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    def wrapper(signature0=None, signature1=None):
        if len(wires0) == len(qnode0.device.wires):
            state_qnode0 = qnode0
        else:
            state_qnode0 = qml.qinfo.density_matrix_transform(qnode0, indices=wires0)

        if len(wires1) == len(qnode1.device.wires):
            state_qnode1 = qnode1
        else:
            state_qnode1 = qml.qinfo.density_matrix_transform(qnode1, indices=wires1)

        if signature0 is not None:
            if isinstance(signature0, Iterable) or isinstance(signature0, ArrayBox):
                state_qnode0 = state_qnode0(signature0)
            else:
                state_qnode0 = state_qnode0(*signature0)
        else:
            # No args
            state_qnode0 = state_qnode0()

        if signature1 is not None:
            if isinstance(signature1, Iterable) or isinstance(signature1, ArrayBox):
                state_qnode1 = state_qnode1(signature1)
            else:
                state_qnode1 = state_qnode1(*signature1)
        else:
            state_qnode1 = state_qnode1()

        fidelity = qml.math.to_fidelity(state_qnode0, state_qnode1)
        return fidelity

    return wrapper

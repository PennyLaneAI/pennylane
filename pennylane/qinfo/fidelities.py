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
from autograd.numpy.numpy_boxes import ArrayBox

import pennylane as qml


def fidelity(qnode0, qnode1, wires0, wires1):
    r"""Compute the fidelity for two for two :class:`.QNode` returning a :func:`~.state` (a state can be a state vector
    or a density matrix, depending on the device) acting on quantum systems with the same size. For two pure states, the
    fidelity corresponds to the squared overlap. For a pure state and a mixed state, it corresponds to the squared
    expectation of the mixed state in the pure state. Finally for two mixed states, it is defined by the last formula:

    .. math::
        F( \ket{\psi} , \ket{\phi}) = \left|\bra{\psi}\ket{\phi}\right|^2

        F( \ket{\psi} , \sigma ) = \left|\bra{\psi} \sigma \ket{\psi}\right|^2

        F( \rho , \sigma ) = -\text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2

    .. warning::
        The second state is coerced to the type and dtype of the first state. The fidelity is returned in the type
        of the interface of the first state.

    Args:
        state0 (QNode): A :class:`.QNode` returning a :func:`~.state`.
        state1 (QNode): A :class:`.QNode` returning a :func:`~.state`.
        wires0 (Sequence[int]): the wires of the first subsystem
        wires1 (Sequence[int]): the wires of the second subsystem

    Returns:
        func: A function with the same arguments as the QNode that returns the mutual information from its output state.

    **Example**

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit_rx(x, y):
            qml.RX(x, wires=0)
            qml.RZ(y, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit_ry(y):
            qml.RY(y, wires=0)
            return qml.state()

        >>> qml.qinfo.fidelity(circuit_rx, circuit_ry, wires0=[0], wires1=[0])((0.1, 0.3), (0.2))
        0.9905158135644924

    """

    if len(wires0) != len(wires1):
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    def evaluate_fidelity(signature0=None, signature1=None):
        """Wrapper used for evaluation of the fidelity between two states computed from QNodes. It allows giving
        the args and kwargs to each :class:`.QNode`.

        Args:
            signature0 (tuple): Tuple containing the arguments (*args, **kwargs) of the first :class:`.QNode`.
            signature1 (tuple): Tuple containing the arguments (*args, **kwargs) of the second :class:`.QNode`.

        Returns:
            float: Fidelity between two quantum states
        """
        print(signature0, signature1)
        if len(wires0) == len(qnode0.device.wires):
            state_qnode0 = qnode0
        else:
            state_qnode0 = qml.qinfo.density_matrix_transform(qnode0, indices=wires0)

        if len(wires1) == len(qnode1.device.wires):
            state_qnode1 = qnode1
        else:
            state_qnode1 = qml.qinfo.density_matrix_transform(qnode1, indices=wires1)

        if signature0 is not None:
            if not isinstance(signature0, Iterable) or isinstance(signature0, ArrayBox):
                state_qnode0 = state_qnode0(signature0)
            else:
                state_qnode0 = state_qnode0(*signature0)
        else:
            # No args
            state_qnode0 = state_qnode0()

        if signature1 is not None:
            if not isinstance(signature1, Iterable) or isinstance(signature1, ArrayBox):
                state_qnode1 = state_qnode1(signature1)
            else:
                state_qnode1 = state_qnode1(*signature1)
        else:
            # No args
            state_qnode1 = state_qnode1()

        # From the two generated states, compute the fidelity.
        fid = qml.math.fidelity(state_qnode0, state_qnode1)
        return fid

    return evaluate_fidelity

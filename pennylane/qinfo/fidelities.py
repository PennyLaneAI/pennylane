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

import pennylane as qml


def fidelity(qnode0, qnode1, wires0, wires1):
    r"""Compute the fidelity for two :class:`.QNode` returning a :func:`~.state` (a state can be a state vector
    or a density matrix, depending on the device) acting on quantum systems with the same size. For two pure states, the
    fidelity corresponds to the squared overlap. For a pure state and a mixed state, it corresponds to the squared
    expectation of the mixed state in the pure state. Finally for two mixed states, it is defined by the last formula:

    .. math::
        \vspace \text{ (1)} F( \ket{\psi} , \ket{\phi}) = \left|\bra{\psi}\ket{\phi}\right|^2

        \vspace \text{ (2)} F( \ket{\psi} , \sigma ) = \left|\bra{\psi} \sigma \ket{\psi}\right|^2

        \vspace \text{ (3)} F( \rho , \sigma ) = -\text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2

    .. warning::
        The second state is coerced to the type and dtype of the first state. The fidelity is returned in the type
        of the interface of the first state.

    Args:
        state0 (QNode): A :class:`.QNode` returning a :func:`~.state`.
        state1 (QNode): A :class:`.QNode` returning a :func:`~.state`.
        wires0 (Sequence[int]): the wires of the first subsystem
        wires1 (Sequence[int]): the wires of the second subsystem

    Returns:
        func: A function with the same arguments as the QNodes that returns the fidelities between the output states.

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

    # Get the state vector if all wires are selected
    if len(wires0) == len(qnode0.device.wires):
        state_qnode0 = qnode0
    else:
        state_qnode0 = qml.qinfo.density_matrix_transform(qnode0, indices=wires0)

    # Get the state vector if all wires are selected
    if len(wires1) == len(qnode1.device.wires):
        state_qnode1 = qnode1
    else:
        state_qnode1 = qml.qinfo.density_matrix_transform(qnode1, indices=wires1)

    def evaluate_fidelity(all_args0=None, all_args1=None):
        """Wrapper used for evaluation of the fidelity between two states computed from QNodes. It allows giving
        the args and kwargs to each :class:`.QNode`.

        Args:
            all_args0 (tuple): Tuple containing the arguments (*args, **kwargs) of the first :class:`.QNode`.
            all_args1 (tuple): Tuple containing the arguments (*args, **kwargs) of the second :class:`.QNode`.

        Returns:
            float: Fidelity between two quantum states
        """
        if not isinstance(all_args0, tuple) and all_args0 is not None:
            all_args0 = (all_args0,)

        if not isinstance(all_args1, tuple) and all_args1 is not None:
            all_args1 = (all_args1,)

        # If no all_args is given, evaluate the QNode without args
        if all_args0 is not None:
            state0 = state_qnode0(*all_args0)
        else:
            # No args
            state0 = state_qnode0()

        # If no all_args is given, evaluate the QNode without args
        if all_args1 is not None:
            state1 = state_qnode1(*all_args1)
        else:
            # No args
            state1 = state_qnode1()

        # From the two generated states, compute the fidelity.
        fid = qml.math.fidelity(state0, state1)
        return fid

    return evaluate_fidelity

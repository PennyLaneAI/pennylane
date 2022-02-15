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
A transform to obtain the matrix representation of a quantum circuit.
"""
from functools import wraps
from pennylane.wires import Wires
import pennylane as qml


def get_unitary_matrix(circuit, wire_order=None):
    r"""Construct the matrix representation of a quantum circuit.

    Args:
        circuit (pennylane.QNode, .QuantumTape, or Callable): A quantum node, tape,
            or function that applies quantum operations.
        wire_order (Sequence[Any], optional): Order of the wires in the quantum circuit.
            Defaults to the order in which the wires appear in the quantum function.

    Returns:
         function: Function which accepts the same arguments as the QNode or quantum function.
         When called, this function will return the unitary matrix in the appropriate autodiff framework
         (Autograd, TensorFlow, PyTorch, JAX) given its parameters.

    **Example**

    Consider the following function (the same applies for a QNode or tape):

    .. code-block:: python3

        def circuit(theta):
            qml.RX(theta, wires=1)
            qml.PauliZ(wires=0)


    We can use ``get_unitary_matrix`` to generate a new function
    that returns the unitary matrix corresponding to the function ``circuit``:


    >>> get_matrix = get_unitary_matrix(circuit)
    >>> theta = np.pi / 4
    >>> get_matrix(theta)
    array([[ 0.92387953+0.j,  0.+0.j ,  0.-0.38268343j,  0.+0.j],
       [ 0.+0.j,  -0.92387953+0.j,  0.+0.j,  0. +0.38268343j],
       [ 0. -0.38268343j,  0.+0.j,  0.92387953+0.j,  0.+0.j],
       [ 0.+0.j,  0.+0.38268343j,  0.+0.j,  -0.92387953+0.j]])


    Note that since ``wire_order`` was not specified, the default order ``[1, 0]``
    for ``circuit`` was used, and the unitary matrix corresponds to the operation
    :math:`Z\otimes R_X(\theta)`. To obtain the matrix for :math:`R_X(\theta)\otimes Z`,
    specify ``wire_order=[0, 1]`` in the function call:

    >>> get_matrix = get_unitary_matrix(circuit, wire_order=[0, 1])

    You can also get the unitary matrix for operations on a subspace of a
    larger Hilbert space. For example, with the same function ``circuit`` and
    ``wire_order=["a", 0, "b", 1]`` you obtain the :math:`16\times 16` matrix for
    the operation :math:`I\otimes Z\otimes I\otimes  R_X(\theta)`.

    This unitary matrix can also be used in differentiable calculations.
    For example, consider the following cost function:

    .. code-block:: python

        def circuit(theta):
            qml.RX(theta, wires=1)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])

        def cost(theta):
            matrix = get_unitary_matrix(circuit)(theta)
            return np.real(np.trace(matrix))

    Since this cost function returns a real scalar as a function of ``theta``,
    we can differentiate it:

    >>> theta = np.array(0.3, requires_grad=True)
    >>> cost(theta)
    1.9775421558720845
    >>> qml.grad(cost)(theta)
    -0.14943813247359922
    """

    wires = wire_order

    @wraps(circuit)
    def wrapper(*args, **kwargs):

        if isinstance(circuit, qml.QNode):
            # user passed a QNode, get the tape
            circuit.construct(args, kwargs)
            tape = circuit.qtape

            # if no wire ordering is specified, take wire list from the device
            wire_order = circuit.device.wires if wires is None else Wires(wires)

        elif isinstance(circuit, qml.tape.QuantumTape):
            # user passed a tape
            tape = circuit
            # if no wire ordering is specified, take wire list from tape
            wire_order = tape.wires if wires is None else Wires(wires)

        elif callable(circuit):
            # user passed something that is callable but not a tape or qnode.
            tape = qml.transforms.make_tape(circuit)(*args, **kwargs)
            # raise exception if it is not a quantum function
            if len(tape.operations) == 0:
                raise ValueError("Function contains no quantum operation")

            # if no wire ordering is specified, take wire list from tape
            wire_order = tape.wires if wires is None else Wires(wires)

        else:
            raise ValueError("Input is not a tape, QNode, or quantum function")

        # get interface of parameters to be used to construct the output matrix in same framework
        params = tape.get_parameters(trainable_only=False)
        interface = qml.math._multi_dispatch(params)  # pylint: disable=protected-access

        n_wires = len(wire_order)

        # check that all wire labels in the circuit are contained in wire_order
        if not set(tape.wires).issubset(wire_order):
            raise ValueError("Wires in circuit are inconsistent with those in wire_order")

        # initialize the unitary matrix
        unitary_matrix = qml.math.eye(2**n_wires, like=interface)

        for op in tape.operations:

            U = op.get_matrix(wire_order=wire_order)

            # add to total matrix if there are multiple ops
            unitary_matrix = qml.math.dot(U, unitary_matrix)

        return unitary_matrix

    return wrapper

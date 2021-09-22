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
import numpy as np
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
         When called, this function will return the unitary matrix as a numpy array.

    **Example**

    Consider the following function (the same applies for a QNode or tape):

    .. code-block:: python3

        def circuit(theta):
            qml.RX(theta, wires=1)
            qml.PauliZ(wires=0)


    We can use ``get_unitary_matrix`` to generate a new function
    that returns the unitary matrix corresponding to the function ``circuit``:


    >>> get_matrix = get_unitary_matrix(circuit)
    >>> theta = np.pi/4
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
    """

    wires = wire_order

    @wraps(circuit)
    def wrapper(*args, **kwargs):

        if isinstance(circuit, qml.QNode):
            # user passed a QNode, get the tape
            circuit.construct(args, kwargs)
            tape = circuit.qtape

        elif isinstance(circuit, qml.tape.QuantumTape):
            # user passed a tape
            tape = circuit

        elif callable(circuit):
            # user passed something that is callable but not a tape or qnode.
            tape = qml.transforms.make_tape(circuit)(*args, **kwargs)
            # raise exception if it is not a quantum function
            if len(tape.operations) == 0:
                raise ValueError("Function contains no quantum operation")

        else:
            raise ValueError("Input is not a tape, QNode, or quantum function")

        # if no wire ordering is specified, take wire list from tape
        wire_order = tape.wires if wires is None else Wires(wires)

        n_wires = len(wire_order)

        # check that all wire labels in the circuit are contained in wire_order
        if not set(tape.wires).issubset(wire_order):
            raise ValueError("Wires in circuit are inconsistent with those in wire_order")

        # initialize the unitary matrix
        unitary_matrix = np.eye(2 ** n_wires)

        with qml.tape.Unwrap(tape):
            for op in tape.operations:
                # operator wire position relative to wire ordering
                op_wire_pos = wire_order.indices(op.wires)

                I = np.reshape(np.eye(2 ** n_wires), [2] * n_wires * 2)
                axes = (np.arange(len(op.wires), 2 * len(op.wires)), op_wire_pos)

                # reshape op.matrix
                U_op_reshaped = np.reshape(op.matrix, [2] * len(op.wires) * 2)
                U_tensordot = np.tensordot(U_op_reshaped, I, axes=axes)

                unused_idxs = [idx for idx in range(n_wires) if idx not in op_wire_pos]
                # permute matrix axes to match wire ordering
                perm = op_wire_pos + unused_idxs
                U = np.moveaxis(U_tensordot, wire_order.indices(wire_order), perm)

                U = np.reshape(U, ((2 ** n_wires, 2 ** n_wires)))

                # add to total matrix if there are multiple ops
                unitary_matrix = np.dot(U, unitary_matrix)

        return unitary_matrix

    return wrapper

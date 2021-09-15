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
from pennylane.math.utils import get_interface


def get_unitary_matrix(circuit, wire_order=None):
    """Construct the matrix representation of a quantum circuit.

    Args:
        circuit (.QNode, .QuantumTape, or Callable): A quantum node, tape, or function that applies quantum operations.
        wire_order (Sequence[Any], optional): Order of the wires in the quantum circuit. Will default to ``[0, 1, 2, ...]`` when not specified.

    Returns:
         function: Function which accepts the same arguments as the QNode or quantum function.
         When called, this function will return the unitary matrix.

    **Example**

    Consider the following function:

    .. code-block:: python3

        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.PauliZ(wires=1)

    Choosing a wire order, we can use ``get_unitary_matrix`` to generate a new function
    that returns the unitary matrix corresponding to the function ``circuit``:

    >>> wires = [0, 1]
    >>> get_matrix = get_unitary_matrix(circuit, wires)
    >>> theta = np.pi/4
    >>> get_matrix(theta)
    array([[ 0.92387953+0.j,    0.+0.j,    0.-0.38268343j,    0.+0.j],
    [0.+0.j,    -0.92387953+0.j,    0.+0.j,    0.+0.38268343j],
    [0.-0.38268343j,    0.+0.j,    0.92387953+0.j,    0.+0.j],
    [0.+0.j,    0.+0.38268343j,    0.+0.j,    -0.92387953+0.j]])
    """

    wires = wire_order

    @wraps(circuit)
    def wrapper(*args, **kwargs):

        if isinstance(circuit, qml.QNode):
            # user passed a QNode, get the tape
            # first, recast arguments to numpy if they are tf variables
            recast_args = []
            for arg in args:
                if get_interface(arg) == "tensorflow":
                    arg = arg.numpy()
                recast_args.append(arg)

            circuit.construct(recast_args, kwargs)
            tape = circuit.qtape

            if wires is None:  # if no wire ordering is specified, take wire list from tape
                wire_order = tape.wires
            else:
                wire_order = Wires(wires)

        elif isinstance(circuit, qml.tape.QuantumTape):
            # user passed a tape
            tape = circuit
            if wires is None:
                wire_order = tape.wires
            else:
                wire_order = Wires(wires)

        elif callable(circuit):
            # user passed something that is callable but not a tape or qnode.
            tape = qml.transforms.make_tape(circuit)(*args, **kwargs)
            # raise exception if it is not a quantum function
            if len(tape.operations) == 0:
                raise ValueError("Function contains no quantum operation")
            if wires is None:
                wire_order = tape.wires
            else:
                wire_order = Wires(wires)

        else:
            raise ValueError("Input is not a tape, QNode, or quantum function")

        n_wires = len(wire_order)

        # initialize the unitary matrix
        unitary_matrix = np.eye(2 ** n_wires)

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

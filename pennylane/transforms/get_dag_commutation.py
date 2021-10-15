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
from collections import OrderedDict
from functools import wraps
from pennylane.wires import Wires
import pennylane as qml


def get_dag_commutation(circuit):
    r"""Construct the matrix representation of a quantum circuit.

    Args:
        circuit (pennylane.QNode, .QuantumTape, or Callable): A quantum node, tape,
            or function that applies quantum operations.

    Returns:
         function: Function which accepts the same arguments as the QNode or quantum function.
         When called, this function will return the commutation DAG representation of the circuit.

    **Example

    >>> get_dag = get_dag_commutation(circuit)
    >>> theta = np.pi/4
    >>> get_dag(theta)

    """

    # pylint: disable=protected-access

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
        wires = tape.wires

        consecutive_wires = Wires(range(len(wires)))
        wires_map = OrderedDict(zip(wires, consecutive_wires))

        for obs in tape.observables:
            obs._wires = Wires([wires_map[wire] for wire in obs.wires.tolist()])

        with qml.tape.Unwrap(tape):
            # Initialize DAG
            dag = qml.commutation_dag.CommutationDAG(consecutive_wires, tape.observables)

            for operation in tape.operations:
                operation._wires = Wires([wires_map[wire] for wire in operation.wires.tolist()])
                dag.add_node(operation)
            dag._add_successors()
        return dag

    return wrapper

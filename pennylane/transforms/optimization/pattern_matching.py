# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transform finding all maximal matches of a pattern in a quantum circuit."""

import pennylane as qml
from pennylane import apply
from pennylane.transforms import qfunc_transform, get_dag_commutation, make_tape


@qfunc_transform
def pattern_matching(tape, pattern_tapes):
    r"""Quantum function transform to optimize a circuit given a list of patterns.

    Args:
        qfunc (function): A quantum function.
        pattern_tapes(list(.QuantumTape)): List of quantum tapes or quantum functions without measurement.

    Returns:
        function: the transformed quantum function

    **Example**

    Consider the following quantum function.

    .. code-block:: python

        def qfunc(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.PauliZ(0))

    The circuit before optimization:

    """
    for pattern in pattern_tapes:
        # Check the validity of the pattern
        if not isinstance(pattern, qml.tape.QuantumTape):
            raise qml.QuantumFunctionError(
                f"The pattern {pattern}, does not appear "
                "to be a valid quantum tape"
            )

        # Check that it does not contain a measurement.
        if pattern.measurements:
            raise qml.QuantumFunctionError(
                f"The pattern {pattern}, contains measurements. "
            )

        # Verify that the pattern is implementing the identity

        # Verify that the pattern has less qubits and less gates

        # Construct Dag representation of the circuit and the pattern.
        circuit_dag = qml.transforms.get_dag_commutation(tape)()
        pattern_dag = qml.transforms.get_dag_commutation(pattern)()

        print(circuit_dag.get_nodes())
        print(pattern_dag.get_nodes())

        # Initial match
        # Different qubit configurations
        # Forward Match
        # Backward match
        # Compatible and optimized maximal matches
        # Create optimized tape

    # Construct optimized circuit
    #for op in tape.operations:
    #    apply(op)

    # Queue the measurements normally
    #for m in tape.measurements:
    #    apply(m)



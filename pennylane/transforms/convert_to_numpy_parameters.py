# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This file contains preprocessings steps that may be called internally
during execution.
"""

import pennylane as qml
from pennylane import math
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn


def _convert_op_to_numpy_data(op: qml.operation.Operator) -> qml.operation.Operator:
    if math.get_interface(*op.data) == "numpy":
        return op
    # Use operator method to change parameters when it become available
    return qml.ops.functions.bind_new_parameters(op, math.unwrap(op.data))


def _convert_measurement_to_numpy_data(
    m: qml.measurements.MeasurementProcess,
) -> qml.measurements.MeasurementProcess:
    if m.obs is None:
        if m.eigvals() is None or math.get_interface(m.eigvals()) == "numpy":
            return m
        return type(m)(wires=m.wires, eigvals=math.unwrap(m.eigvals()))

    if math.get_interface(*m.obs.data) == "numpy":
        return m
    new_obs = qml.ops.functions.bind_new_parameters(m.obs, math.unwrap(m.obs.data))
    return type(m)(obs=new_obs)


@transform
def convert_to_numpy_parameters(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Transforms a circuit to one with purely numpy parameters.

    Args:
        tape (QuantumScript): a circuit with parameters of any interface

    Returns:
        tuple[List[QuantumScript], function]: The transformed circuits along with a dummy post-processing function.

    **Examples:**

    >>> ops = [qml.S(0), qml.RX(torch.tensor(0.1234), 0)]
    >>> measurements = [qml.state(), qml.expval(qml.Hermitian(torch.eye(2), 0))]
    >>> circuit = qml.tape.QuantumScript(ops, measurements)
    >>> [new_circuit], _ = convert_to_numpy_parameters(circuit)
    >>> new_circuit.circuit
    [S(0),
    RX(0.1234000027179718, wires=[0]),
    state(wires=[]),
    expval(Hermitian(array([[1., 0.],
            [0., 1.]], dtype=float32), wires=[0]))]

    If the component's data does not need to be transformed, it is left uncopied.

    >>> circuit[0] is new_circuit[0]
    True
    >>> circuit[1] is new_circuit[1]
    False
    >>> circuit[2] is new_circuit[2]
    True
    >>> circuit[3] is new_circuit[3]
    False

    """
    new_ops = (_convert_op_to_numpy_data(op) for op in tape.operations)
    new_measurements = (_convert_measurement_to_numpy_data(m) for m in tape.measurements)
    new_circuit = tape.__class__(
        new_ops, new_measurements, shots=tape.shots, trainable_params=tape.trainable_params
    )

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_circuit], null_postprocessing

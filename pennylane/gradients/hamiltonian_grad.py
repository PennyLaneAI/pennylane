# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains a gradient recipe for the coefficients of Hamiltonians."""
# pylint: disable=protected-access,unnecessary-lambda
import pennylane as qml


def hamiltonian_grad(tape, idx):
    """Computes the tapes necessary to get the gradient of a tape with respect to
    a Hamiltonian observable's coefficients.

    Args:
        tape (qml.tape.QuantumTape): tape with a single Hamiltonian expectation as measurement
        idx (int): index of parameter that we differentiate with respect to
    """

    op, m_pos, p_idx = tape.get_operation(idx)

    # get position in queue
    queue_position = m_pos - len(tape.operations)
    new_measurements = list(tape.measurements)

    new_parameters = [0 * d for d in op.data]
    new_parameters[p_idx] = qml.math.ones_like(op.data[p_idx])
    new_obs = qml.ops.functions.bind_new_parameters(op, new_parameters)
    new_obs = qml.simplify(new_obs)

    new_measurements[queue_position] = qml.expval(new_obs)

    new_tape = qml.tape.QuantumScript(tape.operations, new_measurements, shots=tape.shots)

    if len(tape.measurements) > 1:

        def processing_fn(results):
            res = results[0][queue_position]
            zeros = qml.math.zeros_like(res)

            final = [res if i == queue_position else zeros for i, _ in enumerate(tape.measurements)]

            return qml.math.expand_dims(qml.math.stack(final), 0)

        return [new_tape], processing_fn

    return [new_tape], lambda x: x

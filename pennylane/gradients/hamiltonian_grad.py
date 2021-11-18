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


def hamiltonian_grad(tape, idx, params=None):
    """Computes the tapes necessary to get the gradient of a tape with respect to
    a Hamiltonian observable's coefficients.

    Args:
        tape (qml.tape.QuantumTape): tape with a single Hamiltonian expectation as measurement
        idx (int): index of parameter that we differentiate with respect to
        params (array): explicit parameters to set
    """
    op, p_idx = tape.get_operation(idx)
    new_tape = tape.copy(copy_operations=True)

    if params is not None:
        # TODO: remove the params argument when the old QNode is removed
        new_tape.set_parameters(params=params)

    # get position in queue

    queue_position = tape.observables.index(op)
    new_tape._measurements[queue_position] = qml.expval(op.ops[p_idx])

    new_tape._par_info = {}
    new_tape._update()

    if len(tape.measurements) > 1:

        def processing_fn(results):
            res = results[0][queue_position]
            zeros = qml.math.zeros_like(res)

            final = []
            for i, _ in enumerate(tape.measurements):
                final.append(res if i == queue_position else zeros)

            return qml.math.expand_dims(qml.math.stack(final), 0)

        return [new_tape], processing_fn

    return [new_tape], lambda x: x

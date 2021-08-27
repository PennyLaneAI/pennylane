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

    t_idx = list(tape.trainable_params)[idx]
    op = tape._par_info[t_idx]["op"]
    p_idx = tape._par_info[t_idx]["p_idx"]

    new_tape = tape.copy(copy_operations=True, tape_cls=qml.tape.QuantumTape)
    if params is not None:
        new_tape.set_parameters(params=params)

    new_tape._measurements = [qml.expval(op.ops[p_idx])]

    new_tape._par_info = {}
    new_tape._update()

    return [new_tape], lambda x: qml.math.squeeze(x)

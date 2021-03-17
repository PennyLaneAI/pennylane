# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Contains the hamiltonian expectation value tape transform
"""
import itertools
import numpy as np
import pennylane as qml


def hamiltonian_expval(tape):
    r"""
    Returns a list of tapes, and a classical processing function, for computing the expectation
    value of a Hamiltonian.

    **Example**

    Given a tape of the form,

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)

            H = qml.PauliZ(0) @ qml.PauliZ(1) + 2 * qml.PauliX(1) + qml.PauliZ(2)
            qml.expval(H)

    We can use the ``expand_hamiltonian_expval`` transform to generate new tapes and a classical
    post-processing function for computing the expectation value of the Hamiltonian.

    >>> tapes, fn = qml.tape.transforms.expand_hamiltonian_expval(tape)
    >>> print(tapes)

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(tapes)
    >>> print(res)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    """

    hamiltonian = tape.measurements[0].obs
    hamiltonian.simplify()
    combined_obs = []

    c = hamiltonian.coeffs

    for o in hamiltonian.ops:
        combined_obs.append(o)

    new_obs, coeffs = qml.grouping.group_observables(combined_obs, c, grouping_type="commuting")
    tapes = []

    for obs in new_obs:
        new_tape = qml.tape.QuantumTape()
        with new_tape:
            for op in tape.operations:
                op.queue()
            for m in obs:
                qml.expval(m)

        tapes.append(new_tape)

    def processing_fn(results):

        new_coeffs = list(itertools.chain.from_iterable(coeffs))
        new_results = list(itertools.chain.from_iterable(results))

        return np.dot(new_coeffs, new_results)

    return tapes, processing_fn

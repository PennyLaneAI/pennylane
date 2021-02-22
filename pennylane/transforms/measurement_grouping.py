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
Contains the measurement grouping transform
"""
import pennylane as qml


def measurement_grouping(tape, obs_list, coeffs_list):
    """Returns a list of measurement optimized tapes, and a classical processing function, for
    evaluating the expectation value of a provided Hamiltonian.

    Args:
        tape (.QuantumTape): input tape
        obs_list (Sequence[.Observable]): The list of observables to measure
            the expectation values of after executing the tape.
        coeffs_list (Sequence[float]): Coefficients of the Hamiltonian expression.
            Must be of the same length as ``obs_list``.

    Returns:
        tuple[list[.QuantumTape], func]: Returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape results to compute the Hamiltonian expectation value.

    **Example**
    """
    obs_groupings, coeffs_groupings = qml.grouping.group_observables(obs_list, coeffs_list)
    tapes = []

    for obs in obs_groupings:

        with tape.__class__() as new_tape:
            for op in tape.operations:
                op.queue()

            for o in obs:
                qml.expval(o)

        new_tape = new_tape.expand(stop_at=lambda obj: True)
        tapes.append(new_tape)

    def processing_fn(res):
        return qml.math.sum([qml.math.dot(c, r) for c, r in zip(coeffs_groupings, res)])

    return tapes, processing_fn

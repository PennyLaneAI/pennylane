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

    Given the following quantum tape,

    >>> with qml.tape.QuantumTape() as tape:
    ...     qml.RX(0.1, wires=0)
    ...     qml.RX(0.2, wires=1)
    ...     qml.CNOT(wires=[0, 1])
    ...     qml.CNOT(wires=[1, 2])

    and list of observables with coefficients,

    >>> obs = [qml.PauliZ(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliX(2)]
    >>> coeffs = [2.0, -0.54, 0.1]

    We can generate generate measurement optimized tapes corresponding
    to a qubit-wise commuting grouping of the provided observables:

    >>> tapes, fn = qml.transforms.measurement_grouping(tape, obs, coeffs)
    >>> print(tapes)
    [<QuantumTape: wires=[0, 1, 2], params=2>,
     <QuantumTape: wires=[0, 1, 2], params=2>]
    >>> print(fn)
    <function measurement_grouping.<locals>.processing_fn at 0x7f1af81287a0>

    The output are the optimized tapes, and a processing function to apply to the
    results of the evaluated tapes to construct the expectation value of the
    Hamiltonian.

    Note that only two tapes have been returned, rather than three (one for each
    observable); this is because ``qml.PauliZ(0)`` and ``qml.PauliX(2)`` are
    qubit-wise commuting, and can be extracted from a single tape evaluation.

    We can now evaluate these tapes, and apply the processing function:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = fn(dev.batch_execute(tapes))
    >>> print(res)
    2.0007186031172046
    """
    obs_groupings, coeffs_groupings = qml.pauli.grouping.group_observables(obs_list, coeffs_list)
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
        dot_products = [
            qml.math.dot(qml.math.convert_like(c, r), r) for c, r in zip(coeffs_groupings, res)
        ]
        return qml.math.sum(qml.math.stack(dot_products))

    return tapes, processing_fn

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
Contains the hamiltonian expand tape transform
"""
# pylint: disable=protected-access
import pennylane as qml


def hamiltonian_expand(tape, group=True):
    r"""
    Splits a tape measuring a Hamiltonian expectation into mutliple tapes of Pauli expectations,
    and provides a function to recombine the results.

    Args:
        tape (.QuantumTape): the tape used when calculating the expectation value
            of the Hamiltonian
        group (bool): whether to compute groups of non-commuting Pauli observables, leading to fewer tapes

    Returns:
        tuple[list[.QuantumTape], function]: Returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions to compute the expectation value.

    **Example**

    Given a Hamiltonian,

    .. code-block:: python3

        H = qml.PauliY(2) @ qml.PauliZ(1) + 0.5 * qml.PauliZ(2) + qml.PauliZ(1)

    and a tape of the form,

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)

            qml.expval(H)

    We can use the ``hamiltonian_expand`` transform to generate new tapes and a classical
    post-processing function for computing the expectation value of the Hamiltonian.

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape)

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    -0.5

    .. Warning::

         Note that defining Hamiltonians inside of QNodes using arithmetic can lead to errors.
         See :class:`~pennylane.Hamiltonian` for more information.

    The ``group`` keyword argument toggles between the creation of one tape per Pauli observable, or
    one tape per group of non-commuting Pauli observables computed by the :func:`.measurement_grouping`
    transform:

    .. code-block:: python3

        H = qml.Hamiltonian([1., 2., 3.], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        # split H into observable groups [qml.PauliZ(0)] and [qml.PauliX(1), qml.PauliX(0)]
        tapes, fn = qml.transforms.hamiltonian_expand(tape)
        print(len(tapes)) # 2

        # split H into observables [qml.PauliZ(0)], [qml.PauliX(1)] and [qml.PauliX(0)]
        tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
        print(len(tapes)) # 3
    """

    hamiltonian = tape.measurements[0].obs

    if (
        not isinstance(hamiltonian, qml.Hamiltonian)
        or len(tape.measurements) > 1
        or tape.measurements[0].return_type != qml.measure.Expectation
    ):
        raise ValueError(
            "Passed tape must end in `qml.expval(H)`, where H is of type `qml.Hamiltonian`"
        )

    if group:
        hamiltonian.simplify()
        return qml.transforms.measurement_grouping(tape, hamiltonian.ops, hamiltonian.coeffs)

    # create tapes that measure the Pauli-words in the Hamiltonian
    tapes = []
    for ob in hamiltonian.ops:
        # we need to create a new tape here, because
        # updating metadata of a copied tape is error-prone
        # when the observables were changed
        with tape.__class__() as new_tape:
            for op in tape.operations:
                qml.apply(op)
            qml.expval(ob)
        tapes.append(new_tape)

    # create processing function that performs linear recombination
    def processing_fn(res):
        dot_products = [
            qml.math.dot(qml.math.squeeze(res[i]), hamiltonian.coeffs[i]) for i in range(len(res))
        ]
        return qml.math.sum(qml.math.stack(dot_products), axis=0)

    return tapes, processing_fn

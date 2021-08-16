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
        group (bool): Whether to compute disjoint groups of commuting Pauli observables, leading to fewer tapes.
            If grouping information can be found in the Hamiltonian, it will be used even if group=False.

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

    Fewer tapes can be constructed by grouping commuting observables. This can be achieved
    by the ``group`` keyword argument:

    .. code-block:: python3

        H = qml.Hamiltonian([1., 2., 3.], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

    With grouping, the Hamiltonian gets split into two groups of observables (here ``[qml.PauliZ(0)]`` and
    ``[qml.PauliX(1), qml.PauliX(0)]``):

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape)
    >>> len(tapes)
    2

    Without grouping it gets split into three groups (``[qml.PauliZ(0)]``, ``[qml.PauliX(1)]`` and ``[qml.PauliX(0)]``):

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
    >>> len(tapes)
    3

    Alternatively, if the Hamiltonian has already computed groups, they are used even if ``group=False``:

    .. code-block:: python3

        obs = [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)]
        coeffs = [1., 2., 3.]
        H = qml.Hamiltonian(coeffs, obs, grouping_type='qwc')

        # the initialisation already computes grouping information and stores it in the Hamiltonian
        assert H.grouping_indices is not None

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

    Grouping information has been used to reduce the number of tapes from 3 to 2:

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
    >>> len(tapes)
    2
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

    if group or hamiltonian.grouping_indices is not None:

        if hamiltonian.grouping_indices is None:
            hamiltonian.compute_grouping()

        # use groups of observables if available or explicitly requested
        coeffs = [
            qml.math.squeeze(qml.math.take(hamiltonian.coeffs, indices, axis=0))
            for indices in hamiltonian.grouping_indices
        ]
        obs_groupings = [
            [hamiltonian.ops[i] for i in indices] for indices in hamiltonian.grouping_indices
        ]

        tapes = []
        for obs in obs_groupings:

            with tape.__class__() as new_tape:
                for op in tape.operations:
                    op.queue()

                for o in obs:
                    qml.expval(o)

            new_tape = new_tape.expand(stop_at=lambda obj: True)
            tapes.append(new_tape)
    else:
        coeffs = hamiltonian.coeffs

        tapes = []
        for o in hamiltonian.ops:
            with tape.__class__() as new_tape:
                for op in tape.operations:
                    op.queue()
                qml.expval(o)

            tapes.append(new_tape)

    def processing_fn(res):
        # note: res could have an extra dimension here if a shots_distribution
        # is used for evaluation
        dot_products = [qml.math.dot(qml.math.squeeze(r), c) for c, r in zip(coeffs, res)]
        return qml.math.sum(qml.math.stack(dot_products), axis=0)

    return tapes, processing_fn

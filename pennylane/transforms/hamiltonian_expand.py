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
from collections import defaultdict

# pylint: disable=protected-access
from typing import List

import pennylane as qml
from pennylane.measurements import Expectation, MeasurementProcess
from pennylane.ops import SProd, Sum
from pennylane.tape import QuantumScript


def hamiltonian_expand(tape: QuantumScript, group=True):
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
        or tape.measurements[0].return_type != qml.measurements.Expectation
    ):
        raise ValueError(
            "Passed tape must end in `qml.expval(H)`, where H is of type `qml.Hamiltonian`"
        )

    # note: for backward passes of some frameworks
    # it is crucial to use the hamiltonian.data attribute,
    # and not hamiltonian.coeffs when recombining the results

    if group or hamiltonian.grouping_indices is not None:

        if hamiltonian.grouping_indices is None:
            # explicitly selected grouping, but indices not yet computed
            hamiltonian.compute_grouping()

        coeff_groupings = [
            qml.math.stack([hamiltonian.data[i] for i in indices])
            for indices in hamiltonian.grouping_indices
        ]
        obs_groupings = [
            [hamiltonian.ops[i] for i in indices] for indices in hamiltonian.grouping_indices
        ]

        # make one tape per grouping, measuring the
        # observables in that grouping
        tapes = []
        for obs in obs_groupings:
            new_tape = tape.__class__(tape._ops, (qml.expval(o) for o in obs), tape._prep)

            new_tape = new_tape.expand(stop_at=lambda obj: True)
            tapes.append(new_tape)

        def processing_fn(res_groupings):
            if qml.active_return():
                dot_products = [
                    qml.math.dot(
                        qml.math.reshape(
                            qml.math.convert_like(r_group, c_group), qml.math.shape(c_group)
                        ),
                        c_group,
                    )
                    for c_group, r_group in zip(coeff_groupings, res_groupings)
                ]
            else:
                dot_products = [
                    qml.math.dot(r_group, c_group)
                    for c_group, r_group in zip(coeff_groupings, res_groupings)
                ]
            return qml.math.sum(qml.math.stack(dot_products), axis=0)

        return tapes, processing_fn

    coeffs = hamiltonian.data

    # make one tape per observable
    tapes = []
    for o in hamiltonian.ops:
        # pylint: disable=protected-access
        new_tape = tape.__class__(tape._ops, [qml.expval(o)], tape._prep)
        tapes.append(new_tape)

    # pylint: disable=function-redefined
    def processing_fn(res):
        dot_products = [qml.math.dot(qml.math.squeeze(r), c) for c, r in zip(coeffs, res)]
        return qml.math.sum(qml.math.stack(dot_products), axis=0)

    return tapes, processing_fn


# pylint: disable=too-many-branches
def sum_expand(tape: QuantumScript, group=True):
    """Splits a tape measuring a Sum expectation into mutliple tapes of summand expectations,
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

    Given a Sum operator,

    .. code-block:: python3

        S = qml.op_sum(qml.prod(qml.PauliY(2), qml.PauliZ(1)), qml.s_prod(0.5, qml.PauliZ(2)), qml.PauliZ(1))

    and a tape of the form,

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)

            qml.expval(qml.PauliZ(0))
            qml.expval(S)
            qml.expval(qml.PauliX(1))
            qml.expval(qml.PauliZ(2))

    We can use the ``sum_expand`` transform to generate new tapes and a classical
    post-processing function to speed-up the computation of the expectation value of the `Sum`.

    >>> tapes, fn = qml.transforms.sum_expand(tape, group=False)
    >>> for tape in tapes:
    ...     print(tape.measurements)
    [expval(PauliY(wires=[2]) @ PauliZ(wires=[1]))]
    [expval(0.5*(PauliZ(wires=[2])))]
    [expval(PauliZ(wires=[1]))]
    [expval(PauliZ(wires=[0])), expval(PauliX(wires=[1])), expval(PauliZ(wires=[2]))]

    Four tapes are generated: the first three contain the summands of the `Sum` operator,
    and the last tape contains the remaining observables.

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

        >>> fn(res)
    [0.0, -0.5, 0.0, -0.9999999999999996]

    Fewer tapes can be constructed by grouping commuting observables. This can be achieved
    by the ``group`` keyword argument:

    .. code-block:: python3

        S = qml.op_sum(qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1)), qml.s_prod(3, qml.PauliX(0)))

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(S)

    With grouping, the Sum gets split into two groups of observables (here
    ``[qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1))]`` and ``[qml.s_prod(3, qml.PauliX(0))]``):

    >>> tapes, fn = qml.transforms.sum_expand(tape, group=True)
    >>> for tape in tapes:
    ...     print(tape.measurements)
    [expval(PauliZ(wires=[0])), expval(2*(PauliX(wires=[1])))]
    [expval(3*(PauliX(wires=[0])))]
    """
    measurements_dict = {}  # {m_hash: measurement}
    idxs_coeffs_dict = {}  # {m_hash: [(location_idx, coeff)]}
    for idx, m in enumerate(tape.measurements):
        obs = m.obs
        coeff = 1
        if isinstance(obs, Sum) and m.return_type is Expectation:
            for summand in obs.operands:
                if isinstance(summand, SProd):
                    coeff = summand.scalar
                    summand = summand.base
                s_m = qml.expval(summand)
                if s_m.hash not in measurements_dict:
                    measurements_dict[s_m.hash] = s_m
                    idxs_coeffs_dict[s_m.hash] = [(idx, coeff)]
                else:
                    idxs_coeffs_dict[s_m.hash].append((idx, coeff))
            continue

        if isinstance(obs, SProd) and m.return_type is Expectation:
            coeff = obs.scalar
            m = qml.expval(obs.base)

        if m.hash not in measurements_dict:
            measurements_dict[m.hash] = m
            idxs_coeffs_dict[m.hash] = [(idx, coeff)]
        else:
            idxs_coeffs_dict[m.hash].append((idx, coeff))

    measurements = list(measurements_dict.values())
    idxs_coeffs = list(idxs_coeffs_dict.values())

    if group:
        m_groups = _group_measurements(measurements)
        tmp_idxs = []
        for m_group in m_groups:
            if len(m_group) == 1:
                tmp_idxs.append(idxs_coeffs[measurements.index(m_group[0])])
            else:
                tmp_idxs.append([idxs_coeffs[measurements.index(m)] for m in m_group])
        idxs_coeffs = tmp_idxs
        tapes = [
            QuantumScript(ops=tape._ops, measurements=m_group, prep=tape._prep)
            for m_group in m_groups
        ]
    else:
        tapes = [
            QuantumScript(ops=tape._ops, measurements=[m], prep=tape._prep) for m in measurements
        ]

    def processing_fn(expanded_results):
        results = defaultdict(lambda: 0)  # {m_idx: result}
        for tape_res, tape_idxs in zip(expanded_results, idxs_coeffs):
            # tape_res contains only one result
            if isinstance(tape_idxs[0], tuple):
                for idx, coeff in tape_idxs:
                    results[idx] += coeff * tape_res[0]
                continue
            # tape_res contains multiple results
            for res, idxs in zip(tape_res, tape_idxs):
                if isinstance(idxs, list):  # result is shared among measurements
                    for idx, coeff in idxs:
                        results[idx] += coeff * res
                else:
                    idx, coeff = idxs
                    results[idx] += coeff * res
        # sort results by idx
        results = [results[key] for key in sorted(results)]
        return results[0] if len(results) == 1 else results

    return tapes, processing_fn


def _group_measurements(measurements: List[MeasurementProcess]) -> List[List[MeasurementProcess]]:
    """Group observables of measurements list into qubit-wise commuting groups.

    Args:
        measurements (List[MeasurementProcess]): list of measurement processes

    Returns:
        List[List[MeasurementProcess]]: list of qubit-wise commuting measurement groups
    """
    qwc_groups = []
    for m in measurements:
        if len(m.wires) == 0:  # measurement acts on all wires: e.g. qml.counts()
            qwc_groups.append()
            continue

        op_added = False
        for idx, (wires, group) in enumerate(qwc_groups):
            if len(wires) > 0 and all(wire not in m.wires for wire in wires):
                qwc_groups[idx] = (wires + m.wires, group + [m])
                op_added = True
                break

        if not op_added:
            qwc_groups.append((m.wires, [m]))

    return [group[1] for group in qwc_groups]

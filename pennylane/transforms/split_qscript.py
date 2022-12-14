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
from typing import List

import pennylane as qml
from pennylane.measurements import ExpectationMP, MeasurementProcess
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, SProd, Sum
from pennylane.pauli import group_observables, is_pauli_word
from pennylane.tape import QuantumScript
from pennylane.wires import Wires


# pylint: disable=too-many-branches, too-many-statements
def split_qscript(qscript: QuantumScript, group=True):
    """If the tape is measuring any Hamiltonian or Sum expectation, this method splits it into
    multiple tapes of Pauli expectations and provides a function to recombine the results.

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

    We can use the ``split_qscript`` transform to generate new tapes and a classical
    post-processing function for computing the expectation value of the Hamiltonian.

    >>> tapes, fn = qml.transforms.split_qscript(tape)

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

    >>> tapes, fn = qml.transforms.split_qscript(tape)
    >>> len(tapes)
    2

    Without grouping it gets split into three groups (``[qml.PauliZ(0)]``, ``[qml.PauliX(1)]`` and ``[qml.PauliX(0)]``):

    >>> tapes, fn = qml.transforms.split_qscript(tape, group=False)
    >>> len(tapes)
    3

    Given a Sum operator,

    .. code-block:: python3

        S = qml.op_sum(qml.prod(qml.PauliY(2), qml.PauliZ(1)), qml.s_prod(0.5, qml.PauliZ(2)), qml.PauliZ(1))

    and a tape of the form,

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)

            qml.expval(S)
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))
            qml.expval(qml.PauliZ(2))

    We can use the ``split_qscript`` transform to generate new tapes and a classical
    post-processing function to speed-up the computation of the expectation value of the `Sum`.

    >>> tapes, fn = qml.transforms.split_qscript(tape, group=False)
    >>> for tape in tapes:
    ...     print(tape.measurements)
    [expval(PauliZ(wires=[0]))]
    [expval(PauliY(wires=[2]) @ PauliZ(wires=[1]))]
    [expval(PauliZ(wires=[2]))]
    [expval(PauliZ(wires=[1]))]
    [expval(PauliX(wires=[1]))]

    Five tapes are generated: the first three contain the summands of the `Sum` operator,
    and the last two contain the remaining observables. Note that the scalars of the scalar products
    have been removed. These values will be multiplied by the result obtained from the execution.

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    [0.0, -0.5, 0.0, -0.9999999999999996]

    Fewer tapes can be constructed by grouping observables acting on different wires. This can be achieved
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

    >>> tapes, fn = qml.transforms.split_qscript(tape, group=True)
    >>> for tape in tapes:
    ...     print(tape.measurements)
    [expval(PauliZ(wires=[0])), expval(PauliX(wires=[1]))]
    [expval(PauliX(wires=[0]))]
    """
    # Populate these 2 dictionaries with the unique measurement objects, the index of the
    # initial measurement on the qscript and the coefficient
    # NOTE: expval(Sum) is expanded into the expectation of each summand
    # NOTE: expval(SProd) is transformed into expval(SProd.base) and the coeff is updated
    # TODO: Separate expval(Prod) and expval(Tensor) into different measurements
    measurements_dict = {}  # {m_hash: measurement}
    idxs_coeffs_dict = {}  # {m_hash: [(location_idx, coeff)]}
    for idx, m in enumerate(qscript.measurements):
        obs = m.obs
        if isinstance(obs, Sum) and isinstance(m, ExpectationMP):
            for summand in obs.operands:
                coeff = 1
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

        if isinstance(obs, Hamiltonian) and isinstance(m, ExpectationMP):
            for o, coeff in zip(obs.ops, obs.data):
                o_m = qml.expval(o)
                if o_m.hash not in measurements_dict:
                    measurements_dict[o_m.hash] = o_m
                    idxs_coeffs_dict[o_m.hash] = [(idx, coeff)]
                else:
                    idxs_coeffs_dict[o_m.hash].append((idx, coeff))
            continue

        coeff = 1
        if isinstance(obs, SProd) and isinstance(m, ExpectationMP):
            coeff = obs.scalar
            m = qml.expval(obs.base)

        if m.hash not in measurements_dict:
            measurements_dict[m.hash] = m
            idxs_coeffs_dict[m.hash] = [(idx, coeff)]
        else:
            idxs_coeffs_dict[m.hash].append((idx, coeff))

    # Cast the dictionaries into lists (we don't need the hashed anymore)
    measurements = list(measurements_dict.values())
    idxs_coeffs = list(idxs_coeffs_dict.values())

    # Create the qscripts, group observables if group==True
    if group:
        m_groups = _group_measurements(measurements)
        # Update ``idxs_coeffs`` list such that it tracks the new ``m_groups`` list of lists
        tmp_idxs = []
        for m_group in m_groups:
            if len(m_group) == 1:
                tmp_idxs.append(idxs_coeffs[measurements.index(m_group[0])])
            else:
                tmp_idxs.append([idxs_coeffs[measurements.index(m)] for m in m_group])
        idxs_coeffs = tmp_idxs
        qscripts = [
            qscript.__class__(ops=qscript._ops, measurements=m_group, prep=qscript._prep)
            for m_group in m_groups
        ]
    else:
        qscripts = [
            qscript.__class__(ops=qscript._ops, measurements=[m], prep=qscript._prep)
            for m in measurements
        ]

    def processing_fn(expanded_results):
        results = []  # [(m_idx, result)]
        for qscript_res, qscript_idxs in zip(expanded_results, idxs_coeffs):
            if isinstance(qscript_idxs[0], tuple):  # qscript_res contains only one result
                if not qml.active_return() and len(qscript_res) == 1:  # old return types
                    qscript_res = qscript_res[0]
                for idx, coeff in qscript_idxs:
                    res = qml.math.convert_like(qml.math.dot(coeff, qscript_res), qscript_res)
                    results.append((idx, res))
                continue
            # qscript_res contains multiple results
            if qml.active_return():
                qscript_res = qml.math.stack(qscript_res)
            qscript_res = qml.math.transpose(qscript_res)  # needed when batching
            for q_res, idxs in zip(qscript_res, qscript_idxs):
                for idx, coeff in idxs:
                    res = qml.math.convert_like(qml.math.dot(coeff, q_res), q_res)
                    results.append((idx, res))

        # sum results by idx
        res_dict = {}
        for idx, res in results:
            if idx in res_dict:
                res_dict[idx] += res
            else:
                res_dict[idx] = res

        # sort results by idx
        results = [res_dict[key] for key in sorted(res_dict)]

        return results[0] if len(results) == 1 else results

    return qscripts, processing_fn


def _group_measurements(measurements: List[MeasurementProcess]) -> List[List[MeasurementProcess]]:
    """Group observables of ``measurements`` into groups with non overlapping wires.

    Args:
        measurements (List[MeasurementProcess]): list of measurement processes

    Returns:
        List[List[MeasurementProcess]]: list of groups of observables with non overlapping wires
    """
    qwc_groups = []  # [(wires (Wires), m_group (list), pauli group (bool))]
    for m in measurements:
        if len(m.wires) == 0:  # measurement acts on all wires: e.g. qml.counts()
            qwc_groups.append((m.wires, [m], True))
            continue

        op_added = False
        obs = m.obs or _pauli_z(m.wires)
        for idx, (wires, group, is_pauli) in enumerate(qwc_groups):
            if is_pauli and is_pauli_word(obs):
                obs_group = [m.obs or _pauli_z(m.wires) for m in group]
                if len(group_observables(obs_group + [obs])) == 1:
                    qwc_groups[idx] = (wires + m.wires, group + [m], True)
                    op_added = True
                    break
            if len(wires) > 0 and all(wire not in m.wires for wire in wires):
                qwc_groups[idx] = (wires + m.wires, group + [m], False)
                op_added = True
                break

        if not op_added:
            qwc_groups.append((m.wires, [m], is_pauli_word(obs)))

    return [group[1] for group in qwc_groups]


def _pauli_z(wires: Wires):
    """Generate ``PauliZ`` operator.

    Args:
        wires (Wires): wires that the operator acts on"""
    if len(wires) == 1:
        return qml.PauliZ(0)
    return Tensor(*[qml.PauliZ(w) for w in wires])

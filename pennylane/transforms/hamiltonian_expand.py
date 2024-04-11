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
from functools import partial
from typing import List, Sequence, Callable

import pennylane as qml
from pennylane.measurements import ExpectationMP, MeasurementProcess
from pennylane.ops import SProd, Sum
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms import transform


def grouping_processing_fn(res_groupings, coeff_groupings, batch_size, offset):
    """Sums up results for the expectation value of a multi-term observable when grouping is involved.

    Args:
        res_groupings (ResultBatch): The results from executing the batch of tapes with grouped observables
        coeff_groupings (List[TensorLike]): The coefficients in the same grouped structure as the results
        batch_size (Optional[int]): The batch size of the tape and corresponding results
        offset (TensorLike): A constant offset from the multi-term observable

    Returns:
        Result: The result of the expectation value for a multi-term observable
    """
    dot_products = []
    for c_group, r_group in zip(coeff_groupings, res_groupings):
        # pylint: disable=no-member
        if isinstance(r_group, (tuple, list, qml.numpy.builtins.SequenceBox)):
            r_group = qml.math.stack(r_group)
        if qml.math.shape(r_group) == ():
            r_group = qml.math.reshape(r_group, (1,))
        if batch_size:
            r_group = r_group.T

        if len(c_group) == 1 and len(r_group) != 1:
            dot_products.append(r_group * c_group)
        else:
            dot_products.append(qml.math.dot(r_group, c_group))
    summed_dot_products = qml.math.sum(qml.math.stack(dot_products), axis=0)
    interface = qml.math.get_deep_interface(res_groupings)
    return qml.math.asarray(summed_dot_products + offset, like=interface)


def _grouping_hamiltonian_expand(tape):
    """Calculate the expectation value of a tape with a multi-term observable using the grouping
    present on the observable.
    """
    hamiltonian = tape.measurements[0].obs
    if hamiltonian.grouping_indices is None:
        # explicitly selected grouping, but indices not yet computed
        hamiltonian.compute_grouping()

    coeff_groupings = []
    obs_groupings = []
    offset = 0
    coeffs, obs = hamiltonian.terms()
    for indices in hamiltonian.grouping_indices:
        group_coeffs = []
        obs_groupings.append([])
        for i in indices:
            if isinstance(obs[i], qml.Identity):
                offset += coeffs[i]
            else:
                group_coeffs.append(coeffs[i])
                obs_groupings[-1].append(obs[i])
        coeff_groupings.append(qml.math.stack(group_coeffs))
    # make one tape per grouping, measuring the
    # observables in that grouping
    tapes = []
    for obs in obs_groupings:
        new_tape = tape.__class__(tape.operations, (qml.expval(o) for o in obs), shots=tape.shots)

        new_tape = new_tape.expand(stop_at=lambda obj: True)
        tapes.append(new_tape)

    batch_size = tape.batch_size

    return tapes, partial(
        grouping_processing_fn,
        coeff_groupings=coeff_groupings,
        batch_size=batch_size,
        offset=offset,
    )


def naive_processing_fn(res, coeffs, offset):
    """Sum up the results weighted by coefficients to get the expectation value of a multi-term observable.

    Args:
        res (ResultBatch): The result of executing a batch of tapes where each tape is a different term in the observable
        coeffs (List(TensorLike)): The weights for each result in ``res``
        offset (TensorLike): Any constant offset from the multi-term observable

    Returns:
        Result: the expectation value of the multi-term observable
    """
    dot_products = []
    for c, r in zip(coeffs, res):
        if qml.math.ndim(c) == 0 and qml.math.size(r) != 1:
            dot_products.append(qml.math.squeeze(r) * c)
        else:
            dot_products.append(qml.math.dot(qml.math.squeeze(r), c))
    summed_dot_products = qml.math.sum(qml.math.stack(dot_products), axis=0)
    return qml.math.convert_like(summed_dot_products + offset, res[0])


def _naive_hamiltonian_expand(tape):
    """Calculate the expectation value of a multi-term observable using one tape per term."""
    # make one tape per observable
    hamiltonian = tape.measurements[0].obs
    tapes = []
    offset = 0
    coeffs = []
    for c, o in zip(*hamiltonian.terms()):
        if isinstance(o, qml.Identity):
            offset += c
        else:
            new_tape = tape.__class__(tape.operations, [qml.expval(o)], shots=tape.shots)
            tapes.append(new_tape)
            coeffs.append(c)

    return tapes, partial(naive_processing_fn, coeffs=coeffs, offset=offset)


@transform
def hamiltonian_expand(tape: QuantumTape, group: bool = True) -> (Sequence[QuantumTape], Callable):
    r"""
    Splits a tape measuring a Hamiltonian expectation into mutliple tapes of Pauli expectations,
    and provides a function to recombine the results.

    Args:
        tape (QNode or QuantumTape or Callable): the quantum circuit used when calculating the
            expectation value of the Hamiltonian
        group (bool): Whether to compute disjoint groups of commuting Pauli observables, leading to fewer tapes.
            If grouping information can be found in the Hamiltonian, it will be used even if group=False.

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    Given a Hamiltonian,

    .. code-block:: python3

        H = qml.Y(2) @ qml.Z(1) + 0.5 * qml.Z(2) + qml.Z(1)

    and a tape of the form,

    .. code-block:: python3

        ops = [qml.Hadamard(0), qml.CNOT((0,1)), qml.X(2)]
        tape = qml.tape.QuantumTape(ops, [qml.expval(H)])

    We can use the ``hamiltonian_expand`` transform to generate new tapes and a classical
    post-processing function for computing the expectation value of the Hamiltonian.

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape)

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    array(-0.5)

    Fewer tapes can be constructed by grouping commuting observables. This can be achieved
    by the ``group`` keyword argument:

    .. code-block:: python3

        H = qml.Hamiltonian([1., 2., 3.], [qml.Z(0), qml.X(1), qml.X(0)])

        tape = qml.tape.QuantumTape(ops, [qml.expval(H)])

    With grouping, the Hamiltonian gets split into two groups of observables (here ``[qml.Z(0)]`` and
    ``[qml.X(1), qml.X(0)]``):

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape)
    >>> len(tapes)
    2

    Without grouping it gets split into three groups (``[qml.Z(0)]``, ``[qml.X(1)]`` and ``[qml.X(0)]``):

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
    >>> len(tapes)
    3

    Alternatively, if the Hamiltonian has already computed groups, they are used even if ``group=False``:

    .. code-block:: python3

        obs = [qml.Z(0), qml.X(1), qml.X(0)]
        coeffs = [1., 2., 3.]
        H = qml.Hamiltonian(coeffs, obs, grouping_type='qwc')

        # the initialisation already computes grouping information and stores it in the Hamiltonian
        assert H.grouping_indices is not None

        tape = qml.tape.QuantumTape(ops, [qml.expval(H)])

    Grouping information has been used to reduce the number of tapes from 3 to 2:

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
    >>> len(tapes)
    2
    """

    if (
        len(tape.measurements) != 1
        or not hasattr(tape.measurements[0].obs, "grouping_indices")
        or not isinstance(tape.measurements[0], ExpectationMP)
    ):
        raise ValueError(
            "Passed tape must end in `qml.expval(H)` where H can define grouping_indices"
        )

    hamiltonian = tape.measurements[0].obs
    if len(hamiltonian.terms()[1]) == 0:
        raise ValueError(
            "The Hamiltonian in the tape has no terms defined - cannot perform the Hamiltonian expansion."
        )

    if group or hamiltonian.grouping_indices is not None:
        return _grouping_hamiltonian_expand(tape)
    return _naive_hamiltonian_expand(tape)


# pylint: disable=too-many-branches, too-many-statements
@transform
def sum_expand(tape: QuantumTape, group: bool = True) -> (Sequence[QuantumTape], Callable):
    """Splits a quantum tape measuring a Sum expectation into multiple tapes of summand
    expectations, and provides a function to recombine the results.

    Args:
        tape (.QuantumTape): the quantum tape used when calculating the expectation value
            of the Hamiltonian
        group (bool): Whether to compute disjoint groups of Pauli observables acting on different
            wires, leading to fewer tapes.

    Returns:
        tuple[Sequence[.QuantumTape], Callable]: Returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions to compute the expectation value.

    **Example**

    Given a Sum operator,

    .. code-block:: python3

        S = qml.sum(qml.prod(qml.Y(2), qml.Z(1)), qml.s_prod(0.5, qml.Z(2)), qml.Z(1))

    and a tape of the form,

    .. code-block:: python3

        ops = [qml.Hadamard(0), qml.CNOT((0,1)), qml.X(2)]
        measurements = [
            qml.expval(S),
            qml.expval(qml.Z(0)),
            qml.expval(qml.X(1)),
            qml.expval(qml.Z(2))
        ]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can use the ``sum_expand`` transform to generate new tapes and a classical
    post-processing function to speed-up the computation of the expectation value of the `Sum`.

    >>> tapes, fn = qml.transforms.sum_expand(tape, group=False)
    >>> for tape in tapes:
    ...     print(tape.measurements)
    [expval(Y(2) @ Z(1))]
    [expval(Z(2))]
    [expval(Z(1))]
    [expval(Z(0))]
    [expval(X(1))]

    Five tapes are generated: the first three contain the summands of the `Sum` operator,
    and the last two contain the remaining observables. Note that the scalars of the scalar products
    have been removed. In the processing function, these values will be multiplied by the result obtained
    from executing the tapes.

    Additionally, the observable expval(Z(2)) occurs twice in the original tape, but only once
    in the transformed tapes. When there are multipe identical measurements in the circuit, the measurement
    is performed once and the outcome is copied when obtaining the final result. This will also be resolved
    when the processing function is applied.

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    [-0.5, 0.0, 0.0, -0.9999999999999996]

    Fewer tapes can be constructed by grouping observables acting on different wires. This can be achieved
    by the ``group`` keyword argument:

    .. code-block:: python3

        S = qml.sum(qml.Z(0), qml.s_prod(2, qml.X(1)), qml.s_prod(3, qml.X(0)))

        ops = [qml.Hadamard(0), qml.CNOT((0,1)), qml.X(2)]
        tape = qml.tape.QuantumTape(ops, [qml.expval(S)])

    With grouping, the Sum gets split into two groups of observables (here
    ``[qml.Z(0), qml.s_prod(2, qml.X(1))]`` and ``[qml.s_prod(3, qml.X(0))]``):

    >>> tapes, fn = qml.transforms.sum_expand(tape, group=True)
    >>> for tape in tapes:
    ...     print(tape.measurements)
    [expval(Z(0)), expval(X(1))]
    [expval(X(0))]
    """
    # Populate these 2 dictionaries with the unique measurement objects, the index of the
    # initial measurement on the tape and the coefficient
    # NOTE: expval(Sum) is expanded into the expectation of each summand
    # NOTE: expval(SProd) is transformed into expval(SProd.base) and the coeff is updated
    measurements_dict = {}  # {m_hash: measurement}
    idxs_coeffs_dict = {}  # {m_hash: [(location_idx, coeff)]}
    for idx, m in enumerate(tape.measurements):
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

        coeff = 1 if isinstance(m, ExpectationMP) else None
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

    # Create the tapes, group observables if group==True
    # pylint: disable=too-many-nested-blocks
    if group:
        m_groups = _group_measurements(measurements)
        # Update ``idxs_coeffs`` list such that it tracks the new ``m_groups`` list of lists
        tmp_idxs = []
        for m_group in m_groups:
            if len(m_group) == 1:
                # pylint: disable=undefined-loop-variable
                for i, m in enumerate(measurements):
                    if m is m_group[0]:
                        break
                tmp_idxs.append(idxs_coeffs[i])
            else:
                inds = []
                for mp in m_group:
                    # pylint: disable=undefined-loop-variable
                    for i, m in enumerate(measurements):
                        if m is mp:
                            break
                    inds.append(idxs_coeffs[i])
                tmp_idxs.append(inds)

        idxs_coeffs = tmp_idxs
        qscripts = [
            QuantumScript(ops=tape.operations, measurements=m_group, shots=tape.shots)
            for m_group in m_groups
        ]
    else:
        qscripts = [
            QuantumScript(ops=tape.operations, measurements=[m], shots=tape.shots)
            for m in measurements
        ]

    def processing_fn(expanded_results):
        results = []  # [(m_idx, result)]
        for qscript_res, qscript_idxs in zip(expanded_results, idxs_coeffs):
            if isinstance(qscript_idxs[0], tuple):  # qscript_res contains only one result
                for idx, coeff in qscript_idxs:
                    results.append((idx, qscript_res if coeff is None else coeff * qscript_res))
                continue
            # qscript_res contains multiple results
            for res, idxs in zip(qscript_res, qscript_idxs):
                for idx, coeff in idxs:
                    results.append((idx, res if coeff is None else coeff * res))

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
    qwc_groups = []
    for m in measurements:
        if len(m.wires) == 0:  # measurement acts on all wires: e.g. qml.counts()
            qwc_groups.append((m.wires, [m]))
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

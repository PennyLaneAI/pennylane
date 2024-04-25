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
from typing import List, Sequence, Callable, Tuple

import pennylane as qml
from pennylane.measurements import ExpectationMP, MeasurementProcess, Shots
from pennylane.ops import SProd, Sum, Prod
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.typing import ResultBatch


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
        if batch_size and batch_size > 1 and len(c_group) > 1:
            r_group = qml.math.moveaxis(r_group, -1, -2)

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

    return tapes, partial(
        grouping_processing_fn,
        coeff_groupings=coeff_groupings,
        batch_size=tape.batch_size,
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
    if len(dot_products) == 0:
        return offset
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


def _group_measurements(
    measurements: Sequence[MeasurementProcess], indices_and_coeffs: List[List[Tuple[int, float]]]
) -> (List[List[MeasurementProcess]], List[List[Tuple[int, int, float]]]):
    """Groups measurements that does not have overlapping wires.

    Returns:
        measurements (List[List[MeasurementProcess]]): the grouped measurements. Each group
            is a list of single-term observable measurements.
        indices_and_coeffs (List[List[Tuple[int, float]]]): the indices and coefficients of
            the single-term measurements to be combined for each original measurement. This
            is a list of lists of tuples. Each list within the list corresponds to an original
            measurement, and the tuples within the list refer to the single-term measurements
            to be combined for this original measurement. Each tuple is of the form ``(group_idx,
            sm_idx, coeff)``, where ``group_idx`` locates the group that this single-term
            measurement belongs to, ``sm_idx`` is the index of the measurement within the group,
            and ``coeff`` is the coefficient of the measurement.

    """

    groups = []  # Groups of measurements and the wires each group acts on
    new_indices_and_coeffs = []
    # Tracks the measurements that have already been grouped, and their location within the groups
    grouped_sm_indices = {}

    for mp_indices_and_coeffs in indices_and_coeffs:
        # For each original measurement, add each single-term measurement associated
        # with it to an existing group or a new group.

        new_mp_indices_and_coeffs = []

        for sm_idx, coeff in mp_indices_and_coeffs:
            # For each single-term measurement currently associated with this measurement

            if sm_idx in grouped_sm_indices:
                # If this single-term measurement has already been grouped, find the group
                # that it belongs to and its index within the group, add it to the new list
                # of indices and coefficients
                new_mp_indices_and_coeffs.append((*grouped_sm_indices[sm_idx], coeff))
                continue

            m = measurements[sm_idx]

            # If this measurement is added to an existing group, the sm_index will be the
            # length of the group. If the measurement is added to a new group, the sm_index
            # should be 0 as it's the first measurement in the group, and the group index
            # will be the current length of the groups.

            if len(m.wires) == 0:  # measurement acting on all wires
                groups.append((m.wires, [m]))
                new_mp_indices_and_coeffs.append((len(groups) - 1, 0, coeff))
                grouped_sm_indices[sm_idx] = (len(groups) - 1, 0)
                continue

            op_added = False
            for grp_idx, (wires, group) in enumerate(groups):
                if len(wires) != 0 and len(qml.wires.Wires.shared_wires([wires, m.wires])) == 0:
                    group.append(m)
                    groups[grp_idx] = (wires + m.wires, group)
                    new_mp_indices_and_coeffs.append((grp_idx, len(group) - 1, coeff))
                    grouped_sm_indices[sm_idx] = (grp_idx, len(group) - 1)
                    op_added = True
                    break

            if not op_added:
                groups.append((m.wires, [m]))
                new_mp_indices_and_coeffs.append((len(groups) - 1, 0, coeff))
                grouped_sm_indices[sm_idx] = (len(groups) - 1, 0)

        new_indices_and_coeffs.append(new_mp_indices_and_coeffs)

    return [group[1] for group in groups], new_indices_and_coeffs


def _sum_expand_processing_fn_grouping(
    res: ResultBatch,
    group_sizes: List[int],
    shots: Shots,
    indices_and_coeffs: List[List[Tuple[int, int, float]]],
    offsets: List[int],
):
    """The processing function for sum_expand with grouping."""

    res_for_each_mp = []
    for mp_indices_and_coeffs, offset in zip(indices_and_coeffs, offsets):
        sub_res = []
        coeffs = []
        for group_idx, sm_idx, coeff in mp_indices_and_coeffs:
            r_group = res[group_idx]
            group_size = group_sizes[group_idx]
            if shots.has_partitioned_shots:
                r_group = qml.math.stack(r_group, axis=0)
                if group_size > 1:
                    # Move dimensions around to make things work
                    r_group = qml.math.moveaxis(r_group, 0, 1)
            sub_res.append(r_group[sm_idx] if group_size > 1 else r_group)
            coeffs.append(coeff)
        res_for_each_mp.append(naive_processing_fn(sub_res, coeffs, offset))
    if shots.has_partitioned_shots:
        res_for_each_mp = qml.math.stack(res_for_each_mp, axis=0)
        res_for_each_mp = qml.math.moveaxis(res_for_each_mp, 0, -1)
    return res_for_each_mp[0] if len(res_for_each_mp) == 1 else res_for_each_mp


def _sum_expand_processing_fn(
    res: ResultBatch,
    shots: Shots,
    indices_and_coeffs: List[List[Tuple[int, float]]],
    offsets: List[int],
):
    """The processing function for sum_expand without grouping."""

    res_for_each_mp = []
    for mp_indices_and_coeffs, offset in zip(indices_and_coeffs, offsets):
        sub_res = []
        coeffs = []
        # For each original measurement, locate the results corresponding to each single-term
        # measurement, and construct a subset of results to be processed.
        for sm_idx, coeff in mp_indices_and_coeffs:
            sub_res.append(res[sm_idx])
            coeffs.append(coeff)
        res_for_each_mp.append(naive_processing_fn(sub_res, coeffs, offset))
    if shots.has_partitioned_shots:
        res_for_each_mp = qml.math.stack(res_for_each_mp, axis=0)
        # Move dimensions around to make things work.
        res_for_each_mp = qml.math.moveaxis(res_for_each_mp, 0, -1)
    return res_for_each_mp[0] if len(res_for_each_mp) == 1 else res_for_each_mp


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

    # The dictionary of all unique single-term observable measurements, and their indices
    # within the list of all single-term observable measurements.
    single_term_obs_measurements = {}

    # Indices and coefficients of single-term observable measurements to be combined for each
    # original measurement. Each element is a list of tuples of the form (index, coeff)
    all_sm_indices_and_coeffs = []

    # Offsets associated with each original measurement in the tape.
    offsets = []

    sm_idx = 0  # Tracks the number of unique single-term observable measurements
    for mp in tape.measurements:
        obs = mp.obs
        offset = 0
        # Indices and coefficients of each single-term observable measurement to be
        # combined for this original measurement.
        sm_indices_and_coeffs = []
        if isinstance(mp, ExpectationMP) and isinstance(obs, (Sum, Prod, SProd)):
            if isinstance(obs, SProd):
                # This is necessary because SProd currently does not flatten into
                # multiple terms if the base is a sum, which is needed here.
                obs = obs.simplify()
            # Break the observable into terms, and construct an ExpectationMP with each term.
            for c, o in zip(*obs.terms()):
                # If the observable is an identity, track it with a constant offset
                if isinstance(o, qml.Identity):
                    offset += c
                # If the single-term measurement already exists, it can be reused by all
                # original measurements. In this case, add the existing single-term measurement
                # to the list corresponding to this original measurement.
                # pylint: disable=superfluous-parens
                elif (sm := qml.expval(o)) in single_term_obs_measurements:
                    sm_indices_and_coeffs.append((single_term_obs_measurements[sm], c))
                # Otherwise, add this new measurement to the list of single-term measurements.
                else:
                    single_term_obs_measurements[sm] = sm_idx
                    sm_indices_and_coeffs.append((sm_idx, c))
                    sm_idx += 1
        else:
            # For all other measurement types, simply add them to the list of measurements.
            if mp not in single_term_obs_measurements:
                single_term_obs_measurements[mp] = sm_idx
                sm_indices_and_coeffs.append((sm_idx, 1))
                sm_idx += 1
            else:
                sm_indices_and_coeffs.append((single_term_obs_measurements[mp], 1))

        all_sm_indices_and_coeffs.append(sm_indices_and_coeffs)
        offsets.append(offset)

    measurements = list(single_term_obs_measurements.keys())
    if group:
        groups, indices_and_coeffs = _group_measurements(measurements, all_sm_indices_and_coeffs)
        tapes = [tape.__class__(tape.operations, m_group, shots=tape.shots) for m_group in groups]
        group_sizes = [len(m_group) for m_group in groups]
        return tapes, partial(
            _sum_expand_processing_fn_grouping,
            indices_and_coeffs=indices_and_coeffs,
            group_sizes=group_sizes,
            shots=tape.shots,
            offsets=offsets,
        )

    tapes = [tape.__class__(tape.operations, [m], shots=tape.shots) for m in measurements]
    return tapes, partial(
        _sum_expand_processing_fn,
        indices_and_coeffs=all_sm_indices_and_coeffs,
        shots=tape.shots,
        offsets=offsets,
    )

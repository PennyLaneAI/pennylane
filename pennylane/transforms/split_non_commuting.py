# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains the tape transform that splits a tape into tapes measuring commuting observables.
"""

# pylint: disable=too-many-arguments,too-many-boolean-expressions

from functools import partial, wraps
from typing import Optional

import pennylane as qml
from pennylane.measurements import ExpectationMP, MeasurementProcess, StateMP
from pennylane.ops import Prod, SProd, Sum
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn, Result, ResultBatch, TensorLike, Union


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


def shot_vector_support(initial_postprocessing: PostprocessingFn) -> PostprocessingFn:
    """Convert a postprocessing function to one with shot vector support."""

    @wraps(initial_postprocessing)
    def shot_vector_postprocessing(results):
        return tuple(initial_postprocessing(r) for r in zip(*results))

    return shot_vector_postprocessing


@transform
def split_non_commuting(
    tape: QuantumScript, grouping_strategy: Optional[str] = "default"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Splits a circuit into tapes measuring groups of commuting observables.

    Args:
        tape (QNode or QuantumScript or Callable): The quantum circuit to be split.
        grouping_strategy (str): The strategy to use for computing disjoint groups of
            commuting observables, can be ``"default"``, ``"wires"``, ``"qwc"``,
            or ``None`` to disable grouping.

    Returns:
        qnode (QNode) or tuple[List[QuantumScript], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    .. note::
        This transform splits expectation values of sums into separate terms, and also distributes the terms into
        multiple executions if there are terms that do not commute with one another. For state-based simulators
        that are able to handle non-commuting measurements in a single execution, but don't natively support sums
        of observables, consider :func:`split_to_single_terms <pennylane.transforms.split_to_single_terms>` instead.

    **Examples:**

    This transform allows us to transform a QNode measuring multiple observables into multiple
    circuit executions, each measuring a group of commuting observables.

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.transforms.split_non_commuting
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(1)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
            ]

    Instead of decorating the QNode, we can also create a new function that yields the same
    result in the following way:

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(1)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
            ]

        circuit = qml.transforms.split_non_commuting(circuit)

    Internally, the QNode is split into multiple circuits when executed:

    >>> print(qml.draw(circuit)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)─┤ ╭<Z@Z>  <Z>
    1: ──RX(0.79)─┤ ╰<Z@Z>
    <BLANKLINE>
    0: ──RY(0.79)─┤  <X>
    1: ──RX(0.79)─┤  <Y>
    <BLANKLINE>
    0: ──RY(0.79)─┤ ╭<X@Z>
    1: ──RX(0.79)─┤ ╰<X@Z>

    Note that the observable ``Y(1)`` occurs twice in the original QNode, but only once in the
    transformed circuits. When there are multiple expectation value measurements that rely on
    the same observable, this observable is measured only once, and the result is copied to each
    original measurement.

    While internally multiple tapes are created, the end result has the same ordering as the user
    provides in the return statement. Executing the above QNode returns the original ordering of
    the expectation values.

    >>> circuit([np.pi/4, np.pi/4])
    [0.7071067811865475, -0.7071067811865475, 0.5, 0.5]

    There are two algorithms used to compute disjoint groups of commuting observables: ``"qwc"``
    grouping uses :func:`~pennylane.pauli.group_observables` which computes groups of qubit-wise
    commuting observables, producing the fewest number of circuit executions, but can be expensive
    to compute for large multi-term Hamiltonians, while ``"wires"`` grouping simply ensures
    that no circuit contains two measurements with overlapping wires, disregarding commutativity
    between the observables being measured.

    The ``grouping_strategy`` keyword argument can be used to specify the grouping strategy. By
    default, qwc grouping is used whenever possible, except when the circuit contains multiple
    measurements that includes an expectation value of a ``qml.Hamiltonian``, in which case wires
    grouping is used in case the Hamiltonian is very large, to save on classical runtime. To force
    qwc grouping in all cases, set ``grouping_strategy="qwc"``. Similarly, to force wires grouping,
    set ``grouping_strategy="wires"``:

    .. code-block:: python3

        @functools.partial(qml.transforms.split_non_commuting, grouping_strategy="wires")
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(1)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
            ]

    In this case, four circuits are created as follows:

    >>> print(qml.draw(circuit)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)─┤  <X>
    1: ──RX(0.79)─┤  <Y>
    <BLANKLINE>
    0: ──RY(0.79)─┤ ╭<Z@Z>
    1: ──RX(0.79)─┤ ╰<Z@Z>
    <BLANKLINE>
    0: ──RY(0.79)─┤ ╭<X@Z>
    1: ──RX(0.79)─┤ ╰<X@Z>
    <BLANKLINE>
    0: ──RY(0.79)─┤  <Z>
    1: ──RX(0.79)─┤

    Alternatively, to disable grouping completely, set ``grouping_strategy=None``:

    .. code-block:: python3

        @functools.partial(qml.transforms.split_non_commuting, grouping_strategy=None)
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(1)),
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.Z(1) + 0.5 * qml.Y(1) + qml.Z(0)),
            ]

    In this case, each observable is measured in a separate circuit execution.

    >>> print(qml.draw(circuit)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)─┤  <X>
    1: ──RX(0.79)─┤
    <BLANKLINE>
    0: ──RY(0.79)─┤
    1: ──RX(0.79)─┤  <Y>
    <BLANKLINE>
    0: ──RY(0.79)─┤ ╭<Z@Z>
    1: ──RX(0.79)─┤ ╰<Z@Z>
    <BLANKLINE>
    0: ──RY(0.79)─┤ ╭<X@Z>
    1: ──RX(0.79)─┤ ╰<X@Z>
    <BLANKLINE>
    0: ──RY(0.79)─┤  <Z>
    1: ──RX(0.79)─┤

    Note that there is an exception to the above rules: if the circuit only contains a single
    expectation value measurement of a ``Hamiltonian`` or ``Sum`` with pre-computed grouping
    indices, the grouping information will be used regardless of the requested ``grouping_strategy``

    .. details::
        :title: Usage Details

        Internally, this function works with tapes. We can create a tape with multiple
        measurements of non-commuting observables:

        .. code-block:: python3

            measurements = [
                qml.expval(qml.Z(0) @ qml.Z(1)),
                qml.expval(qml.X(0) @ qml.X(1)),
                qml.expval(qml.Z(0)),
                qml.expval(qml.X(0))
            ]
            tape = qml.tape.QuantumScript(measurements=measurements)
            tapes, processing_fn = qml.transforms.split_non_commuting(tape)

        Now ``tapes`` is a list of two tapes, each contains a group of commuting observables:

        >>> [t.measurements for t in tapes]
        [[expval(Z(0) @ Z(1)), expval(Z(0))], [expval(X(0) @ X(1)), expval(X(0))]]

        The processing function becomes important as the order of the inputs has been modified.

        >>> dev = qml.device("default.qubit", wires=2)
        >>> result_batch = [dev.execute(t) for t in tapes]
        >>> result_batch
        [(1.0, 1.0), (0.0, 0.0)]

        The processing function can be used to reorganize the results:

        >>> processing_fn(result_batch)
        (1.0, 0.0, 1.0, 0.0)

        Measurements that accept both observables and ``wires`` so that e.g. ``qml.counts``,
        ``qml.probs`` and ``qml.sample`` can also be used. When initialized using only ``wires``,
        these measurements are interpreted as measuring with respect to the observable
        ``qml.Z(wires[0])@qml.Z(wires[1])@...@qml.Z(wires[len(wires)-1])``

        .. code-block:: python3

            measurements = [
                qml.expval(qml.X(0)),
                qml.probs(wires=[1]),
                qml.probs(wires=[0, 1])
            ]
            tape = qml.tape.QuantumScript(measurements=measurements)
            tapes, processing_fn = qml.transforms.split_non_commuting(tape)

        This results in two tapes, each with commuting measurements:

        >>> [t.measurements for t in tapes]
        [[expval(X(0)), probs(wires=[1])], [probs(wires=[0, 1])]]

    """
    if len(tape.measurements) == 0:
        return [tape], null_postprocessing

    # Special case for a single measurement of a Sum, in which case
    # the grouping information can be computed and cached in the observable.
    if (
        len(tape.measurements) == 1
        and isinstance(tape.measurements[0], ExpectationMP)
        and isinstance(tape.measurements[0].obs, Sum)
        and (
            (
                grouping_strategy in ("default", "qwc")
                and all(qml.pauli.is_pauli_word(o) for o in tape.measurements[0].obs.terms()[1])
            )
            or tape.measurements[0].obs.grouping_indices is not None
        )
    ):
        return _split_ham_with_grouping(tape)

    single_term_obs_mps, offsets = _split_all_multi_term_obs_mps(tape)

    if grouping_strategy is None:
        measurements = list(single_term_obs_mps.keys())
        tapes = [tape.copy(measurements=[m]) for m in measurements]
        fn = partial(
            _processing_fn_no_grouping,
            single_term_obs_mps=single_term_obs_mps,
            offsets=offsets,
            batch_size=tape.batch_size,
        )
        if tape.shots.has_partitioned_shots:
            fn = shot_vector_support(fn)
        return tapes, fn

    if grouping_strategy == "wires" or any(
        m.obs is not None and not qml.pauli.is_pauli_word(m.obs) for m in single_term_obs_mps
    ):
        # TODO: here we fall back to wire-based grouping if any of the observables in the tape
        #       is not a pauli word. As a result, adding a single measurement to a circuit could
        #       significantly increase the number of circuit executions. We should be able to
        #       separate the logic for pauli-word observables and non-pauli-word observables,
        #       putting non-pauli-word observables in separate wire-based groups, but using qwc
        #       based grouping for the rest of the observables. [sc-79686]
        return _split_using_wires_grouping(tape, single_term_obs_mps, offsets)

    return _split_using_qwc_grouping(tape, single_term_obs_mps, offsets)


def _split_ham_with_grouping(tape: qml.tape.QuantumScript):
    """Splits a tape measuring a single Sum and group commuting observables."""

    obs = tape.measurements[0].obs
    if obs.grouping_indices is None:
        obs.compute_grouping()

    coeffs, obs_list = obs.terms()

    # The constant offset of the Sum, typically arising from Identity terms.
    offset = 0

    # A dictionary for measurements of each unique single-term observable, mapped to the
    # indices of the original measurements it belongs to, its coefficients, the index of
    # the group it belongs to, and the index of the measurement in the group.
    single_term_obs_mps = {}

    # A list of lists for each group of commuting measurement processes.
    mp_groups = []

    # The number of measurements in each group
    group_sizes = []

    # obs.grouping_indices is a list of lists, where each list contains the indices of
    # observables that belong in each group.
    for group_idx, obs_indices in enumerate(obs.grouping_indices):
        mp_group = []
        group_size = 0
        for obs_idx in obs_indices:
            # Do not measure Identity terms, but track their contribution with the offset.
            if isinstance(obs_list[obs_idx], qml.Identity):
                offset += coeffs[obs_idx]
            else:
                new_mp = qml.expval(obs_list[obs_idx])
                if new_mp in single_term_obs_mps:
                    # If the Sum contains duplicate observables, it can be reused,
                    # and the coefficients for each duplicate should be combined.
                    single_term_obs_mps[new_mp] = (
                        single_term_obs_mps[new_mp][0],
                        [single_term_obs_mps[new_mp][1][0] + coeffs[obs_idx]],
                        single_term_obs_mps[new_mp][2],
                        single_term_obs_mps[new_mp][3],
                    )
                else:
                    mp_group.append(new_mp)
                    single_term_obs_mps[new_mp] = (
                        [0],
                        [coeffs[obs_idx]],
                        group_idx,
                        group_size,  # the index of this measurement in the group
                    )
                    group_size += 1

        if group_size > 0:
            mp_groups.append(mp_group)
            group_sizes.append(group_size)

    tapes = [tape.copy(measurements=mps) for mps in mp_groups]
    fn = partial(
        _processing_fn_with_grouping,
        single_term_obs_mps=single_term_obs_mps,
        offsets=[offset],
        group_sizes=group_sizes,
        batch_size=tape.batch_size,
    )
    if tape.shots.has_partitioned_shots:
        fn = shot_vector_support(fn)
    return tapes, fn


def _split_using_qwc_grouping(
    tape: qml.tape.QuantumScript,
    single_term_obs_mps: dict[MeasurementProcess, tuple[list[int], list[Union[float, TensorLike]]]],
    offsets: list[TensorLike],
):
    """Split tapes using group_observables in the Pauli module.

    Args:
        tape (~qml.tape.QuantumScript): The tape to be split.
        single_term_obs_mps (Dict[MeasurementProcess, Tuple[List[int], List[TensorLike]]]): A dictionary
            of measurements of each unique single-term observable, mapped to the indices of the
            original measurements it belongs to, and its coefficients.
        offsets (List[TensorLike]): Offsets associated with each original measurement in the tape.

    """

    # The legacy device does not support state measurements combined with any other
    # measurement, so each state measurement must be in its own tape.
    state_measurements = [m for m in single_term_obs_mps if isinstance(m, StateMP)]

    measurements = [m for m in single_term_obs_mps if not isinstance(m, StateMP)]
    obs_list = [_mp_to_obs(m, tape) for m in measurements]
    index_groups = []
    if len(obs_list) > 0:
        index_groups = qml.pauli.compute_partition_indices(obs_list)

    # A dictionary for measurements of each unique single-term observable, mapped to the
    # indices of the original measurements it belongs to, its coefficients, the index of
    # the group it belongs to, and the index of the measurement in the group.
    single_term_obs_mps_grouped = {}

    mp_groups = [[] for _ in index_groups]
    group_sizes = []
    for group_idx, obs_indices in enumerate(index_groups):
        group_size = 0
        for obs_idx in obs_indices:
            new_mp = measurements[obs_idx]
            mp_groups[group_idx].append(new_mp)
            single_term_obs_mps_grouped[new_mp] = (
                *single_term_obs_mps[new_mp],
                group_idx,
                group_size,
            )
            group_size += 1
        group_sizes.append(group_size)

    for state_mp in state_measurements:
        mp_groups.append([state_mp])
        single_term_obs_mps_grouped[state_mp] = (
            *single_term_obs_mps[state_mp],
            len(mp_groups) - 1,
            0,
        )
        group_sizes.append(1)
    tapes = [tape.copy(measurements=mps) for mps in mp_groups]
    fn = partial(
        _processing_fn_with_grouping,
        single_term_obs_mps=single_term_obs_mps_grouped,
        offsets=offsets,
        group_sizes=group_sizes,
        batch_size=tape.batch_size,
    )
    if tape.shots.has_partitioned_shots:
        fn = shot_vector_support(fn)
    return tapes, fn


def _split_using_wires_grouping(
    tape: qml.tape.QuantumScript,
    single_term_obs_mps: dict[MeasurementProcess, tuple[list[int], list[Union[float, TensorLike]]]],
    offsets: list[Union[float, TensorLike]],
):
    """Split tapes by grouping observables based on overlapping wires.

    Args:
        tape (~qml.tape.QuantumScript): The tape to be split.
        single_term_obs_mps (Dict[MeasurementProcess, Tuple[List[int], List[Union[float, TensorLike]]]]): A dictionary
            of measurements of each unique single-term observable, mapped to the indices of the
            original measurements it belongs to, and its coefficients.
        offsets (List[Union[float, TensorLike]]): Offsets associated with each original measurement in the tape.

    """

    mp_groups = []
    wires_for_each_group = []
    group_sizes = []

    # A dictionary for measurements of each unique single-term observable, mapped to the
    # indices of the original measurements it belongs to, its coefficient, the index of
    # the group it belongs to, and the index of the measurement in the group.
    single_term_obs_mps_grouped = {}
    num_groups = 0

    for smp, (mp_indices, coeffs) in single_term_obs_mps.items():

        if len(smp.wires) == 0:  # measurement acting on all wires
            mp_groups.append([smp])
            wires_for_each_group.append(tape.wires)
            group_sizes.append(1)
            single_term_obs_mps_grouped[smp] = (mp_indices, coeffs, num_groups, 0)
            num_groups += 1
            continue

        group_idx = 0
        added_to_existing_group = False
        while not added_to_existing_group and group_idx < num_groups:
            wires = wires_for_each_group[group_idx]
            if len(wires) != 0 and len(qml.wires.Wires.shared_wires([wires, smp.wires])) == 0:
                mp_groups[group_idx].append(smp)
                wires_for_each_group[group_idx] += smp.wires
                single_term_obs_mps_grouped[smp] = (
                    mp_indices,
                    coeffs,
                    group_idx,
                    group_sizes[group_idx],
                )
                group_sizes[group_idx] += 1
                added_to_existing_group = True
            group_idx += 1

        if not added_to_existing_group:
            mp_groups.append([smp])
            wires_for_each_group.append(smp.wires)
            group_sizes.append(1)
            single_term_obs_mps_grouped[smp] = (mp_indices, coeffs, num_groups, 0)
            num_groups += 1

    tapes = [tape.copy(measurements=mps) for mps in mp_groups]
    fn = partial(
        _processing_fn_with_grouping,
        single_term_obs_mps=single_term_obs_mps_grouped,
        offsets=offsets,
        group_sizes=group_sizes,
        batch_size=tape.batch_size,
    )
    if tape.shots.has_partitioned_shots:
        fn = shot_vector_support(fn)
    return tapes, fn


def _split_all_multi_term_obs_mps(tape: qml.tape.QuantumScript):
    """Splits all multi-term observables in a tape to measurements of single-term observables.

    Args:
        tape (~qml.tape.QuantumScript): The tape with measurements to split.

    Returns:
        single_term_obs_mps (Dict[MeasurementProcess, Tuple[List[int], List[Union[float, TensorLike]]]]): A
            dictionary for measurements of each unique single-term observable, mapped to the
            indices of the original measurements it belongs to, and its coefficients.
        offsets (List[Union[float, TensorLike]]): Offsets associated with each original measurement in the tape.

    """

    # The dictionary for measurements of each unique single-term observable, mapped the indices
    # of the original measurements it belongs to, and its coefficients.
    single_term_obs_mps = {}

    # Offsets associated with each original measurement in the tape (from Identity)
    offsets = []

    for mp_idx, mp in enumerate(tape.measurements):
        obs = mp.obs
        offset = 0
        if isinstance(mp, ExpectationMP) and isinstance(obs, (Sum, Prod, SProd)):
            # Break the observable into terms, and construct an ExpectationMP with each term.
            for c, o in zip(*obs.terms()):
                # If the observable is an identity, track it with a constant offset
                if isinstance(o, qml.Identity):
                    offset += c
                # If the single-term measurement already exists, it can be reused by all original
                # measurements. In this case, add the existing single-term measurement to the list
                # corresponding to this original measurement.
                # pylint: disable=superfluous-parens
                elif (sm := qml.expval(o)) in single_term_obs_mps:
                    single_term_obs_mps[sm][0].append(mp_idx)
                    single_term_obs_mps[sm][1].append(c)
                # Otherwise, add this new measurement to the list of single-term measurements.
                else:
                    single_term_obs_mps[sm] = ([mp_idx], [c])
        elif isinstance(obs, qml.Identity):
            offset += 1
        else:
            if isinstance(obs, (SProd, Prod)):
                obs = obs.simplify()
            if isinstance(obs, Sum):
                raise RuntimeError(
                    f"Cannot split up terms in sums for MeasurementProcess {type(mp)}"
                )
            # For all other measurement types, simply add them to the list of measurements.
            if mp not in single_term_obs_mps:
                single_term_obs_mps[mp] = ([mp_idx], [1])
            else:
                single_term_obs_mps[mp][0].append(mp_idx)
                single_term_obs_mps[mp][1].append(1)

        offsets.append(offset)

    return single_term_obs_mps, offsets


def _processing_fn_no_grouping(
    res: ResultBatch,
    single_term_obs_mps: dict[MeasurementProcess, tuple[list[int], list[Union[float, TensorLike]]]],
    offsets: list[Union[float, TensorLike]],
    batch_size: Union[None, int],
):
    """Postprocessing function for the split_non_commuting transform without grouping.

    Args:
        res (ResultBatch): The results from executing the tapes. Assumed to have a shape
            of (n_groups [,n_shots] [,n_mps] [,batch_size])
        single_term_obs_mps (Dict[MeasurementProcess, Tuple[List[int], List[Union[float, TensorLike]]]]): A dictionary
            of measurements of each unique single-term observable, mapped to the indices of the
            original measurements it belongs to, and its coefficients.
        offsets (List[Union[float, TensorLike]]): Offsets associated with each original measurement in the tape.
        shots (Shots): The shots settings of the original tape.

    """

    res_batch_for_each_mp = [[] for _ in offsets]
    coeffs_for_each_mp = [[] for _ in offsets]

    for smp_idx, (_, (mp_indices, coeffs)) in enumerate(single_term_obs_mps.items()):
        for mp_idx, coeff in zip(mp_indices, coeffs):
            res_batch_for_each_mp[mp_idx].append(res[smp_idx])
            coeffs_for_each_mp[mp_idx].append(coeff)

    result_shape = (batch_size,) if batch_size and batch_size > 1 else ()
    # Sum up the results for each original measurement

    res_for_each_mp = [
        _sum_terms(_sub_res, coeffs, offset, result_shape)
        for _sub_res, coeffs, offset in zip(res_batch_for_each_mp, coeffs_for_each_mp, offsets)
    ]
    # res_for_each_mp should have shape (n_mps, [,n_shots] [,batch_size])
    if len(res_for_each_mp) == 1:
        return res_for_each_mp[0]

    return tuple(res_for_each_mp)


def _processing_fn_with_grouping(
    res: ResultBatch,
    single_term_obs_mps: dict[
        MeasurementProcess, tuple[list[int], list[Union[float, TensorLike]], int, int]
    ],
    offsets: list[TensorLike],
    group_sizes: list[int],
    batch_size: int,
):
    """Postprocessing function for the split_non_commuting transform with grouping.

    Args:
        res (ResultBatch): The results from executing the tapes. Assumed to have a shape
            of (n_groups [,n_shots] [,n_mps_in_group] [,batch_size])
        single_term_obs_mps (Dict[MeasurementProcess, Tuple[List[int], List[Union[float, TensorLike]], int, int]]):
            A dictionary of measurements of each unique single-term observable, mapped to the
            indices of the original measurements it belongs to, its coefficients, its group
            index, and the index of the measurement within the group.
        offsets (List[Union[float, TensorLike]]): Offsets associated with each original measurement in the tape.
        group_sizes (List[int]): The number of tapes in each group.
        shots (Shots): The shots setting of the original tape.

    Returns:
        The results combined into a single result for each original measurement.

    """

    res_batch_for_each_mp = [[] for _ in offsets]  # ([n_mps] [,n_shots] [,batch_size])
    coeffs_for_each_mp = [[] for _ in offsets]

    for _, (mp_indices, coeffs, group_idx, mp_idx_in_group) in single_term_obs_mps.items():

        res_group = res[group_idx]  # ([n_shots] [,n_mps] [,batch_size])
        group_size = group_sizes[group_idx]

        # If there is only one term in the group, the n_mps dimension would have
        # been squeezed out, use the entire result directly.
        sub_res = res_group if group_size == 1 else res_group[mp_idx_in_group]

        # Add this result to the result batch for the corresponding original measurement
        for mp_idx, coeff in zip(mp_indices, coeffs):
            res_batch_for_each_mp[mp_idx].append(sub_res)
            coeffs_for_each_mp[mp_idx].append(coeff)

    result_shape = (batch_size,) if batch_size and batch_size > 1 else ()

    # Sum up the results for each original measurement
    res_for_each_mp = [
        _sum_terms(_sub_res, coeffs, offset, result_shape)
        for _sub_res, coeffs, offset in zip(res_batch_for_each_mp, coeffs_for_each_mp, offsets)
    ]

    # res_for_each_mp should have shape (n_mps, [,n_shots] [,batch_size])
    if len(res_for_each_mp) == 1:
        return res_for_each_mp[0]

    return tuple(res_for_each_mp)


def _sum_terms(
    res: ResultBatch,
    coeffs: list[Union[float, TensorLike]],
    offset: Union[float, TensorLike],
    shape: tuple,
) -> Result:
    """Sum results from measurements of multiple terms in a multi-term observable."""
    if (
        coeffs
        and not qml.math.is_abstract(coeffs[0])
        and not qml.math.is_abstract(offset)
        and coeffs == [1]
        and offset == 0
    ):
        return res[0]

    # The shape of res at this point is (n_terms, [,n_shots] [,batch_size])
    dot_products = []
    for c, r in zip(coeffs, res):
        if qml.math.get_interface(r) == "autograd":
            r = qml.math.array(r)
        if isinstance(r, (list, tuple)):
            r = qml.math.stack(r)
        dot_products.append(qml.math.dot(c, qml.math.squeeze(r)))
    if len(dot_products) == 0:
        return qml.math.ones(shape) * offset
    summed_dot_products = qml.math.sum(qml.math.stack(dot_products), axis=0)
    if qml.math.get_interface(offset) == "autograd" and qml.math.requires_grad(summed_dot_products):
        offset = qml.math.array(offset)
    return summed_dot_products + offset


def _mp_to_obs(mp: MeasurementProcess, tape: qml.tape.QuantumScript) -> qml.operation.Operator:
    """Extract the observable from a measurement process.

    If the measurement process has an observable, return it. Otherwise, return a dummy
    observable that is a tensor product of Z gates on every wire.

    """

    if mp.obs is not None:
        return mp.obs

    obs_wires = mp.wires if mp.wires else tape.wires
    return qml.prod(*(qml.Z(wire) for wire in obs_wires))

# Copyright 2024 Xanadu Quantum Technologies Inc.

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

from functools import partial
from typing import Callable, Dict, List, Sequence, Tuple

import pennylane as qml
from pennylane.measurements import ExpectationMP, MeasurementProcess, Shots, StateMP
from pennylane.ops import Hamiltonian, LinearCombination, Prod, SProd, Sum
from pennylane.transforms import transform
from pennylane.typing import Result, ResultBatch


@transform
def split_non_commuting(
    tape: qml.tape.QuantumScript,
    group: bool = True,
    grouping_strategy: str = None,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Splits a tape into tapes measuring groups of commuting observables."""

    # Special case for a single measurement of a Sum or Hamiltonian, in which case
    # the grouping information can be computed and cached in the observable.
    if (
        len(tape.measurements) == 1
        and isinstance(tape.measurements[0], ExpectationMP)
        and isinstance(tape.measurements[0].obs, (Hamiltonian, Sum))
        and (group or tape.measurements[0].obs.grouping_indices is not None)
    ):
        return _split_ham_with_grouping(tape)

    single_term_obs_mps, offsets = _split_all_multi_term_obs_mps(tape)

    if not group:
        measurements = list(single_term_obs_mps.keys())
        tapes = [tape.__class__(tape.operations, [m], shots=tape.shots) for m in measurements]
        return tapes, partial(
            _processing_fn_no_grouping,
            single_term_obs_mps=single_term_obs_mps,
            offsets=offsets,
            shots=tape.shots,
        )

    if grouping_strategy == "naive" or (
        grouping_strategy is None
        and any(
            isinstance(m, ExpectationMP) and isinstance(m.obs, (LinearCombination, Hamiltonian))
            for m in tape.measurements
        )
    ):
        # This is a loose check to see whether naive grouping or pauli grouping should be used,
        # which does not necessarily make perfect sense but consistent with the old decision
        # logic in `LegacyDevice.batch_transform`. The premise is that pauli grouping is
        # classically expensive but produces fewer tapes, whereas naive grouping is classically
        # faster to compute, but inefficient quantum-wise. If this transform is to be added to a
        # device's `preprocess`, it will be performed for every circuit execution, which can get
        # very expensive if there is a large number of observables. The reasoning here is, large
        # Hamiltonians typically come in the form of a `LinearCombination` or `Hamiltonian`, so
        # if we see one of those, use naive grouping to be safe. Otherwise, use pauli grouping.
        return _split_using_naive_grouping(tape, single_term_obs_mps, offsets)

    return _split_using_pauli_grouping(tape, single_term_obs_mps, offsets)


def _split_ham_with_grouping(tape: qml.tape.QuantumScript):
    """Splits a tape measuring a single Hamiltonian or Sum and group commuting observables."""

    obs = tape.measurements[0].obs
    if obs.grouping_indices is None:
        obs.compute_grouping()

    coeffs, obs_list = obs.terms()
    offset = 0
    single_term_obs_mps = {}
    mp_groups = []
    group_sizes = []

    for group_idx, obs_indices in enumerate(obs.grouping_indices):
        mp_group = []
        group_size = 0
        for obs_idx in obs_indices:
            if isinstance(obs_list[obs_idx], qml.Identity):
                offset += coeffs[obs_idx]
            else:
                new_mp = qml.expval(obs_list[obs_idx])
                mp_group.append(new_mp)
                group_size += 1
                if new_mp in single_term_obs_mps:
                    single_term_obs_mps[new_mp] = (
                        [0],
                        single_term_obs_mps[new_mp][1] + coeffs[obs_idx],
                        single_term_obs_mps[new_mp][2],
                        single_term_obs_mps[new_mp][3],
                    )
                else:
                    single_term_obs_mps[new_mp] = (
                        [0],
                        coeffs[obs_idx],
                        group_idx,
                        group_size - 1,
                    )
        if group_size > 0:
            mp_groups.append(mp_group)
            group_sizes.append(group_size)

    tapes = [tape.__class__(tape.operations, mps, shots=tape.shots) for mps in mp_groups]
    return tapes, partial(
        _processing_fn_with_grouping,
        single_term_obs_mps=single_term_obs_mps,
        offsets=[offset],
        group_sizes=group_sizes,
        shots=tape.shots,
    )


def _split_using_pauli_grouping(
    tape: qml.tape.QuantumScript,
    single_term_obs_mps: Dict[MeasurementProcess, Tuple[List[int], float]],
    offsets: List[float],
):
    """Split tapes using group_observables in the Pauli module.

    Args:
        tape (~qml.tape.QuantumScript): The tape to be split.
        single_term_obs_mps (Dict[MeasurementProcess, Tuple[List[int], float]]): A dictionary of
            measurements of each unique single-term observable, mapped to the indices of the
            original measurements it belongs to, and its coefficient.
        offsets (List[float]): Offsets associated with each original measurement in the tape.

    """

    # The legacy device does not support state measurements combined with any other
    # measurement, so each state measurement must be in its own tape.
    state_measurements = [m for m in single_term_obs_mps if isinstance(m, StateMP)]

    measurements = [m for m in single_term_obs_mps if not isinstance(m, StateMP)]
    obs_list = [_mp_to_obs(m, tape) for m in measurements]
    index_groups = []
    if len(obs_list) > 0:
        _, index_groups = qml.pauli.group_observables(obs_list, range(len(obs_list)))
    single_term_obs_mps_grouped = {}

    mp_groups = [[] for _ in index_groups]
    group_sizes = []
    for group_idx, obs_indices in enumerate(index_groups):
        group_size = 0
        for obs_idx in obs_indices:
            new_mp = measurements[obs_idx]
            mp_groups[group_idx].append(new_mp)
            group_size += 1
            single_term_obs_mps_grouped[new_mp] = (
                *single_term_obs_mps[new_mp],
                group_idx,
                group_size - 1,
            )
        group_sizes.append(group_size)

    for state_mp in state_measurements:
        mp_groups.append([state_mp])
        single_term_obs_mps_grouped[state_mp] = (
            *single_term_obs_mps[state_mp],
            len(mp_groups) - 1,
            0,
        )
        group_sizes.append(1)

    tapes = [tape.__class__(tape.operations, mps, shots=tape.shots) for mps in mp_groups]
    return tapes, partial(
        _processing_fn_with_grouping,
        single_term_obs_mps=single_term_obs_mps_grouped,
        offsets=offsets,
        group_sizes=group_sizes,
        shots=tape.shots,
    )


def _split_using_naive_grouping(
    tape: qml.tape.QuantumScript,
    single_term_obs_mps: Dict[MeasurementProcess, Tuple[List[int], float]],
    offsets: List[float],
):
    """Split tapes by grouping observables based on overlapping wires.

    Args:
        tape (~qml.tape.QuantumScript): The tape to be split.
        single_term_obs_mps (Dict[MeasurementProcess, Tuple[List[int], float]]): A dictionary of
            measurements of each unique single-term observable, mapped to the indices of the
            original measurements it belongs to, and its coefficient.
        offsets (List[float]): Offsets associated with each original measurement in the tape.

    """

    mp_groups = []
    wires_for_each_group = []
    group_sizes = []
    single_term_obs_mps_grouped = {}
    num_groups = 0

    for smp, (mp_indices, coeff) in single_term_obs_mps.items():

        if len(smp.wires) == 0:  # measurement acting on all wires
            mp_groups.append([smp])
            wires_for_each_group.append(tape.wires)
            group_sizes.append(1)
            single_term_obs_mps_grouped[smp] = (mp_indices, coeff, num_groups, 0)
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
                    coeff,
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
            single_term_obs_mps_grouped[smp] = (mp_indices, coeff, num_groups, 0)
            num_groups += 1

    tapes = [tape.__class__(tape.operations, mps, shots=tape.shots) for mps in mp_groups]
    return tapes, partial(
        _processing_fn_with_grouping,
        single_term_obs_mps=single_term_obs_mps_grouped,
        offsets=offsets,
        group_sizes=group_sizes,
        shots=tape.shots,
    )


def _split_all_multi_term_obs_mps(tape: qml.tape.QuantumScript):
    """Splits all multi-term observables in a tape to measurements of single-term observables.

    Args:
        tape (~qml.tape.QuantumScript): The tape with measurements to split.

    Returns:
        single_term_obs_mps (Dict[MeasurementProcess, Tuple[List[int], float]]): A dictionary
            for measurements of each unique single-term observable, mapped to the indices of the
            original measurements it belongs to, and its coefficient.
        offsets (List[float]): Offsets associated with each original measurement in the tape.

    """

    # The dictionary for measurements of each unique single-term observable, mapped the indices
    # of the original measurements it belongs to, and its coefficient.
    single_term_obs_mps = {}

    # Offsets associated with each original measurement in the tape (from Identity)
    offsets = []

    for mp_idx, mp in enumerate(tape.measurements):
        obs = mp.obs
        offset = 0
        if isinstance(mp, ExpectationMP) and isinstance(obs, (Hamiltonian, Sum, Prod, SProd)):
            if isinstance(obs, SProd):
                # This is necessary because SProd currently does not flatten into
                # multiple terms if the base is a sum, which is needed here.
                obs = obs.simplify()
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
                # Otherwise, add this new measurement to the list of single-term measurements.
                else:
                    single_term_obs_mps[sm] = ([mp_idx], c)
        else:
            # For all other measurement types, simply add them to the list of measurements.
            if mp not in single_term_obs_mps:
                single_term_obs_mps[mp] = ([mp_idx], 1)
            else:
                single_term_obs_mps[mp][0].append(mp_idx)

        offsets.append(offset)

    return single_term_obs_mps, offsets


def _processing_fn_no_grouping(
    res: ResultBatch,
    single_term_obs_mps: Dict[MeasurementProcess, Tuple[List[int], float]],
    offsets: List[float],
    shots: Shots,
):
    """Postprocessing function for the split_non_commuting transform without grouping.

    Args:
        res (ResultBatch): The results from executing the tapes. Assumed to have a shape
            of (n_groups [,n_shots] [,n_mps] [,n_parameters])
        single_term_obs_mps (Dict[MeasurementProcess, Tuple[List[int], float]]): A dictionary of
            measurements of each unique single-term observable, mapped to the indices of the
            original measurements it belongs to, and its coefficient.
        offsets (List[float]): Offsets associated with each original measurement in the tape.
        shots (Shots): The shots settings of the original tape.

    """

    res_batch_for_each_mp = [[] for _ in offsets]
    coeffs_for_each_mp = [[] for _ in offsets]

    for smp_idx, (_, (mp_indices, coeff)) in enumerate(single_term_obs_mps.items()):

        for mp_idx in mp_indices:
            res_batch_for_each_mp[mp_idx].append(res[smp_idx])
            coeffs_for_each_mp[mp_idx].append(coeff)

    # Sum up the results for each original measurement
    res_for_each_mp = [
        _sum_terms(_sub_res, coeffs, offset)
        for _sub_res, coeffs, offset in zip(res_batch_for_each_mp, coeffs_for_each_mp, offsets)
    ]

    # res_for_each_mp should have shape (n_mps, [,n_shots] [,n_parameters])
    if len(res_for_each_mp) == 1:
        return res_for_each_mp[0]

    if shots.has_partitioned_shots:
        # If the shot vector dimension exists, it should be moved to the first axis
        # Basically, the shape becomes (n_shots, n_mps, [,n_parameters])
        res_for_each_mp = [
            tuple(res_for_each_mp[j][i] for j in range(len(res_for_each_mp)))
            for i in range(len(res_for_each_mp[0]))
        ]

    return tuple(res_for_each_mp)


def _processing_fn_with_grouping(
    res: ResultBatch,
    single_term_obs_mps: Dict[MeasurementProcess, Tuple[List[int], float, int, int]],
    offsets: List[float],
    group_sizes: List[int],
    shots: Shots,
):
    """Postprocessing function for the split_non_commuting transform with grouping.

    Args:
        res (ResultBatch): The results from executing the tapes. Assumed to have a shape
            of (n_groups [,n_shots] [,n_mps] [,n_parameters])
        single_term_obs_mps (Dict[MeasurementProcess, Tuple[List[int], float, int, int]]):
            A dictionary of measurements of each unique single-term observable, mapped to the
            indices of the original measurements it belongs to, its coefficient, its group
            index, and the index of the measurement within the group.
        offsets (List[float]): Offsets associated with each original measurement in the tape.
        group_sizes (List[int]): The number of tapes in each group.
        shots (Shots): The shots setting of the original tape.

    Returns:
        The results combined into a single result for each original measurement.

    """

    res_batch_for_each_mp = [[] for _ in offsets]
    coeffs_for_each_mp = [[] for _ in offsets]

    for _, (mp_indices, coeff, group_idx, mp_idx_in_group) in single_term_obs_mps.items():

        res_group = res[group_idx]  # ([n_shots] [,n_mps] [,n_parameters])
        group_size = group_sizes[group_idx]

        if group_size > 1 and shots.has_partitioned_shots:
            # Each result should have shape ([n_shots] [,n_parameters])
            sub_res = [_res[mp_idx_in_group] for _res in res_group]
        else:
            # If there is only one term in the group, the n_mps dimension would have
            # been squeezed out, use the entire result directly.
            sub_res = res_group if group_size == 1 else res_group[mp_idx_in_group]

        # Add this result to the result batch for the corresponding original measurement
        for mp_idx in mp_indices:
            res_batch_for_each_mp[mp_idx].append(sub_res)
            coeffs_for_each_mp[mp_idx].append(coeff)

    # Sum up the results for each original measurement
    res_for_each_mp = [
        _sum_terms(_sub_res, coeffs, offset)
        for _sub_res, coeffs, offset in zip(res_batch_for_each_mp, coeffs_for_each_mp, offsets)
    ]

    # res_for_each_mp should have shape (n_mps, [,n_shots] [,n_parameters])
    if len(res_for_each_mp) == 1:
        return res_for_each_mp[0]

    if shots.has_partitioned_shots:
        # If the shot vector dimension exists, it should be moved to the first axis
        # Basically, the shape becomes (n_shots, n_mps, [,n_parameters])
        res_for_each_mp = [
            tuple(res_for_each_mp[j][i] for j in range(len(res_for_each_mp)))
            for i in range(len(res_for_each_mp[0]))
        ]

    return tuple(res_for_each_mp)


def _sum_terms(res: ResultBatch, coeffs: List[float], offset: float) -> Result:
    """Sum results from measurements of multiple terms in a multi-term observable."""

    # Trivially return the original result
    if coeffs == [1] and offset == 0:
        return res[0]

    # The shape of res at this point is (n_terms, [,n_shots] [,n_parameters])
    dot_products = []
    for c, r in zip(coeffs, res):
        dot_products.append(qml.math.dot(qml.math.squeeze(r), c))
    if len(dot_products) == 0:
        return offset
    summed_dot_products = qml.math.sum(qml.math.stack(dot_products), axis=0)
    return qml.math.convert_like(summed_dot_products + offset, res[0])


def _mp_to_obs(mp: MeasurementProcess, tape: qml.tape.QuantumScript) -> qml.operation.Operator:
    """Extract the observable from a measurement process.

    If the measurement process has an observable, return it. Otherwise, return a dummy
    observable that is a tensor product of Z gates on every wire.

    """

    if mp.obs is not None:
        return mp.obs

    obs_wires = mp.wires if mp.wires else tape.wires
    return qml.prod(*(qml.Z(wire) for wire in obs_wires))

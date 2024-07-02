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
Contains the tape transform that splits multi-term measurements on a tape into single-term measurements,
all included on the same tape. This transform expands sums but does not divide non-commuting measurements
between different tapes.
"""

from functools import partial
from typing import Dict, List, Tuple

from pennylane.measurements import MeasurementProcess, Shots
from pennylane.transforms import transform
from pennylane.transforms.split_non_commuting import (
    _infer_result_shape,
    _split_all_multi_term_obs_mps,
    _sum_terms,
)
from pennylane.typing import ResultBatch


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def split_to_single_terms(tape):
    """Placeholder docstring for pylint"""

    if len(tape.measurements) == 0:
        return [tape], null_postprocessing

    single_term_obs_mps, offsets = _split_all_multi_term_obs_mps(tape)

    new_tape = tape.__class__(
        tape.operations, measurements=list(single_term_obs_mps), shots=tape.shots
    )

    print(new_tape.measurements)

    def post_processing_fn(res):

        process_shot_copy = partial(
            process_tape_results_from_shot_copy,
            single_term_obs_mps=single_term_obs_mps,
            offsets=offsets,
            shots=tape.shots,
            batch_size=tape.batch_size,
        )

        if tape.shots.has_partitioned_shots:
            return tuple(process_shot_copy(c) for c in res)
        return process_shot_copy(res)

    return (new_tape,), post_processing_fn


def process_tape_results_from_shot_copy(
    res: ResultBatch,
    single_term_obs_mps: Dict[MeasurementProcess, Tuple[List[int], List[float]]],
    offsets: List[float],
    shots: Shots,
    batch_size: int,
):
    """Placeholder docstring for pylint"""

    # remove the extra dimension added by nesting the single tape as (tape,)
    res = res[0]

    # res dimensions are: (len(new_measurements), tape.batch_size, shots), or with
    # partitioned shots: (tape.shots.num_copies, len(new_measurements), tape.batch_size, shots)

    res_batch_for_each_mp = [[] for _ in offsets]
    coeffs_for_each_mp = [[] for _ in offsets]

    for smp_idx, (_, (mp_indices, coeffs)) in enumerate(single_term_obs_mps.items()):
        for mp_idx, coeff in zip(mp_indices, coeffs):
            res_batch_for_each_mp[mp_idx].append(res[smp_idx])
            coeffs_for_each_mp[mp_idx].append(coeff)

    result_shape = _infer_result_shape(shots, batch_size)

    # Sum up the results for each original measurement
    res_for_each_mp = [
        _sum_terms(_sub_res, coeffs, offset, result_shape)
        for _sub_res, coeffs, offset in zip(res_batch_for_each_mp, coeffs_for_each_mp, offsets)
    ]

    # res_for_each_mp should have shape (n_mps, [,batch_size] [,n_shots] )
    if len(res_for_each_mp) == 1:
        return res_for_each_mp[0]

    return tuple(res_for_each_mp)

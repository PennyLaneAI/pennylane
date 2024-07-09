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

from pennylane.transforms import transform
from pennylane.transforms.split_non_commuting import (
    _processing_fn_no_grouping,
    _split_all_multi_term_obs_mps,
)


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
    new_measurements = list(single_term_obs_mps)

    if new_measurements == tape.measurements:
        # measurements are unmodified by the transform
        return [tape], null_postprocessing

    new_tape = tape.__class__(tape.operations, measurements=new_measurements, shots=tape.shots)

    def post_processing_fn(res):
        """We results are the same as those produced by split_non_commuting with
        grouping_strategy=None, except that we return them all on a single tape,
        reorganizing the shape of the results. In post-processing, we remove the
        extra dimension added by (tape,), and swap the order of MPs vs shot copies
        to get results in a format identical to the split_non_commuting transform,
        and then use the same post-processing function on the transformed results."""

        process = partial(
            _processing_fn_no_grouping,
            single_term_obs_mps=single_term_obs_mps,
            offsets=offsets,
            shots=tape.shots,
            batch_size=tape.batch_size,
        )

        # something about offsets and only a single measurement ends up different
        # what is going on here?
        if len(new_tape.measurements) == 1:
            return process(res)

        # we go from ((mp1_res, mp2_res, mp3_res),) as result output
        # to (mp1_res, mp2_res, mp3_res) as expected by post-processing
        res = res[0]
        if tape.shots.has_partitioned_shots:
            print(res)
            # swap dimension order of mps vs shot copies for _processing_fn_no_grouping
            res = [
                tuple(res[j][i] for j in range(tape.shots.num_copies))
                for i in range(len(new_tape.measurements))
            ]

        return process(res)

    return [new_tape], post_processing_fn

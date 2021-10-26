# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO"""
from typing import Any, Dict, Sequence, Optional
from pennylane.transforms import batch_transform, single_tape_transform
from pennylane.tape import QuantumTape
from pennylane import apply


@batch_transform
def mitigate_with_zne(tape: QuantumTape, scale_factors: Sequence[float], folding: callable, extrapolate: callable, folding_kwargs: Optional[Dict[str, Any]]=None, extrapolate_kwargs: Optional[Dict[str, Any]]=None) -> float:
    folding_kwargs = folding_kwargs or {}
    extrapolate_kwargs = extrapolate_kwargs or {}

    tape_without_measurements = _remove_measurements(tape)

    tapes = [folding(tape_without_measurements, s, **folding_kwargs) for s in scale_factors]
    tapes = [_add_measurements(t, tape.measurements) for t in tapes]

    def processing_fn(results):
        return extrapolate(scale_factors, results, **extrapolate_kwargs)

    return tapes, processing_fn


def _remove_measurements(tape):
    """Removes the measurements of a given tape

    Args:
        tape (QuantumTape): input quantum tape which may include measurements

    Returns:
        QuantumTape: the input tape with the measurements removed
    """
    with QuantumTape() as new_tape:
        for op in tape.operations:
            apply(op)
    return new_tape


@single_tape_transform
def _add_measurements(tape, measurements):
    """TODO"""
    for op in tape.operations:
        apply(op)
    for m in measurements:
        apply(m)

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
from functools import wraps
from typing import Any, Dict, Sequence, Optional, Tuple
from pennylane.transforms import batch_transform, single_tape_transform
from pennylane.tape import QuantumTape
from pennylane import apply
from pennylane.math import mean
from pennylane import QubitStateVector, BasisState
from pennylane.operation import Operation
from pennylane.measure import MeasurementProcess


@batch_transform
def mitigate_with_zne(tape: QuantumTape, scale_factors: Sequence[float], folding: callable, extrapolate: callable, folding_kwargs: Optional[Dict[str, Any]]=None, extrapolate_kwargs: Optional[Dict[str, Any]]=None, reps_per_factor=1) -> float:
    folding_kwargs = folding_kwargs or {}
    extrapolate_kwargs = extrapolate_kwargs or {}

    tape_no_meas = _remove_measurements(tape)

    tapes = [[folding(tape_no_meas, s, **folding_kwargs) for _ in range(reps_per_factor)] for s in scale_factors]
    tapes = [tape_ for tapes_ in tapes for tape_ in tapes_]
    tapes = [_add_measurements(t, tape.measurements) for t in tapes]

    def processing_fn(results):
        results = [results[i:i + reps_per_factor] for i in range(0, len(results), reps_per_factor)]
        results = mean(results, axis=1)

        return extrapolate(scale_factors, results, **extrapolate_kwargs)

    return tapes, processing_fn

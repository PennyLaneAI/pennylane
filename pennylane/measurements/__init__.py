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
This module contains measurements supported by PennyLane.
"""
from .classical_shadow import ClassicalShadowMP, ShadowExpvalMP, classical_shadow, shadow_expval
from .counts import CountsMP, counts
from .expval import ExpectationMP, expval
from .measurements import (
    AllCounts,
    Counts,
    Expectation,
    MeasurementProcess,
    MeasurementShapeError,
    MeasurementTransform,
    MidMeasure,
    MutualInfo,
    ObservableReturnTypes,
    Probability,
    Sample,
    SampleMeasurement,
    Shadow,
    ShadowExpval,
    State,
    StateMeasurement,
    Variance,
    VnEntropy,
)
from .mid_measure import MeasurementValue, MeasurementValueError, MidMeasureMP, measure
from .mutual_info import MutualInfoMP, mutual_info
from .probs import ProbabilityMP, probs
from .sample import SampleMP, sample
from .state import StateMP, density_matrix, state
from .var import VarianceMP, var
from .vn_entropy import VnEntropyMP, vn_entropy

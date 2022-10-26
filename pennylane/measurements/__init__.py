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
from .classical_shadow import (
    Shadow,
    ShadowExpval,
    ShadowMeasurementProcess,
    classical_shadow,
    shadow_expval,
)
from .counts import AllCounts, Counts, counts
from .expval import Expectation, expval
from .measurements import MeasurementProcess, MeasurementValue
from .mid_measure import MidMeasure, measure
from .mutual_info import MutualInfo, mutual_info
from .probs import Probability, probs
from .sample import Sample, sample
from .state import State, density_matrix, state
from .var import Variance, var
from .vn_entropy import VnEntropy, vn_entropy

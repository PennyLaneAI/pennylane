# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PennyLane-xDSL transformations API."""

from .cancel_inverses import IterativeCancelInversesPass, iterative_cancel_inverses_pass
from .measurements_from_samples import MeasurementsFromSamplesPass, measurements_from_samples_pass
from .merge_rotations import MergeRotationsPass, merge_rotations_pass
from .combine_global_phases import combine_global_phases_pass, CombineGlobalPhasesPass
from .convert_to_mbqc_formalism import convert_to_mbqc_formalism, ConvertToMBQCFormalismPass

__all__ = [
    "combine_global_phases_pass",
    "CombineGlobalPhasesPass",
    "convert_to_mbqc_formalism",
    "ConvertToMBQCFormalismPass",
    "iterative_cancel_inverses_pass",
    "IterativeCancelInversesPass",
    "measurements_from_samples_pass",
    "MeasurementsFromSamplesPass",
    "merge_rotations_pass",
    "MergeRotationsPass",
]

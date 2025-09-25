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

from .mbqc import (
    convert_to_mbqc_formalism_pass,
    ConvertToMBQCFormalismPass,
    decompose_graph_state_pass,
    DecomposeGraphStatePass,
    null_decompose_graph_state_pass,
    NullDecomposeGraphStatePass,
)

from .quantum import (
    combine_global_phases_pass,
    CombineGlobalPhasesPass,
    diagonalize_final_measurements_pass,
    DiagonalizeFinalMeasurementsPass,
    iterative_cancel_inverses_pass,
    IterativeCancelInversesPass,
    measurements_from_samples_pass,
    MeasurementsFromSamplesPass,
    merge_rotations_pass,
    MergeRotationsPass,
)


__all__ = [
    # Quantum
    "combine_global_phases_pass",
    "CombineGlobalPhasesPass",
    "diagonalize_final_measurements_pass",
    "DiagonalizeFinalMeasurementsPass",
    "iterative_cancel_inverses_pass",
    "IterativeCancelInversesPass",
    "measurements_from_samples_pass",
    "MeasurementsFromSamplesPass",
    "merge_rotations_pass",
    "MergeRotationsPass",
    # MBQC
    "convert_to_mbqc_formalism_pass",
    "ConvertToMBQCFormalismPass",
    "decompose_graph_state_pass",
    "DecomposeGraphStatePass",
    "null_decompose_graph_state_pass",
    "NullDecomposeGraphStatePass",
]

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

from xdsl.transforms.transform_interpreter import TransformInterpreterPass
from .apply_transform_sequence import ApplyTransformSequence, register_pass
from .cancel_inverses import iterative_cancel_inverses_pass, IterativeCancelInversesPass
from .combine_global_phases import combine_global_phases_pass, CombineGlobalPhasesPass
from .merge_rotations import merge_rotations_pass, MergeRotationsPass
from .utils import xdsl_transform


__all__ = [
    "ApplyTransformSequence",
    "combine_global_phases_pass",
    "CombineGlobalPhasesPass",
    "iterative_cancel_inverses_pass",
    "IterativeCancelInversesPass",
    "merge_rotations_pass",
    "MergeRotationsPass",
    "TransformInterpreterPass",
    "register_pass",
    "xdsl_transform",
]

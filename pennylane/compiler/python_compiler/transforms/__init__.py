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
"""PennyLane-xDSL transformations."""

from .cancel_inverses import IterativeCancelInversesPass, iterative_cancel_inverses_pass
from .merge_rotations import MergeRotationsPass, merge_rotations_pass

__all__ = [
    "iterative_cancel_inverses_pass",
    "IterativeCancelInversesPass",
    "merge_rotations_pass",
    "MergeRotationsPass",
]

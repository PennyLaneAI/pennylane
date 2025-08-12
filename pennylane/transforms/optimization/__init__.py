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
This subpackage contains quantum function transforms for optimizing quantum circuits.
"""

from .remove_barrier import remove_barrier
from .cancel_inverses import cancel_inverses
from .commute_controlled import commute_controlled
from .merge_rotations import merge_rotations
from .merge_amplitude_embedding import merge_amplitude_embedding
from .single_qubit_fusion import single_qubit_fusion
from .undo_swaps import undo_swaps
from .pattern_matching import pattern_matching, pattern_matching_optimization

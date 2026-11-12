# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility exports for decomposition rule factories."""

from pennylane.decomposition_rules.transforms.decompositions.rz_phase_gradient import (
    make_rz_to_phase_gradient_decomp,
    phase_gradient_precision_stats,
    validate_phase_gradient_wires,
)

__all__ = [
    "make_rz_to_phase_gradient_decomp",
    "phase_gradient_precision_stats",
    "validate_phase_gradient_wires",
]

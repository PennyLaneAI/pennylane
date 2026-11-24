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

r"""Decomposition rules for phase-gradient rotations and state preparation.

Rules, factories and shared precision helpers are loaded during normal PennyLane
initialization and can be imported from this package.

.. autosummary::

    common
    rz
    crz
    select_pauli_rot
    state_preparation
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common import phase_gradient_precision_stats, validate_phase_gradient_wires
    from .crz import _CRZPhaseGradientCompute, make_crz_to_phase_gradient_decomp
    from .rz import make_rz_to_phase_gradient_decomp
    from .select_pauli_rot import (
        _SelectPauliRotPhaseGradientCompute,
        make_selectpaulirot_to_phase_gradient_decomp,
    )
    from .state_preparation import (
        make_phase_gradient_distillation_decomp,
        phase_gradient_state_prep_to_clifford_t,
        phase_gradient_state_prep_to_ppr,
    )

__all__ = [
    "_CRZPhaseGradientCompute",
    "_SelectPauliRotPhaseGradientCompute",
    "make_crz_to_phase_gradient_decomp",
    "make_phase_gradient_distillation_decomp",
    "make_rz_to_phase_gradient_decomp",
    "make_selectpaulirot_to_phase_gradient_decomp",
    "phase_gradient_precision_stats",
    "phase_gradient_state_prep_to_clifford_t",
    "phase_gradient_state_prep_to_ppr",
    "validate_phase_gradient_wires",
]

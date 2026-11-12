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

"""Built-in decomposition rules for templates.state_preparations.phase_gradient."""

from collections import Counter

import pennylane as qp
from pennylane.decomposition import add_decomps, register_condition, register_resources
from pennylane.templates.state_preparations.phase_gradient import PhaseGradientStatePrep
from pennylane.templates.state_preparations.phase_gradient_decomp_data import (
    CLIFFORD_T_GLOBAL_PHASES,
    CLIFFORD_T_SEQUENCES,
    GLOBAL_PHASES,
    INITIAL_PPRS,
    PI_OVER_8_PPRS,
)
from pennylane.typing import AbstractWires, Wire
from pennylane.wires import WiresLike


def _ppr_counts(num_wires):
    counts = Counter()
    for init, pprs in zip(INITIAL_PPRS[:num_wires], PI_OVER_8_PPRS[:num_wires], strict=True):
        if init is not None:
            counts[init] += 1
        counts.update((8 if p.isupper() else -8, p.upper()) for p in pprs)
    return counts


def _ppr(denominator, pauli_word, wires):
    op_class = {2: qp.PPR_2, 4: qp.PPR_4, 8: qp.PPR_8}[abs(denominator)]
    return op_class(1 if denominator > 0 else -1, pauli_word, wires=wires)


def _phase_gradient_state_prep_ppr_resources(wires: AbstractWires):
    resources = {
        _ppr(denominator, pauli, wires=Wire[1]): count
        for (denominator, pauli), count in _ppr_counts(len(wires)).items()
    }
    resources[qp.GlobalPhase] = 1
    return resources


@register_condition(lambda wires: len(wires) <= len(PI_OVER_8_PPRS))
@register_resources(_phase_gradient_state_prep_ppr_resources)
def _phase_gradient_state_prep_ppr_decomposition(wires: WiresLike):
    # Wire j is prepared in the state (|0> + exp(-i pi / 2**j)|1>) / sqrt(2) by the PPR
    # ``INITIAL_PPRS[j]`` (angle denominator, Pauli word; no gate if ``None``), followed by the π/8
    # PPRs in ``PI_OVER_8_PPRS[j]``, where upper (lower) case letters denote PPRs with angle π/8
    # (-π/8). Together with the global phase given by summing ``GLOBAL_PHASES`` over the wires,
    # the state prepared on 30 wires deviates from the phase gradient state by
    # less than 1e-12 in the 2-norm.

    num_wires = len(wires)
    for wire, init, pprs in zip(
        wires, INITIAL_PPRS[:num_wires], PI_OVER_8_PPRS[:num_wires], strict=True
    ):
        if init is not None:
            _ppr(*init, wires=wire)
        for p in pprs:
            qp.PPR_8(1 if p.isupper() else -1, p.upper(), wires=wire)
    qp.GlobalPhase(sum(GLOBAL_PHASES[: len(wires)]))


def _phase_gradient_state_prep_clifford_t_resources(wires: AbstractWires):
    counts = Counter("".join(CLIFFORD_T_SEQUENCES[: len(wires)]))
    resources = {getattr(qp, gate): count for gate, count in counts.items()}
    resources[qp.GlobalPhase] = 1
    return resources


@register_condition(lambda wires: len(wires) <= len(CLIFFORD_T_SEQUENCES))
@register_resources(_phase_gradient_state_prep_clifford_t_resources)
def _phase_gradient_state_prep_clifford_t_decomposition(wires: WiresLike):
    # Wire j is prepared by the gates in ``CLIFFORD_T_SEQUENCES[j]``, which are obtained from the
    # PPR decomposition and contain one T gate per π/8 PPR. Together with the global phase
    # given by summing ``CLIFFORD_T_GLOBAL_PHASES`` over the wires, the state prepared on 30 wires
    # deviates from the phase gradient state by less than 1e-12 in the 2-norm.

    num_wires = len(wires)
    for wire, sequence in zip(wires, CLIFFORD_T_SEQUENCES[:num_wires], strict=True):
        for gate in sequence:
            getattr(qp, gate)(wire)
    qp.GlobalPhase(sum(CLIFFORD_T_GLOBAL_PHASES[:num_wires]))


add_decomps(
    PhaseGradientStatePrep,
    _phase_gradient_state_prep_ppr_decomposition,
    _phase_gradient_state_prep_clifford_t_decomposition,
)

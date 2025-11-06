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

"""A fixed set of decomposition rules for testing purposes."""

# pylint: disable=too-few-public-methods,protected-access

from collections import defaultdict

import pennylane as qml
from pennylane.decomposition import Resources
from pennylane.decomposition.decomposition_rule import auto_wrap
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    pow_involutory,
    pow_rotation,
    self_adjoint,
)
from pennylane.ops.identity import _controlled_g_phase_decomp
from pennylane.ops.qubit.non_parametric_ops import _controlled_hadamard, _controlled_x_decomp

decompositions = defaultdict(list)


def to_resources(gate_count: dict, weighted_cost: float = None) -> Resources:
    """Wrap a dictionary of gate counts in a Resources object."""
    return Resources(
        {auto_wrap(op): count for op, count in gate_count.items() if count >= 0},
        (
            sum(count for gate, count in gate_count.items())
            if weighted_cost is None
            else weighted_cost
        ),
    )


@qml.register_resources({qml.Hadamard: 2, qml.CNOT: 1})
def _cz_to_cnot(*_, **__):
    raise NotImplementedError


decompositions["CZ"] = [_cz_to_cnot]


@qml.register_resources({qml.Hadamard: 2, qml.CZ: 1})
def _cnot_to_cz(*_, **__):
    raise NotImplementedError


decompositions["CNOT"] = [_cnot_to_cz]


def _multi_rz_decomposition_resources(num_wires):
    return {qml.RZ: 1, qml.CNOT: 2 * (num_wires - 1)}


@qml.register_resources(_multi_rz_decomposition_resources)
def _multi_rz_decomposition(*_, **__):
    raise NotImplementedError


decompositions["MultiRZ"] = [_multi_rz_decomposition]


@qml.register_resources({qml.RZ: 2, qml.RX: 1, qml.GlobalPhase: 1})
def _hadamard_to_rz_rx(*_, **__):
    raise NotImplementedError


@qml.register_resources({qml.RZ: 1, qml.RY: 1, qml.GlobalPhase: 1})
def _hadamard_to_rz_ry(*_, **__):
    raise NotImplementedError


decompositions["Hadamard"] = [_hadamard_to_rz_rx, _hadamard_to_rz_ry]


@qml.register_resources({qml.RX: 1, qml.RZ: 2})
def _ry_to_rx_rz(*_, **__):
    raise NotImplementedError


decompositions["RY"] = [_ry_to_rx_rz]


@qml.register_resources({qml.RX: 2, qml.CZ: 2})
def _crx_to_rx_cz(*_, **__):
    raise NotImplementedError


@qml.register_resources({qml.RX: 2, qml.CNOT: 2, qml.RY: 4, qml.GlobalPhase: 4, qml.RZ: 4})
def _crx_to_rx_ry_cnot_ry_cnot_ry_cnot_rz(*_, **__):
    raise NotImplementedError


decompositions["CRX"] = [_crx_to_rx_cz, _crx_to_rx_ry_cnot_ry_cnot_ry_cnot_rz]


@qml.register_resources({qml.RZ: 3, qml.CNOT: 2, qml.GlobalPhase: 1})
def _cphase_to_rz_cnot(*_, **__):
    raise NotImplementedError


decompositions["ControlledPhaseShift"] = [_cphase_to_rz_cnot]


@qml.register_resources({qml.RZ: 1, qml.GlobalPhase: 1})
def _phase_shift_to_rz_gp(*_, **__):
    raise NotImplementedError


decompositions["PhaseShift"] = [_phase_shift_to_rz_gp]


@qml.register_resources({qml.RX: 1, qml.GlobalPhase: 1})
def _x_to_rx(*_, **__):
    raise NotImplementedError


decompositions["PauliX"] = [_x_to_rx]


@qml.register_resources({qml.PhaseShift: 1})
def _u1_ps(phi, wires, **__):
    qml.PhaseShift(phi, wires=wires)


decompositions["U1"] = [_u1_ps]


@qml.register_resources({qml.PhaseShift: 1})
def _t_ps(wires, **__):
    raise NotImplementedError


decompositions["T"] = [_t_ps]


@qml.register_resources({qml.RZ: 3, qml.RY: 2, qml.CNOT: 2})
def _crot(*_, **__):
    raise NotImplementedError


decompositions["CRot"] = [_crot]

################################################
# Custom Decompositions For Symbolic Operators #
################################################

decompositions["C(PauliX)"] = [_controlled_x_decomp]
decompositions["C(GlobalPhase)"] = [_controlled_g_phase_decomp]
decompositions["C(Hadamard)"] = [_controlled_hadamard]
decompositions["Adjoint(Hadamard)"] = [self_adjoint]
decompositions["Pow(Hadamard)"] = [pow_involutory]
decompositions["Adjoint(RX)"] = [adjoint_rotation]
decompositions["Pow(RX)"] = [pow_rotation]
decompositions["Adjoint(CNOT)"] = [self_adjoint]
decompositions["Adjoint(PhaseShift)"] = [adjoint_rotation]
decompositions["Adjoint(ControlledPhaseShift)"] = [adjoint_rotation]

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

from collections import defaultdict
from contextvars import ContextVar

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.decomposition import Resources
from pennylane.decomposition.decomposition_rule import DecompCollection
from pennylane.decomposition.symbolic_decomposition import (
    _adjoint_negate_angle,
    _pow_scale_angle,
    self_adjoint,
)
from pennylane.ops.identity import _controlled_g_phase_standard
from pennylane.ops.op_math.adjoint2 import _adjoint_negate_angle as adjoint_rotation2
from pennylane.ops.op_math.controlled2 import flip_zero_control
from pennylane.ops.op_math.pow2 import pow_involutory as pow_involutory2
from pennylane.ops.qubit.non_parametric_ops import (
    _controlled_hadamard_standard,
    _controlled_z_standard,
    _ctrl_x_to_mcx,
)
from pennylane.ops.qubit.parametric_ops_single_qubit import (
    _controlled_phase_shift_standard,
    _controlled_rot_standard,
    _controlled_rx_standard,
)

_decompositions = defaultdict(DecompCollection)
decompositions = ContextVar("_test_decompositions", default=_decompositions)


def to_resources(gate_count: dict, weighted_cost: float | None = None) -> Resources:
    """Wrap a dictionary of gate counts in a Resources object."""
    if weighted_cost is None:
        weighted_cost = sum(count for count in gate_count.values())
    gate_count = {abstractify(op): count for op, count in gate_count.items() if count >= 0}
    return Resources(gate_count, weighted_cost=weighted_cost)


@qp.register_name("cz_to_cnot")
@qp.register_resources({qp.Hadamard: 2, qp.CNOT: 1})
def _cz_to_cnot(*_, **__):
    raise NotImplementedError


decompositions.get()["CZ"].append(_cz_to_cnot)


@qp.register_name("cnot_to_cz")
@qp.register_resources({qp.Hadamard: 2, qp.CZ: 1})
def _cnot_to_cz(*_, **__):
    raise NotImplementedError


decompositions.get()["CNOT"].append(_cnot_to_cz)


def _multi_rz_standard_resources(num_wires):
    return {qp.RZ: 1, qp.CNOT: 2 * (num_wires - 1)}


@qp.register_name("multi_rz_standard")
@qp.register_resources(_multi_rz_standard_resources)
def _multi_rz_standard(*_, **__):
    raise NotImplementedError


decompositions.get()["MultiRZ"].append(_multi_rz_standard)


@qp.register_name("hadamard_to_rz_rx")
@qp.register_resources({qp.RZ: 2, qp.RX: 1, qp.GlobalPhase: 1})
def _hadamard_to_rz_rx(*_, **__):
    raise NotImplementedError


@qp.register_name("hadamard_to_rz_ry")
@qp.register_resources({qp.RZ: 1, qp.RY: 1, qp.GlobalPhase: 1})
def _hadamard_to_rz_ry(*_, **__):
    raise NotImplementedError


decompositions.get()["Hadamard"].append(_hadamard_to_rz_rx)
decompositions.get()["Hadamard"].append(_hadamard_to_rz_ry)


@qp.register_name("ry_to_rx_rz")
@qp.register_resources({qp.RX: 1, qp.RZ: 2})
def _ry_to_rx_rz(*_, **__):
    raise NotImplementedError


decompositions.get()["RY"].append(_ry_to_rx_rz)


@qp.register_name("crx_to_rx_cz")
@qp.register_resources({qp.RX: 2, qp.CZ: 2})
def _crx_to_rx_cz(*_, **__):
    raise NotImplementedError


@qp.register_name("crx_to_rx_ry_cnot_ry_cnot_ry_cnot_rz")
@qp.register_resources(
    {qp.RX: 2, qp.CNOT: 2, qp.RY: 4, qp.GlobalPhase: 4, qp.RZ: 4},
)
def _crx_to_rx_ry_cnot_ry_cnot_ry_cnot_rz(*_, **__):
    raise NotImplementedError


decompositions.get()["CRX"].append(_crx_to_rx_cz)
decompositions.get()["CRX"].append(_crx_to_rx_ry_cnot_ry_cnot_ry_cnot_rz)


@qp.register_name("cphase_to_rz_cnot")
@qp.register_resources({qp.RZ: 3, qp.CNOT: 2, qp.GlobalPhase: 1})
def _cphase_to_rz_cnot(*_, **__):
    raise NotImplementedError


decompositions.get()["ControlledPhaseShift"].append(_cphase_to_rz_cnot)


@qp.register_name("phase_shift_to_rz_gp")
@qp.register_resources({qp.RZ: 1, qp.GlobalPhase: 1})
def _phase_shift_to_rz_gp(*_, **__):
    raise NotImplementedError


decompositions.get()["PhaseShift"].append(_phase_shift_to_rz_gp)


@qp.register_name("x_to_rx")
@qp.register_resources({qp.RX: 1, qp.GlobalPhase: 1})
def _x_to_rx(*_, **__):
    raise NotImplementedError


decompositions.get()["PauliX"].append(_x_to_rx)


@qp.register_name("u1_ps")
@qp.register_resources({qp.PhaseShift: 1})
def _u1_ps(phi, wires, **__):
    qp.PhaseShift(phi, wires=wires)


decompositions.get()["U1"].append(_u1_ps)


@qp.register_name("t_ps")
@qp.register_resources({qp.PhaseShift: 1})
def _t_ps(wires, **__):
    raise NotImplementedError


decompositions.get()["T"].append(_t_ps)


@qp.register_name("crot_to_rot_cnot")
@qp.register_resources({qp.RZ: 3, qp.RY: 2, qp.CNOT: 2})
def _crot_to_rot_cnot(*_, **__):
    raise NotImplementedError


decompositions.get()["CRot"].append(_crot_to_rot_cnot)

################################################
# Custom Decompositions For Symbolic Operators #
################################################

decompositions.get()["C(GlobalPhase)"].append(_controlled_g_phase_standard)
decompositions.get()["C(Hadamard)"].append(_controlled_hadamard_standard)
decompositions.get()["C(PauliX)"].append(_ctrl_x_to_mcx)
decompositions.get()["C(PauliZ)"].append(flip_zero_control(_controlled_z_standard))
decompositions.get()["C(PhaseShift)"].append(flip_zero_control(_controlled_phase_shift_standard))
decompositions.get()["C(RX)"].append(flip_zero_control(_controlled_rx_standard))
decompositions.get()["C(Rot)"].append(flip_zero_control(_controlled_rot_standard))
decompositions.get()["Adjoint(Hadamard)"].append(self_adjoint)
decompositions.get()["Pow(Hadamard)"].append(pow_involutory2)
decompositions.get()["Adjoint(RX)"].append(_adjoint_negate_angle)
decompositions.get()["Pow(RX)"].append(_pow_scale_angle)
decompositions.get()["Adjoint(CNOT)"].append(self_adjoint)
decompositions.get()["Adjoint(PhaseShift)"].append(adjoint_rotation2)
decompositions.get()["Adjoint(ControlledPhaseShift)"].append(adjoint_rotation2)

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
from contextvars import ContextVar

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.decomposition import Resources
from pennylane.decomposition.decomposition_rule import DecompCollection
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    pow_involutory,
    pow_rotation,
    self_adjoint_legacy,
)
from pennylane.ops.identity import _controlled_g_phase_decomp
from pennylane.ops.qubit.non_parametric_ops import _controlled_hadamard

_decompositions = defaultdict(DecompCollection)
decompositions = ContextVar("_test_decompositions", default=_decompositions)


def to_resources(gate_count: dict, weighted_cost: float | None = None) -> Resources:
    """Wrap a dictionary of gate counts in a Resources object."""
    if weighted_cost is None:
        weighted_cost = sum(count for count in gate_count.values())
    gate_count = {abstractify(op): count for op, count in gate_count.items() if count >= 0}
    return Resources(gate_count, weighted_cost=weighted_cost)


@qp.register_resources({qp.Hadamard: 2, qp.CNOT: 1})
def _cz_to_cnot(*_, **__):
    raise NotImplementedError


decompositions.get()["CZ"].append(_cz_to_cnot)


@qp.register_resources({qp.Hadamard: 2, qp.CZ: 1})
def _cnot_to_cz(*_, **__):
    raise NotImplementedError


decompositions.get()["CNOT"].append(_cnot_to_cz)


def _multi_rz_decomposition_resources(num_wires):
    return {qp.RZ: 1, qp.CNOT: 2 * (num_wires - 1)}


@qp.register_resources(_multi_rz_decomposition_resources)
def _multi_rz_decomposition(*_, **__):
    raise NotImplementedError


decompositions.get()["MultiRZ"].append(_multi_rz_decomposition)


@qp.register_resources({qp.RZ: 2, qp.RX: 1, qp.GlobalPhase: 1})
def _hadamard_to_rz_rx(*_, **__):
    raise NotImplementedError


@qp.register_resources({qp.RZ: 1, qp.RY: 1, qp.GlobalPhase: 1})
def _hadamard_to_rz_ry(*_, **__):
    raise NotImplementedError


decompositions.get()["Hadamard"].append(_hadamard_to_rz_rx)
decompositions.get()["Hadamard"].append(_hadamard_to_rz_ry)


@qp.register_resources({qp.RX: 1, qp.RZ: 2})
def _ry_to_rx_rz(*_, **__):
    raise NotImplementedError


decompositions.get()["RY"].append(_ry_to_rx_rz)


@qp.register_resources({qp.RX: 2, qp.CZ: 2})
def _crx_to_rx_cz(*_, **__):
    raise NotImplementedError


@qp.register_resources({qp.RX: 2, qp.CNOT: 2, qp.RY: 4, qp.GlobalPhase: 4, qp.RZ: 4})
def _crx_to_rx_ry_cnot_ry_cnot_ry_cnot_rz(*_, **__):
    raise NotImplementedError


decompositions.get()["CRX"].append(_crx_to_rx_cz)
decompositions.get()["CRX"].append(_crx_to_rx_ry_cnot_ry_cnot_ry_cnot_rz)


@qp.register_resources({qp.RZ: 3, qp.CNOT: 2, qp.GlobalPhase: 1})
def _cphase_to_rz_cnot(*_, **__):
    raise NotImplementedError


decompositions.get()["ControlledPhaseShift"].append(_cphase_to_rz_cnot)


@qp.register_resources({qp.RZ: 1, qp.GlobalPhase: 1})
def _phase_shift_to_rz_gp(*_, **__):
    raise NotImplementedError


decompositions.get()["PhaseShift"].append(_phase_shift_to_rz_gp)


@qp.register_resources({qp.RX: 1, qp.GlobalPhase: 1})
def _x_to_rx(*_, **__):
    raise NotImplementedError


decompositions.get()["PauliX"].append(_x_to_rx)


@qp.register_resources({qp.PhaseShift: 1})
def _u1_ps(phi, wires, **__):
    qp.PhaseShift(phi, wires=wires)


decompositions.get()["U1"].append(_u1_ps)


@qp.register_resources({qp.PhaseShift: 1})
def _t_ps(wires, **__):
    raise NotImplementedError


decompositions.get()["T"].append(_t_ps)


@qp.register_resources({qp.RZ: 3, qp.RY: 2, qp.CNOT: 2})
def _crot(*_, **__):
    raise NotImplementedError


decompositions.get()["CRot"].append(_crot)

################################################
# Custom Decompositions For Symbolic Operators #
################################################

decompositions.get()["C(GlobalPhase)"].append(_controlled_g_phase_decomp)
decompositions.get()["C(Hadamard)"].append(_controlled_hadamard)
decompositions.get()["Adjoint(Hadamard)"].append(self_adjoint_legacy)
decompositions.get()["Pow(Hadamard)"].append(pow_involutory)
decompositions.get()["Adjoint(RX)"].append(adjoint_rotation)
decompositions.get()["Pow(RX)"].append(pow_rotation)
decompositions.get()["Adjoint(CNOT)"].append(self_adjoint_legacy)
decompositions.get()["Adjoint(PhaseShift)"].append(adjoint_rotation)
decompositions.get()["Adjoint(ControlledPhaseShift)"].append(adjoint_rotation)

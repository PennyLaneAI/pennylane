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
"""Resource effects of the built-in Catalyst passes.

.. warning::

    This module is a prototype and may change or be removed without deprecation.
"""

from math import log2

from .effects import FunctionResources, Op, Wires, identity, register_effect, substitution, transfer

register_effect("symbol-dce", identity)

GLOBAL_PHASE = Op("GlobalPhase", 0)

# +---------------------------------+
# |---- to-ppr ---------------------|
# +---------------------------------+

# Each Clifford+T gate becomes Pauli product rotations, with a global phase
# (mlir/lib/PBC/Transforms/ToPPR.cpp)
_PPR_OF_GATE = {
    "Hadamard": {Op("PPR-pi/4", 1): 3},
    "S": {Op("PPR-pi/4", 1): 1},
    "T": {Op("PPR-pi/8", 1): 1},
    "PauliX": {Op("PPR-pi/2", 1): 1},
    "PauliY": {Op("PPR-pi/2", 1): 1},
    "PauliZ": {Op("PPR-pi/2", 1): 1},
    "CNOT": {Op("PPR-pi/4", 2): 1, Op("PPR-pi/4", 1): 2},
    "CY": {Op("PPR-pi/4", 2): 1, Op("PPR-pi/4", 1): 2},
    "CZ": {Op("PPR-pi/4", 2): 1, Op("PPR-pi/4", 1): 2},
}

# Rotations by an angle that is not a known multiple of pi/4 become arbitrary-angle PPRs, without
# a global phase
_ROTATIONS = {"RX", "RY", "RZ", "IsingXX", "IsingYY", "IsingZZ", "MultiRZ", "PauliRot"}


def _base_name(name: str) -> str:
    """The name of an operator without its adjoint modifier, which only flips the rotation
    signs."""
    return name[len("Adjoint(") : -1] if name.startswith("Adjoint(") else name


@substitution("to-ppr")
def _to_ppr(op: Op, _options):
    name = _base_name(op.name)
    if name in _PPR_OF_GATE:
        return _PPR_OF_GATE[name] | {GLOBAL_PHASE: 1}
    if name == "Identity":
        # Converted to an identity PPR, which is then removed
        return {}
    if name == "MidCircuitMeasure":
        return {Op("PPM", 1): 1}
    if name in _ROTATIONS:
        # Constant angles that are multiples of pi/4 are converted exactly by the pass, but the
        # angles are not known here
        return {Op("PPR-Phi", op.wires): 1}
    return None


# +---------------------------------+
# |---- ppr-to-ppm -----------------|
# +---------------------------------+


@substitution("ppr-to-ppm")
def _ppr_to_ppm(op: Op, options):
    if options.get("decompose_method", "pauli-corrected") != "pauli-corrected" or options.get(
        "avoid_y_measure", False
    ):
        raise NotImplementedError(
            "The resource effect of ppr-to-ppm is only declared for decompose_method="
            "'pauli-corrected' and avoid_y_measure=False."
        )
    n = op.wires
    if op.name == "PPR-pi/8":
        # A magic state, a joint measurement with it, a measurement of it, and a Pauli correction
        # (mlir/lib/PBC/Transforms/DecomposeNonCliffordPPR.cpp)
        return {
            Op("pbc.fabricate", 0): 1,
            Op("PPM", n + 1): 1,
            Op("PPM", 1): 1,
            Op("PPR-pi/2", n): 1,
        }, Wires(alloc=1, peak=1)
    if op.name == "PPR-pi/4":
        # An ancilla, a joint measurement with it, a measurement of it, and a Pauli correction
        # (mlir/lib/PBC/Transforms/DecomposeCliffordPPR.cpp)
        return {Op("PPM", n + 1): 1, Op("PPM", 1): 1, Op("PPR-pi/2", n): 1}, Wires(alloc=1, peak=1)
    return None


# +---------------------------------+
# |---- combine-global-phases ------|
# +---------------------------------+


@transfer("combine-global-phases")
def _combine_global_phases(function: FunctionResources, _options):
    # The pass merges the global phases of each block. The resources of a function cover its
    # blocks, except for loop bodies and callees, which are functions of their own.
    ops = function.ops
    phases = ops.pop(GLOBAL_PHASE, 0) + ops.pop(Op("Adjoint(GlobalPhase)", 0), 0)
    if phases:
        ops[GLOBAL_PHASE] = min(phases, 1)
    function.ops = ops


# +---------------------------------+
# |---- gridsynth ------------------|
# +---------------------------------+

# Statistics of the gate sequences of the runtime, as attached to the decomposition loops by the
# pass (mlir/lib/Quantum/Transforms/GridsynthPatterns.cpp, `SequenceStatistics`): the expected
# number of entries of the T group and of the other cases is linear in log2(1/epsilon), and the
# frequency of each case within its group is fixed.

# Clifford+T cases: T, HT, SHT, I, X, Y, Z, H, S, S†
_CLIFFORD_T_CASES = (
    {"T": 1},
    {"Hadamard": 1, "T": 1},
    {"S": 1, "Hadamard": 1, "T": 1},
    {"Identity": 1},
    {"PauliX": 1},
    {"PauliY": 1},
    {"PauliZ": 1},
    {"Hadamard": 1},
    {"S": 1},
    {"Adjoint(S)": 1},
)
_CLIFFORD_T_T_CASES = (0, 1, 2)
_CLIFFORD_T_DEFAULT = {"Identity": 1}

# PPR cases: I, then X2, X4, X8, adjX2, adjX4, adjX8, and the same for Y and Z
_PPR_CASES = ({},) + tuple(
    {f"PPR-pi/{k}": 1} for _pauli in "XYZ" for _adjoint in (False, True) for k in (2, 4, 8)
)
_PPR_T_CASES = (3, 6, 9, 12, 15, 18)
_PPR_DEFAULT = {}

# (T slope, T offset, other slope, other offset, case frequencies)
_STATISTICS = {
    ("deterministic", False): (
        3.0590,
        1.1278,
        -0.0074,
        1.9849,
        (0.0111, 0.4982, 0.4907, 0.0408, 0.0652, 0.0694, 0.1654, 0.2603, 0.2344, 0.1646),
    ),
    ("deterministic", True): (
        3.0590,
        1.1278,
        0.0,
        0.0,
        (0.0, 0.0, 0.0, 0.1678, 0.0, 0.0, 0.1646, 0.0, 0.0, 0.1673)
        + (0.0, 0.0, 0.1630, 0.0, 0.0, 0.1716, 0.0, 0.0, 0.1657),
    ),
    ("mixed", False): (
        1.5346,
        0.3606,
        0.0024,
        2.5843,
        (0.0096, 0.4944, 0.4960, 0.0058, 0.0410, 0.0429, 0.2347, 0.2045, 0.2590, 0.2121),
    ),
    ("mixed", True): (
        1.5346,
        0.3606,
        0.0,
        0.0021,
        (0.0, 0.0, 0.0, 0.1658, 0.0, 0.0, 0.1661, 0.0, 0.0, 0.1654)
        + (0.0, 0.0, 0.1661, 0.4483, 0.3190, 0.1682, 0.0, 0.2328, 0.1685),
    ),
    ("projective", False): (
        0.5428,
        3.8695,
        -0.0010,
        1.9591,
        (0.0215, 0.4883, 0.4901, 0.0251, 0.0583, 0.0589, 0.1787, 0.2849, 0.2334, 0.1606),
    ),
    ("projective", True): (
        0.5428,
        3.8695,
        0.0,
        0.0023,
        (0.0, 0.0, 0.0, 0.1601, 0.0, 0.0, 0.1706, 0.0, 0.0, 0.1594)
        + (0.0, 0.0, 0.1689, 0.0, 0.7692, 0.1695, 0.0, 0.2308, 0.1715),
    ),
    ("recovery", False): (
        1.5602,
        -8.6637,
        0.0066,
        2.4081,
        (0.0116, 0.4943, 0.4942, 0.0080, 0.0416, 0.0413, 0.2358, 0.2027, 0.2586, 0.2119),
    ),
    ("recovery", True): (
        1.5602,
        -8.6637,
        0.0,
        0.0339,
        (0.0, 0.0, 0.0, 0.1661, 0.0, 0.0, 0.1660, 0.0, 0.0, 0.1640)
        + (0.0, 0.0, 0.1668, 0.3579, 0.3158, 0.1703, 0.0, 0.3263, 0.1668),
    ),
}

# Probability that the projective rotation of the mixed fallback method fails
_FALLBACK_FAILURE_PROBABILITY = 0.0244


def _sequence(statistics: str, epsilon: float, ppr_basis: bool) -> dict[Op, float]:
    """The expected gates of one gate sequence of the runtime."""
    t_slope, t_offset, other_slope, other_offset, frequencies = _STATISTICS[statistics, ppr_basis]
    cases, t_cases, default = (
        (_PPR_CASES, _PPR_T_CASES, _PPR_DEFAULT)
        if ppr_basis
        else (_CLIFFORD_T_CASES, _CLIFFORD_T_T_CASES, _CLIFFORD_T_DEFAULT)
    )
    num_t = max(t_slope * log2(1 / epsilon) + t_offset, 0)
    num_other = max(other_slope * log2(1 / epsilon) + other_offset, 0)
    t_sum = sum(f for i, f in enumerate(frequencies) if i in t_cases)
    other_sum = sum(f for i, f in enumerate(frequencies) if i not in t_cases)

    gates = {}
    num_cases = 0
    for i, (case, frequency) in enumerate(zip(cases, frequencies)):
        group, group_sum = (num_t, t_sum) if i in t_cases else (num_other, other_sum)
        count = group / group_sum * frequency if group_sum else 0
        num_cases += count
        for name, n in case.items():
            gates[Op(name, 1)] = gates.get(Op(name, 1), 0) + n * count
    # The loop runs at least once, and the default case takes the remaining iterations
    for name, n in default.items():
        gates[Op(name, 1)] = gates.get(Op(name, 1), 0) + n * max(1 - num_cases, 0)
    return gates


def _add(total: dict[Op, float], gates: dict[Op, float], scale: float = 1):
    for op, count in gates.items():
        total[op] = total.get(op, 0) + scale * count


@substitution("gridsynth")
def _gridsynth(op: Op, options):
    is_rotation = op.wires == 1 and _base_name(op.name) in ("RZ", "PhaseShift")
    # Only Z rotations are decomposed, but the Pauli word is not known here
    if not (is_rotation or op == Op("PPR-Phi", 1)):
        return None
    epsilon = options.get("epsilon", 1e-4)
    ppr_basis = options.get("ppr_basis", False)
    method = options.get("method", "deterministic")

    gates = {GLOBAL_PHASE: 1}
    if _base_name(op.name) == "PhaseShift":
        _add(gates, {GLOBAL_PHASE: 1})
    if method != "mixed-fallback":
        _add(gates, _sequence(method, epsilon, ppr_basis))
        return gates

    # An ancilla rotated between two CY gates and measured, with a fallback sequence on failure
    # (`buildMixedFallback`)
    if ppr_basis:
        # Each CY is converted as in to-ppr
        _add(gates, _PPR_OF_GATE["CY"], scale=2)
        _add(gates, {Op("PPM", 1): 1})
    else:
        _add(gates, {Op("CY", 2): 2, Op("MidCircuitMeasure", 1): 1})
    _add(gates, _sequence("projective", epsilon, ppr_basis))
    _add(gates, _sequence("recovery", epsilon, ppr_basis), scale=_FALLBACK_FAILURE_PROBABILITY)
    return gates, Wires(alloc=1, peak=1)

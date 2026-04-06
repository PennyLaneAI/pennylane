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

"""This module contains definitions of common gate sets.

.. currentmodule:: pennylane.decomposition.gate_sets

.. autosummary::
   :toctree: api

   ~IDENTITY
   ~CLIFFORD_T
   ~CLIFFORD_T_PLUS_RZ
   ~ROTATIONS_PLUS_CNOT
   ~MBQC_GATES
   ~ALL_QUBIT_OPS
   ~ALL_OPS

"""

from pennylane import ops

from .gate_set import GateSet

_CLIFFORD_T_ORIGINAL = GateSet(
    {
        ops.X,
        ops.Y,
        ops.Z,
        ops.H,
        ops.S,
        ops.SX,
        ops.T,
        ops.CNOT,
        ops.CY,
        ops.CZ,
        ops.SWAP,
        ops.ISWAP,
    }
)

_ADJOINT_CLIFFORD_T = GateSet({f"Adjoint({op})" for op in _CLIFFORD_T_ORIGINAL})

_CLIFFORD_T = _CLIFFORD_T_ORIGINAL | _ADJOINT_CLIFFORD_T

_MID_MEASURE = GateSet({ops.MidMeasure})

IDENTITY = GateSet({ops.Identity: 0.0, ops.GlobalPhase: 0.0}, name="Identity")
"""The gate set that contains only the ``Identity`` and the ``GlobalPhase``."""

CLIFFORD_T = _CLIFFORD_T | IDENTITY | _MID_MEASURE
"""The standard Clifford+T gate set.

The Clifford+T gate set contains the following gates: ``PauliX``, ``PauliY``, ``PauliZ``, 
``Hadamard``, ``S``, ``SX``, ``T``, ``CNOT``, ``CY``, ``CZ``, ``SWAP``, and ``ISWAP``, as 
well as the ``Identity`` and ``GlobalPhase``. The ``MidMeasure`` operator is also included.

"""

CLIFFORD_T.name = "Clifford+T"

CLIFFORD_T_PLUS_RZ = CLIFFORD_T | GateSet({ops.RZ: 100})
"""The Clifford+T gate set with an additional ``RZ`` gate.

This gate set is used to decompose all non-parametrized operators to the standard Clifford+T
gate set, and all parametrized operators to the Clifford+T gate set and an ``RZ``. This
is typically useful when preparing for a discretization step such as the gridsynth which
completes the Clifford+T decomposition of a circuit.

"""


CLIFFORD_T_PLUS_RZ.name = "Clifford+T+RZ"

# All the PennyLane gates that are supported by PyZX. Note that PyZX also supports many more gates,
# detailed here: https://pyzx.readthedocs.io/en/latest/notebooks/gates.html
PYZX = GateSet(
    {
        ops.Z,
        ops.X,
        ops.Y,
        ops.H,
        ops.RX,
        ops.RY,
        ops.RZ,
        ops.U2,
        ops.U3,
        ops.S,
        ops.T,
        ops.SX,
        ops.SWAP,
        ops.CNOT,
        ops.CY,
        ops.CZ,
        ops.CRX,
        ops.CRY,
        ops.CRZ,
        ops.CPhase,
        ops.CSWAP,
        ops.Toffoli,
        ops.CCZ,
    }
)

ROTATIONS_PLUS_CNOT = GateSet({ops.RX, ops.RY, ops.RZ, ops.CNOT}) | IDENTITY | _MID_MEASURE
"""The gate set that contains single-qubit rotations and ``CNOT``."""

ROTATIONS_PLUS_CNOT.name = "Rotations+CNOT"

_MBQC_GATES = GateSet({ops.CNOT, ops.H, ops.S, "RotXZX", ops.RZ, ops.X, ops.Y, ops.Z})

MBQC_GATES = _MBQC_GATES | IDENTITY | _MID_MEASURE
"""The gate set used by the ``pennylane.ftqc`` module."""

MBQC_GATES.name = "MBQC"

ALL_QUBIT_OPS = GateSet(ops.qubit.__all__, name="All Qubit Gates")
"""The gate set that contains all operators defined in ``qml.ops.qubit``."""

ALL_OPS = GateSet(ops.__all__, name="All PennyLane Gates")
"""The gate set that contains all operators defined in ``qml.ops``."""

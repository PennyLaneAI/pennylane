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
"""Shared helpers for TrotterCDF and TrotterCGF tests."""

import numpy as np
from scipy.linalg import expm

import pennylane as qp

TROTTER_GATE_SET = {
    "Hadamard",
    "PauliX",
    "BasisRotation",
    "RZ",
    "IsingZZ",
    "CNOT",
    "GlobalPhase",
    "PhaseShift",
    "StatePrep",
}


def random_orthogonal(n, rng):
    """Generate a random orthogonal matrix via expm of a skew-symmetric matrix."""
    A = rng.normal(size=(n, n)) * 0.5
    A = A - A.T
    return expm(A)


def phase_free_close(A, B, atol=1e-8):
    """Compare two matrices up to a global phase (``A == e^{i.} B``)."""
    tr = np.trace(B.conj().T @ A)
    phase = tr / abs(tr) if abs(tr) > 1e-12 else 1.0
    return np.allclose(A, phase * B, atol=atol)


def hadamard_test(
    trotter_cls, ham, sys_wires, t, steps, double_phase
):  # pylint: disable=too-many-arguments
    """Return (measured <X_anc>, psi) for the H-ctrl-<X> Hadamard-test circuit."""
    anc = "anc"
    dev = qp.device("default.qubit", wires=[anc] + list(sys_wires))
    rng = np.random.default_rng(2024)
    dim = 2 ** len(sys_wires)
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    psi /= np.linalg.norm(psi)

    @qp.qnode(dev)
    @qp.transforms.decompose(gate_set=TROTTER_GATE_SET)
    def circ():
        qp.StatePrep(psi, wires=sys_wires)
        qp.H(anc)
        qp.ctrl(
            trotter_cls(t, steps, ham, wires=sys_wires, double_phase=double_phase), control=[anc]
        )
        return qp.expval(qp.X(anc))

    return float(circ()), psi


def control_branches(
    trotter_cls, ham, sys_wires, t, steps, double_phase
):  # pylint: disable=too-many-arguments
    """Return the (control-0, control-1) branch unitaries of ctrl(Trotter*)."""
    anc = "anc"
    op = qp.ctrl(
        trotter_cls(t, steps, ham, wires=sys_wires, double_phase=double_phase), control=[anc]
    )
    [tape], _ = qp.transforms.decompose(
        [qp.tape.QuantumScript([op], [])], gate_set=TROTTER_GATE_SET
    )
    matrix = qp.matrix(tape, wire_order=[anc] + list(sys_wires))
    dim = 2 ** len(sys_wires)
    return matrix[:dim, :dim], matrix[dim:, dim:]

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

# Gate sets used by the catalyst integration tests of both templates. The genuine control
# compiles the controlled evolution into (controlled) PhaseShift/RZ rotations, while the
# double-phase variant keeps bare IsingZZ rotations sandwiched by CNOTs.
CATALYST_GATE_SET_GENUINE = {"Hadamard", "BasisRotation", "RZ", "CNOT", "PhaseShift", "ForLoop"}
CATALYST_GATE_SET_DOUBLE_PHASE = {"Hadamard", "BasisRotation", "RZ", "CNOT", "IsingZZ", "ForLoop"}


def random_orthogonal(n, rng):
    """Generate a random orthogonal matrix via expm of a skew-symmetric matrix."""
    A = rng.normal(size=(n, n)) * 0.5
    A = A - A.T
    return expm(A)


_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _single_z(wire, n_wires):
    """Return the ``Z`` operator on ``wire`` embedded in ``n_wires`` qubits (big-endian)."""
    out = np.array([[1.0]], dtype=complex)
    for w in range(n_wires):
        out = np.kron(out, _PAULI_Z if w == wire else np.eye(2))
    return out


def cdf_reference_hamiltonian(ham):
    """Exact Hamiltonian matrix implied by a CDF Hamiltonian dict.

    This is the documented mathematical definition (independent of the template's angle
    prefactors): under ``n_p = (I - Z_p) / 2`` a diagonal (identity-leaf) CDF Hamiltonian is

    ``H = s I + sum_wire (-Z0[p, p] / 2) Z_wire + sum_frag sum_{i<j} (Z[frag][p, q] / 4) Z_i Z_j``

    with ``s = _energy_shift(ham)``, spatial indices ``p = i // 2``, ``q = j // 2`` over the
    ``2N`` spin-orbital wires. For an identity-leaf Hamiltonian ``matrix(TrotterCDF)`` must
    equal ``expm(-i H t)``.
    """
    # Local import to avoid a module-load-time dependency on the template.
    from pennylane.templates.subroutines.time_evolution.trotter_cdf import (  # pylint: disable=import-outside-toplevel
        _energy_shift,
    )

    Z = np.asarray(ham["core_tensors"], dtype=float)
    num_cas = Z.shape[-1]
    n_wires = 2 * num_cas
    dim = 2**n_wires

    z_ops = [_single_z(w, n_wires) for w in range(n_wires)]
    H = _energy_shift(ham) * np.eye(dim, dtype=complex)
    Z0 = Z[0]
    for wire in range(n_wires):
        H += (-Z0[wire // 2, wire // 2] / 2) * z_ops[wire]
    for frag in range(1, Z.shape[0]):
        for i in range(n_wires):
            for j in range(i + 1, n_wires):
                H += (Z[frag][i // 2, j // 2] / 4) * (z_ops[i] @ z_ops[j])
    return H


def cgf_reference_hamiltonian(ham):
    """Exact Hamiltonian matrix implied by a (regrouped) CGF Hamiltonian dict.

    This is the documented mathematical definition (independent of the template's angle
    prefactors): under ``n^l_p = (I - Z_{l,p}) / 2`` a diagonal (identity-leaf) CGF
    Hamiltonian is

    ``H = s I + sum_{l,p} (-Z0[l,l,p,p] / 2) Z_{lp}
          + sum_frag sum_{l>m} sum_{p,q} (Z[frag][l,m][p,q] / 4) Z_{lp} Z_{mq}``

    with ``s = _energy_shift(ham)`` and wire index ``l * n_states + p``. For an identity-leaf
    Hamiltonian ``matrix(TrotterCGF)`` must equal ``expm(-i H t)``.
    """
    from pennylane.templates.subroutines.time_evolution.trotter_cgf import (  # pylint: disable=import-outside-toplevel
        _energy_shift,
    )

    Z = np.asarray(ham["core_tensors"], dtype=float)
    num_modes = Z.shape[1]
    n_states = Z.shape[-1]
    n_wires = num_modes * n_states
    dim = 2**n_wires

    def wire(l, p):
        return l * n_states + p

    z_ops = [_single_z(w, n_wires) for w in range(n_wires)]
    H = _energy_shift(ham) * np.eye(dim, dtype=complex)
    Z0 = Z[0]
    for l in range(num_modes):
        for p in range(n_states):
            H += (-Z0[l, l, p, p] / 2) * z_ops[wire(l, p)]
    for frag in range(1, Z.shape[0]):
        for l in range(num_modes):
            for m in range(l):
                for p in range(n_states):
                    for q in range(n_states):
                        H += (Z[frag][l, m][p, q] / 4) * (z_ops[wire(l, p)] @ z_ops[wire(m, q)])
    return H


def hadamard_test(
    trotter_cls, ham, sys_wires, t, steps, double_phase, seed
):  # pylint: disable=too-many-arguments
    """Return (measured <X_anc>, psi) for the H-ctrl-<X> Hadamard-test circuit."""
    anc = "anc"
    dev = qp.device("default.qubit", wires=[anc] + list(sys_wires))
    rng = np.random.default_rng(seed)
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

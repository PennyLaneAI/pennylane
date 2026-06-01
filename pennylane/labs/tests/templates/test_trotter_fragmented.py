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
"""
Tests for the trotter_fragmented module (CGF scheme only).
"""

# pylint: disable=no-value-for-parameter
import itertools
import math
import os

import numpy as np
import pytest
from scipy.linalg import expm

import pennylane as qp
from pennylane.labs.templates.trotter_fragmented import _energy_shift, trotter_fragmented

# pylint: disable=too-many-arguments, too-many-nested-blocks, redefined-outer-name, too-few-public-methods


def _random_orthogonal(n, rng):
    """Generate a random orthogonal matrix via expm of a skew-symmetric."""
    A = rng.normal(size=(n, n)) * 0.5
    A = A - A.T
    return expm(A)


@pytest.fixture(scope="module")
def toy_hamiltonian():
    """Synthetic CGF on 2 modes x 2 modals with 1 two-body fragment.
    4 qubits, 16-dim space, fast for convergence sweeps."""
    rng = np.random.default_rng(1)
    num_modes = 2
    n_states = 2

    eps = rng.normal(size=(num_modes, n_states)) * 0.4
    one_body_core_full = np.zeros((num_modes, num_modes, n_states, n_states))
    for l in range(num_modes):
        one_body_core_full[l, l] = np.diag(eps[l])
    one_body_leaf = np.stack([_random_orthogonal(n_states, rng) for _ in range(num_modes)])

    lam = rng.normal(size=(n_states, n_states)) * 0.35
    core_2b = np.zeros((1, num_modes, num_modes, n_states, n_states))
    core_2b[0, 1, 0] = lam
    leaf_2b = np.stack([np.stack([_random_orthogonal(n_states, rng) for _ in range(num_modes)])])

    core_tensors = np.concatenate([np.expand_dims(one_body_core_full, axis=0), core_2b], axis=0)
    leaf_tensors = np.concatenate([np.expand_dims(one_body_leaf, axis=0), leaf_2b], axis=0)
    hamiltonian = {
        "core_tensors": core_tensors,
        "leaf_tensors": leaf_tensors,
        "nuc_constant": 0.0,
    }
    return hamiltonian, num_modes, n_states


@pytest.fixture(scope="module")
def toy_multi_fragment():
    """Synthetic CGF on 2 modes x 2 modals with 2 two-body fragments."""
    rng = np.random.default_rng(42)
    num_modes = 2
    n_states = 2

    eps = rng.normal(size=(num_modes, n_states)) * 0.3
    one_body_core_full = np.zeros((num_modes, num_modes, n_states, n_states))
    for l in range(num_modes):
        one_body_core_full[l, l] = np.diag(eps[l])
    one_body_leaf = np.stack([_random_orthogonal(n_states, rng) for _ in range(num_modes)])

    # Two two-body fragments
    num_frags = 2
    core_2b = np.zeros((num_frags, num_modes, num_modes, n_states, n_states))
    leaf_2b_list = []
    for f in range(num_frags):
        lam = rng.normal(size=(n_states, n_states)) * 0.25
        core_2b[f, 1, 0] = lam
        leaf_2b_list.append(np.stack([_random_orthogonal(n_states, rng) for _ in range(num_modes)]))
    leaf_2b = np.stack(leaf_2b_list)

    core_tensors = np.concatenate([np.expand_dims(one_body_core_full, axis=0), core_2b], axis=0)
    leaf_tensors = np.concatenate([np.expand_dims(one_body_leaf, axis=0), leaf_2b], axis=0)
    hamiltonian = {
        "core_tensors": core_tensors,
        "leaf_tensors": leaf_tensors,
        "nuc_constant": 0.123,
    }
    return hamiltonian, num_modes, n_states


# Helper functions


def _qml_basis_rotation_matrix(leaf_frag, num_modes, n_states):
    """Return the full unitary for per-mode BasisRotation, matching
    what trotter_fragmented applies to each fragment."""
    num_qubits = num_modes * n_states
    wires = list(range(num_qubits))

    def _circuit():
        for l in range(num_modes):
            qp.BasisRotation(
                unitary_matrix=leaf_frag[l],
                wires=list(range(l * n_states, (l + 1) * n_states)),
            )

    return qp.matrix(_circuit, wire_order=wires)()


def build_H_exact(hamiltonian, num_modes, n_states):
    """Build the exact CGF Hamiltonian matrix in SBE encoding."""
    core_tensors = np.asarray(hamiltonian["core_tensors"])
    leaf_tensors = np.asarray(hamiltonian["leaf_tensors"])
    nuc_constant = hamiltonian["nuc_constant"]

    num_qubits = num_modes * n_states
    dim = 2**num_qubits
    wires = list(range(num_qubits))

    def get_Z(wire):
        return qp.matrix(qp.PauliZ(wire), wire_order=wires)

    Z_cache = [get_Z(w) for w in range(num_qubits)]
    I_full = np.eye(dim, dtype=complex)
    n_cache = [0.5 * (I_full - Z) for Z in Z_cache]

    H = nuc_constant * I_full

    # One-body (frag index 0)
    H_1b_diag = np.zeros((dim, dim), dtype=complex)
    for l in range(num_modes):
        for p in range(n_states):
            eps_lp = core_tensors[0, l, l, p, p]
            H_1b_diag = H_1b_diag + eps_lp * n_cache[l * n_states + p]
    U_1b = _qml_basis_rotation_matrix(leaf_tensors[0], num_modes, n_states)
    H = H + U_1b @ H_1b_diag @ U_1b.conj().T

    # Two-body fragments
    num_frags = leaf_tensors.shape[0] - 1
    for f in range(1, num_frags + 1):
        H_2b_diag = np.zeros((dim, dim), dtype=complex)
        for l in range(num_modes):
            for m in range(l):
                for p in range(n_states):
                    for q in range(n_states):
                        lam = core_tensors[f, l, m, p, q]
                        if abs(lam) > 0.0:
                            Z_p = Z_cache[l * n_states + p]
                            Z_q = Z_cache[m * n_states + q]
                            H_2b_diag = H_2b_diag + (lam / 4.0) * (Z_p @ Z_q)
        U_2b = _qml_basis_rotation_matrix(leaf_tensors[f], num_modes, n_states)
        H = H + U_2b @ H_2b_diag @ U_2b.conj().T

    H = 0.5 * (H + H.conj().T)
    return H


def sbe_subspace_indices(num_modes, n_states):
    """Indices of the physical (one-hot per mode) subspace."""
    num_qubits = num_modes * n_states
    indices = []
    for modal_choices in itertools.product(range(n_states), repeat=num_modes):
        idx = 0
        for l, p in enumerate(modal_choices):
            qubit = l * n_states + p
            idx |= 1 << (num_qubits - 1 - qubit)
        indices.append(idx)
    return np.array(sorted(indices))


def subspace_unitary_fidelity(U_ref, U_trial, subspace_idx):
    """|Tr(U_ref^dag U_trial)_sub| / dim_sub."""
    M = (U_ref.conj().T @ U_trial)[np.ix_(subspace_idx, subspace_idx)]
    return float(np.abs(np.trace(M)) / subspace_idx.size)


def subspace_operator_error(U_ref, U_trial, subspace_idx):
    """Operator-norm error on SBE subspace after global-phase alignment."""
    Aref = U_ref[np.ix_(subspace_idx, subspace_idx)]
    Atrial = U_trial[np.ix_(subspace_idx, subspace_idx)]
    overlap = np.trace(Aref.conj().T @ Atrial)
    phase = np.exp(-1j * np.angle(overlap)) if np.abs(overlap) > 1e-14 else 1.0
    return float(np.linalg.norm(Aref - phase * Atrial, ord=2))


def run_trotter_circuit(hamiltonian, num_modes, n_states, t, num_steps):
    """Run the Trotter circuit and return the full unitary matrix."""
    num_qubits = num_modes * n_states
    wires = list(range(num_qubits))

    def _circuit():
        trotter_fragmented(t, num_steps, hamiltonian, wires)

    return qp.matrix(_circuit, wire_order=wires)()


@pytest.mark.slow
class TestHighNConvergence:
    """Test that many Trotter steps converge U_trotter to expm(-i H t)."""

    @pytest.mark.skip
    @pytest.mark.parametrize("num_steps", [128, 256])
    def test_toy_convergence(self, toy_hamiltonian, num_steps):
        """At high N, subspace fidelity should be > 1 - 1e-4."""
        ham, num_modes, n_states = toy_hamiltonian
        H = build_H_exact(ham, num_modes, n_states)
        H_norm = float(np.linalg.norm(H, ord=2))
        sub_idx = sbe_subspace_indices(num_modes, n_states)

        t = 0.5 / max(H_norm, 1e-12)
        U_ref = expm(-1j * H * t)
        U_tr = run_trotter_circuit(ham, num_modes, n_states, t, num_steps)
        fidelity = subspace_unitary_fidelity(U_ref, U_tr, sub_idx)

        assert fidelity > 1 - 1e-4

    @pytest.mark.skip
    @pytest.mark.parametrize("num_steps", [128, 256])
    def test_multi_fragment_convergence(self, toy_multi_fragment, num_steps):
        """Multi-fragment Hamiltonian should also converge."""
        ham, num_modes, n_states = toy_multi_fragment
        H = build_H_exact(ham, num_modes, n_states)
        H_norm = float(np.linalg.norm(H, ord=2))
        sub_idx = sbe_subspace_indices(num_modes, n_states)

        t = 0.5 / max(H_norm, 1e-12)
        U_ref = expm(-1j * H * t)
        U_tr = run_trotter_circuit(ham, num_modes, n_states, t, num_steps)
        fidelity = subspace_unitary_fidelity(U_ref, U_tr, sub_idx)

        assert fidelity > 1 - 1e-4


class TestDtScaling:
    """Test that single-step Trotter error scales as dt^3 (2nd-order)."""

    @pytest.mark.parametrize(
        "dt",
        [
            (0.2),  # halving dt -> 8x error reduction
            (0.1),
            (0.05),
        ],
    )
    def test_dt_cubic_scaling(self, toy_hamiltonian, dt):
        """For a second-order Trotter evolution, error for a single time step should scale as dt^3.
        Check that halving dt reduces error by ~8x."""
        ham, num_modes, n_states = toy_hamiltonian
        H = build_H_exact(ham, num_modes, n_states)
        H_norm = float(np.linalg.norm(H, ord=2))
        sub_idx = sbe_subspace_indices(num_modes, n_states)

        dt_a = dt / max(H_norm, 1e-12)
        dt_b = dt / 2 / max(H_norm, 1e-12)

        U_ref_a = expm(-1j * H * dt_a)
        U_tr_a = run_trotter_circuit(ham, num_modes, n_states, dt_a, 1)
        err_a = subspace_operator_error(U_ref_a, U_tr_a, sub_idx)

        U_ref_b = expm(-1j * H * dt_b)
        U_tr_b = run_trotter_circuit(ham, num_modes, n_states, dt_b, 1)
        err_b = subspace_operator_error(U_ref_b, U_tr_b, sub_idx)

        if err_b <= 0:
            pytest.skip("Denominator error is zero; scaling check not meaningful.")

        ratio = err_a / err_b
        # expected_ratio = (dt_a / dt_b)^3 = 8.0 for halving
        log_dev = abs(math.log2(ratio + 1e-30) - math.log2(8.0)) / math.log2(8.0)
        assert log_dev <= 0.35


@pytest.mark.slow
class TestStepScaling:
    """Test that doubling Trotter steps reduces error by ~4x (1/N^2)."""

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "num_steps",
        [(2), (4), (8), (16)],
    )
    def test_step_quartic_scaling(self, toy_hamiltonian, num_steps):
        """Doubling number of steps should reduce subspace error by ~4x."""
        ham, num_modes, n_states = toy_hamiltonian
        H = build_H_exact(ham, num_modes, n_states)
        H_norm = float(np.linalg.norm(H, ord=2))
        sub_idx = sbe_subspace_indices(num_modes, n_states)

        t = 0.5 / max(H_norm, 1e-12)
        U_ref = expm(-1j * H * t)

        N_a = num_steps
        N_b = num_steps * 2
        U_tr_a = run_trotter_circuit(ham, num_modes, n_states, t, N_a)
        U_tr_b = run_trotter_circuit(ham, num_modes, n_states, t, N_b)

        err_a = subspace_operator_error(U_ref, U_tr_a, sub_idx)
        err_b = subspace_operator_error(U_ref, U_tr_b, sub_idx)

        if err_b <= 0:
            pytest.skip("Denominator error is zero; scaling check not meaningful.")

        ratio = err_a / err_b
        log_dev = abs(math.log2(ratio + 1e-30) - math.log2(4.0)) / math.log2(4.0)
        assert log_dev <= 0.35


class TestGlobalPhase:
    """Test that _energy_shift correctly tracks the Hamiltonian identity terms."""

    def test_energy_shift_toy(self, toy_hamiltonian):
        """Test that phase difference between exact and Trotter matches _energy_shift * t."""
        ham, num_modes, n_states = toy_hamiltonian
        H = build_H_exact(ham, num_modes, n_states)
        sub_idx = sbe_subspace_indices(num_modes, n_states)
        t = 0.05

        U_ref = expm(-1j * H * t)
        U_tr = run_trotter_circuit(ham, num_modes, n_states, t, num_steps=64)

        Aref = U_ref[np.ix_(sub_idx, sub_idx)]
        Atrial = U_tr[np.ix_(sub_idx, sub_idx)]

        overlap = np.trace(Aref.conj().T @ Atrial)
        measured_phase = np.angle(overlap)

        e_shift = _energy_shift(ham, frag_scheme="cgf")
        expected_phase = e_shift * t

        # Normalize to [-pi, pi]
        measured_phase = (measured_phase + np.pi) % (2 * np.pi) - np.pi
        expected_phase = (expected_phase + np.pi) % (2 * np.pi) - np.pi

        phase_error = abs(measured_phase - expected_phase)
        assert phase_error < 1e-5

    @pytest.mark.skip
    @pytest.mark.slow
    def test_energy_shift_with_nuc_constant(self, toy_multi_fragment):
        """Test that energy shift accounts for nonzero nuc_constant."""
        ham, num_modes, n_states = toy_multi_fragment
        H = build_H_exact(ham, num_modes, n_states)
        sub_idx = sbe_subspace_indices(num_modes, n_states)
        t = 0.05

        U_ref = expm(-1j * H * t)
        U_tr = run_trotter_circuit(ham, num_modes, n_states, t, num_steps=64)

        Aref = U_ref[np.ix_(sub_idx, sub_idx)]
        Atrial = U_tr[np.ix_(sub_idx, sub_idx)]

        overlap = np.trace(Aref.conj().T @ Atrial)
        measured_phase = np.angle(overlap)

        e_shift = _energy_shift(ham, frag_scheme="cgf")
        expected_phase = e_shift * t

        measured_phase = (measured_phase + np.pi) % (2 * np.pi) - np.pi
        expected_phase = (expected_phase + np.pi) % (2 * np.pi) - np.pi

        phase_error = abs(measured_phase - expected_phase)
        assert phase_error < 1e-5


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_zero_trotter_steps_is_identity(self, toy_hamiltonian):
        """Test that num_steps=0 produces the identity unitary."""
        ham, num_modes, n_states = toy_hamiltonian
        num_qubits = num_modes * n_states
        t = 1.0

        U = run_trotter_circuit(ham, num_modes, n_states, t, num_steps=0)
        I_expected = np.eye(2**num_qubits, dtype=complex)

        assert np.allclose(U, I_expected, atol=1e-12)

    def test_zero_evolution_time(self, toy_hamiltonian):
        """Check that the t=0 produces the identity regardless of steps."""
        ham, num_modes, n_states = toy_hamiltonian
        num_qubits = num_modes * n_states

        U = run_trotter_circuit(ham, num_modes, n_states, t=0.0, num_steps=10)
        I_expected = np.eye(2**num_qubits, dtype=complex)

        assert np.allclose(U, I_expected, atol=1e-12)

    def test_hermiticity_of_exact_H(self, toy_hamiltonian):
        """check that the exact Hamiltonian being built is Hermitian."""
        ham, num_modes, n_states = toy_hamiltonian
        H = build_H_exact(ham, num_modes, n_states)
        assert np.linalg.norm(H - H.conj().T) < 1e-12


class TestInputValidation:
    """Test that invalid inputs raise appropriate errors."""

    def test_invalid_tensor_ndim(self):
        """Test that mismatched core/leaf dimensions raise ValueError."""
        bad_ham = {
            "core_tensors": np.zeros((2, 3)),  # 2D - invalid
            "leaf_tensors": np.zeros((2, 3, 3)),  # 3D
            "nuc_constant": 0.0,
        }
        wires = list(range(6))

        with pytest.raises(ValueError, match="Could not auto-detect"):
            dev = qp.device("default.qubit", wires=wires)

            @qp.qnode(dev)
            def _circuit():
                trotter_fragmented(0.1, 1, bad_ham, wires)
                return qp.state()

            qp.matrix(_circuit)()


@pytest.mark.skip
class TestMonotonicity:
    """Test that increasing num_steps decreases error."""

    def test_error_decreases_with_steps(self, toy_hamiltonian):
        """Error should strictly decrease as num_steps increases."""
        ham, num_modes, n_states = toy_hamiltonian
        H = build_H_exact(ham, num_modes, n_states)
        H_norm = float(np.linalg.norm(H, ord=2))
        sub_idx = sbe_subspace_indices(num_modes, n_states)

        t = 0.5 / max(H_norm, 1e-12)
        U_ref = expm(-1j * H * t)

        num_steps_list = [2, 4, 8, 16, 32]
        errors = []
        for num_steps in num_steps_list:
            U_tr = run_trotter_circuit(ham, num_modes, n_states, t, num_steps)
            err = subspace_operator_error(U_ref, U_tr, sub_idx)
            errors.append(err)

        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1]


@pytest.fixture(scope="class")
def h2s_hamiltonian():
    """Fixture to load the data once per test class."""
    # Find the file relative to this test script
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(curr_dir, "cgf_corrected_2modals.npz")

    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found at {data_path}")

    with np.load(data_path) as data:
        return {
            "core_tensors": data["core_tensors"],
            "leaf_tensors": data["leaf_tensors"],
            "nuc_constant": data["nuc_constant"],
        }


@pytest.mark.slow
class TestH2SConvergence:
    """Integration tests on the real H2S molecule."""

    @pytest.mark.skip  # takes 8 mins
    def test_high_n_convergence(self, h2s_hamiltonian):
        """H2S should converge at N=64."""
        ham = h2s_hamiltonian
        num_modes, n_states = h2s_hamiltonian["core_tensors"].shape[2:4]

        H = build_H_exact(ham, num_modes, n_states)

        H_norm = float(np.linalg.norm(H, ord=2))
        sub_idx = sbe_subspace_indices(num_modes, n_states)

        t = 0.3 / max(H_norm, 1e-12)
        U_ref = expm(-1j * H * t)
        U_tr = run_trotter_circuit(ham, num_modes, n_states, t, num_steps=64)
        fidelity = subspace_unitary_fidelity(U_ref, U_tr, sub_idx)

        assert fidelity > 1 - 1e-4

    @pytest.mark.skip  # takes 40s to 3min
    @pytest.mark.parametrize("num_steps", [(2), (4), (8)])
    def test_step_scaling(self, h2s_hamiltonian, num_steps):
        """H2S step-doubling should reduce error by ~4x."""
        ham = h2s_hamiltonian
        num_modes, n_states = h2s_hamiltonian["core_tensors"].shape[2:4]
        H = build_H_exact(ham, num_modes, n_states)
        H_norm = float(np.linalg.norm(H, ord=2))
        sub_idx = sbe_subspace_indices(num_modes, n_states)

        t = 0.3 / max(H_norm, 1e-12)
        U_ref = expm(-1j * H * t)

        N_a = num_steps
        N_b = 2 * num_steps
        U_tr_a = run_trotter_circuit(ham, num_modes, n_states, t, N_a)
        U_tr_b = run_trotter_circuit(ham, num_modes, n_states, t, N_b)

        err_a = subspace_operator_error(U_ref, U_tr_a, sub_idx)
        err_b = subspace_operator_error(U_ref, U_tr_b, sub_idx)

        if err_b <= 0:
            pytest.skip("Denominator error is zero.")

        ratio = err_a / err_b
        expected = (N_b / N_a) ** 2
        log_dev = abs(math.log2(ratio + 1e-30) - math.log2(expected)) / math.log2(expected)
        assert log_dev <= 0.35

    @pytest.mark.skip  # takes 10min
    def test_global_phase(self, h2s_hamiltonian):
        """H2S energy shift should match the measured global phase."""
        ham = h2s_hamiltonian
        num_modes, n_states = h2s_hamiltonian["core_tensors"].shape[2:4]
        H = build_H_exact(ham, num_modes, n_states)
        sub_idx = sbe_subspace_indices(num_modes, n_states)
        t = 0.05

        U_ref = expm(-1j * H * t)
        U_tr = run_trotter_circuit(ham, num_modes, n_states, t, num_steps=64)

        Aref = U_ref[np.ix_(sub_idx, sub_idx)]
        Atrial = U_tr[np.ix_(sub_idx, sub_idx)]

        overlap = np.trace(Aref.conj().T @ Atrial)
        measured_phase = np.angle(overlap)

        e_shift = _energy_shift(ham, frag_scheme="cgf")
        expected_phase = e_shift * t

        measured_phase = (measured_phase + np.pi) % (2 * np.pi) - np.pi
        expected_phase = (expected_phase + np.pi) % (2 * np.pi) - np.pi

        phase_error = abs(measured_phase - expected_phase)
        assert phase_error < 1e-5


@pytest.mark.catalyst
def test_catalyst_legacy_frontend():
    """Test that the template runs while using the legacy catalyst frontend"""

    with qp.decomposition.toggle_graph_ctx(
        True
    ):  # safe alternative to avoid enabling graph globally on the labs test runner
        L = 2
        M = 2
        N = 2
        hamiltonian = {
            "core_tensors": np.random.rand(L, M, M, N, N),
            "leaf_tensors": np.random.rand(L, M, N, N),
            "nuc_constant": 0.5,
        }

        registers = qp.registers({"hadamard": 1, "system": M * N})

        target_gates = {
            "Hadamard",
            "BasisRotation",
            "RZ",
            "IsingZZ",
            "CNOT",
            "ForLoop",
        }

        @qp.qjit
        @qp.transforms.decompose(gate_set=target_gates)
        @qp.qnode(qp.device("lightning.qubit"))
        def trotter_circuit():
            qp.H(registers["hadamard"])

            trotter_fragmented(
                evolution_time=1.0,
                num_trotter_steps=10,
                hamiltonian=hamiltonian,
                wires=registers["system"],
                control_wires=registers["hadamard"],
            )

            return qp.expval(qp.X(registers["hadamard"]))

        assert not np.isclose(trotter_circuit(), 0)

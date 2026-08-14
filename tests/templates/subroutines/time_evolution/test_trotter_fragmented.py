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
Tests for the TrotterFragmented template.

There are no independent-physics-reference tests here (e.g. no Trotter-error-scaling
checks against an independently reconstructed Hamiltonian matrix). Instead, the private
helper functions for both the CDF (electronic-structure) and CGF (vibrational) branches
are unit tested directly, and the full registered decomposition rule is checked for
self-consistency: its output is compared against calling the very same private helpers
directly in plain Python.
"""

# pylint: disable=too-many-arguments, too-many-nested-blocks, redefined-outer-name, too-few-public-methods, wrong-import-position, protected-access
import numpy as np
import pytest
from scipy.linalg import expm

jax = pytest.importorskip("jax")

import pennylane as qp
from pennylane.decomposition.resources import Resources
from pennylane.templates.subroutines.time_evolution.trotter_fragmented import (
    _apply_one_body_diagonal,
    _apply_system_basis_rotation,
    _apply_two_body_diagonal,
    _energy_shift,
    _frag_scheme,
    _merge_leaves,
    _transpose_leaf,
    _trotter_step,
)
from pennylane.typing import Wire
from pennylane.wires import Wires

pytestmark = pytest.mark.jax


def _random_orthogonal(n, rng):
    """Generate a random orthogonal matrix via expm of a skew-symmetric."""
    A = rng.normal(size=(n, n)) * 0.5
    A = A - A.T
    return expm(A)


@pytest.fixture(scope="module")
def toy_hamiltonian():
    """Synthetic CGF Hamiltonian on 2 modes x 2 modals with 1 two-body fragment
    (4 qubits, 16-dim space)."""
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
        "nuc_constant": 0.7,
    }
    return hamiltonian, num_modes, n_states


@pytest.fixture(scope="module")
def toy_hamiltonian_cdf():
    """Synthetic CDF (electronic-structure) Hamiltonian: 2 spatial orbitals
    (4 qubits, alpha/beta interleaved), 1 two-body fragment."""
    rng = np.random.default_rng(7)
    num_orbitals = 2

    eps = rng.normal(size=num_orbitals) * 0.4
    one_body_core = np.diag(eps)
    one_body_leaf = _random_orthogonal(num_orbitals, rng)

    lam = rng.normal(size=(num_orbitals, num_orbitals)) * 0.3
    core_2b = np.expand_dims(lam, axis=0)
    leaf_2b = np.expand_dims(_random_orthogonal(num_orbitals, rng), axis=0)

    core_tensors = np.concatenate([np.expand_dims(one_body_core, axis=0), core_2b], axis=0)
    leaf_tensors = np.concatenate([np.expand_dims(one_body_leaf, axis=0), leaf_2b], axis=0)
    hamiltonian = {
        "core_tensors": core_tensors,
        "leaf_tensors": leaf_tensors,
        "nuc_constant": 0.6,
    }
    return hamiltonian, num_orbitals


# Helper functions


def run_trotter_circuit(hamiltonian, wires, t, num_steps, control_wires=()):
    """Run the Trotter circuit and return the full unitary matrix."""
    all_wires = list(wires) + list(control_wires)

    def _circuit():
        qp.TrotterFragmented(t, num_steps, hamiltonian, wires, control_wires)

    return qp.matrix(_circuit, wire_order=all_wires)()


def _matrix_from_ops(ops, wires):
    """Build the unitary matrix corresponding to applying ``ops`` in sequence."""

    def _circuit():
        for op in ops:
            qp.apply(op)

    return qp.matrix(_circuit, wire_order=wires)()


def _manual_one_step_decomposition(hamiltonian, wires, t, control_wires, frag_scheme):
    """Queue the ops for a single Trotter step by calling the private helpers
    directly in plain Python (i.e. without going through the registered rule's
    outer ``for_loop`` over Trotter steps)."""
    _trotter_step(0, t, hamiltonian, wires, control_wires, frag_scheme)
    very_last_U = _transpose_leaf(hamiltonian["leaf_tensors"][1], frag_scheme)
    _apply_system_basis_rotation(very_last_U, wires, frag_scheme)
    energy_shift = _energy_shift(hamiltonian, frag_scheme)
    phi = (energy_shift * t) % (4 * np.pi)
    if len(control_wires) > 0:
        qp.RZ(-phi, control_wires)
    else:
        qp.GlobalPhase(phi)


class TestInitialization:
    """Test that TrotterFragmented is initialized correctly."""

    def test_init_correctly(self, toy_hamiltonian):
        """Test that arguments and wires are stored correctly."""
        ham, num_modes, n_states = toy_hamiltonian
        wires = list(range(num_modes * n_states))
        op = qp.TrotterFragmented(0.3, 5, ham, wires)

        assert op.arguments["evolution_time"] == 0.3
        assert op.arguments["num_trotter_steps"] == 5
        assert op.arguments["hamiltonian"] is ham
        assert op.wires == Wires(wires)
        assert op._frag_scheme == "cgf"

    def test_control_wires_default_is_empty(self, toy_hamiltonian):
        """Test that omitting control_wires results in an empty Wires object, not None."""
        ham, num_modes, n_states = toy_hamiltonian
        wires = list(range(num_modes * n_states))
        op = qp.TrotterFragmented(0.3, 5, ham, wires)
        assert op.arguments["control_wires"] == Wires([])

    def test_explicit_control_wires(self, toy_hamiltonian):
        """Test that explicit control_wires are stored correctly and included in op.wires."""
        ham, num_modes, n_states = toy_hamiltonian
        wires = list(range(num_modes * n_states))
        op = qp.TrotterFragmented(0.3, 5, ham, wires, control_wires=[99])
        assert op.arguments["control_wires"] == Wires([99])
        assert 99 in op.wires


class TestValidity:
    """Basic structural validity tests for the TrotterFragmented operator."""

    def test_assert_valid_cgf(self, toy_hamiltonian):
        """Run qp.ops.functions.assert_valid on a concrete CGF TrotterFragmented instance."""
        ham, num_modes, n_states = toy_hamiltonian
        wires = list(range(num_modes * n_states))
        op = qp.TrotterFragmented(0.1, 3, ham, wires)
        # Differentiating through the (non-trainable) hamiltonian dict is not supported.
        qp.ops.functions.assert_valid(op, skip_differentiation=True)

    def test_assert_valid_cdf(self, toy_hamiltonian_cdf):
        """Run qp.ops.functions.assert_valid on a concrete CDF TrotterFragmented instance."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        op = qp.TrotterFragmented(0.1, 3, ham, wires)
        qp.ops.functions.assert_valid(op, skip_differentiation=True)


class TestCDFScheme:
    """Structural unit tests for the CDF-format (electronic structure) branch of
    the private helper functions, called directly and in isolation."""

    def test_frag_scheme_detects_cdf(self, toy_hamiltonian_cdf):
        """Test that a (3, 3) core/leaf tensor pair is detected as CDF."""
        ham, _ = toy_hamiltonian_cdf
        assert _frag_scheme(ham) == "cdf"

    def test_merge_leaves_cdf(self):
        """Test the CDF merge rule: U_prev^dagger @ U_curr."""
        rng = np.random.default_rng(0)
        U_prev = _random_orthogonal(3, rng)
        U_curr = _random_orthogonal(3, rng)
        assert np.allclose(_merge_leaves(U_prev, U_curr, "cdf"), U_prev.T @ U_curr)

    def test_transpose_leaf_cdf(self):
        """Test the CDF leaf transpose."""
        U = _random_orthogonal(3, np.random.default_rng(0))
        assert np.allclose(_transpose_leaf(U, "cdf"), U.T)

    def test_apply_system_basis_rotation_cdf_concrete(self):
        """Test that a non-identity rotation is applied to both spin channels."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        U = _random_orthogonal(num_cas, np.random.default_rng(1))
        with qp.queuing.AnnotatedQueue() as q:
            _apply_system_basis_rotation(U, wires, "cdf")
        tape = qp.tape.QuantumScript.from_queue(q)
        assert [type(op) for op in tape.operations] == [qp.BasisRotation, qp.BasisRotation]
        assert list(tape.operations[0].wires) == wires[::2]
        assert list(tape.operations[1].wires) == wires[1::2]

    def test_apply_system_basis_rotation_cdf_identity_skipped(self):
        """Test that an identity rotation is skipped for concrete (non-traced) data."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        with qp.queuing.AnnotatedQueue() as q:
            _apply_system_basis_rotation(np.eye(num_cas), wires, "cdf")
        tape = qp.tape.QuantumScript.from_queue(q)
        assert len(tape.operations) == 0

    def test_apply_system_basis_rotation_cdf_abstract(self):
        """Test that the identity-skip optimization does not apply under jax tracing."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        captured = {}

        def _fn(U):
            with qp.queuing.AnnotatedQueue() as q:
                _apply_system_basis_rotation(U, wires, "cdf")
            captured["ops"] = list(q.queue)
            return U

        jax.jit(_fn)(jax.numpy.eye(num_cas))
        assert [type(op) for op in captured["ops"]] == [qp.BasisRotation, qp.BasisRotation]

    def test_apply_two_body_diagonal_cdf(self):
        """Test the CDF two-body diagonal IsingZZ gates (all spin-orbital pairs)."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        Z = np.array([[0.0, 0.5], [0.5, 0.0]])
        t = 0.3
        with qp.queuing.AnnotatedQueue() as q:
            _apply_two_body_diagonal(Z, wires, t, [], "cdf")
        tape = qp.tape.QuantumScript.from_queue(q)
        ising_ops = [op for op in tape.operations if isinstance(op, qp.IsingZZ)]
        expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert len(ising_ops) == len(expected_pairs)
        for op, (i, j) in zip(ising_ops, expected_pairs):
            assert list(op.wires) == [wires[i], wires[j]]
            assert np.isclose(op.parameters[0], -0.25 * Z[i // 2, j // 2] * t)

    def test_apply_two_body_diagonal_cdf_with_control(self):
        """Test that CNOTs sandwich the IsingZZ block when control_wires is set."""
        wires = [0, 1]
        control_wires = [10]
        Z = np.array([[0.0]])
        with qp.queuing.AnnotatedQueue() as q:
            _apply_two_body_diagonal(Z, wires, 0.3, control_wires, "cdf")
        tape = qp.tape.QuantumScript.from_queue(q)
        cnots = [op for op in tape.operations if isinstance(op, qp.CNOT)]
        assert len(cnots) == 2
        assert all(list(op.wires) == [10, 0] for op in cnots)

    def test_apply_one_body_diagonal_cdf(self):
        """Test the CDF one-body diagonal RZ gates."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        Z = np.diag([0.3, -0.2])
        t = 0.1
        with qp.queuing.AnnotatedQueue() as q:
            _apply_one_body_diagonal(Z, wires, t, [], "cdf")
        tape = qp.tape.QuantumScript.from_queue(q)
        rz_ops = [op for op in tape.operations if isinstance(op, qp.RZ)]
        assert len(rz_ops) == 2 * num_cas
        for wire_idx, op in enumerate(rz_ops):
            assert list(op.wires) == [wires[wire_idx]]
            assert np.isclose(op.parameters[0], Z[wire_idx // 2, wire_idx // 2] * t)

    def test_apply_one_body_diagonal_cdf_with_control(self):
        """Test that CNOTs sandwich each RZ when control_wires is set."""
        wires = [0, 1]
        control_wires = [10]
        Z = np.diag([0.3])
        with qp.queuing.AnnotatedQueue() as q:
            _apply_one_body_diagonal(Z, wires, 0.1, control_wires, "cdf")
        tape = qp.tape.QuantumScript.from_queue(q)
        cnots = [op for op in tape.operations if isinstance(op, qp.CNOT)]
        assert len(cnots) == 2 * len(wires)
        assert all(op.wires[0] == 10 for op in cnots)

    def test_energy_shift_cdf(self):
        """Test the CDF zero-of-energy shift formula (Eq. A29, first line)."""
        rng = np.random.default_rng(3)
        num_cas = 2
        Z0 = np.diag(rng.normal(size=num_cas))
        Z_frag = rng.normal(size=(1, num_cas, num_cas))
        core_tensors = np.concatenate([Z0[np.newaxis], Z_frag], axis=0)
        ham = {"core_tensors": core_tensors, "nuc_constant": 0.42}

        shift = _energy_shift(ham, "cdf")
        expected = (
            0.42
            + np.trace(Z0)
            + (-np.sum(Z_frag) / 2 + np.sum(np.trace(Z_frag, axis1=1, axis2=2)) / 4)
        )
        assert np.isclose(shift, expected)


class TestCGFScheme:
    """Structural unit tests for the CGF-format (vibrational) branch of the
    private helper functions, called directly and in isolation."""

    def test_frag_scheme_detects_cgf(self, toy_hamiltonian):
        """Test that a (5, 4) core/leaf tensor pair is detected as CGF."""
        ham, _, _ = toy_hamiltonian
        assert _frag_scheme(ham) == "cgf"

    def test_merge_leaves_cgf(self):
        """Test the CGF merge rule: per-mode U_prev^dagger @ U_curr."""
        rng = np.random.default_rng(4)
        num_modes, n_states = 2, 3
        U_prev = np.stack([_random_orthogonal(n_states, rng) for _ in range(num_modes)])
        U_curr = np.stack([_random_orthogonal(n_states, rng) for _ in range(num_modes)])
        expected = np.stack([U_prev[l].T @ U_curr[l] for l in range(num_modes)])
        assert np.allclose(_merge_leaves(U_prev, U_curr, "cgf"), expected)

    def test_transpose_leaf_cgf(self):
        """Test the CGF leaf transpose (batched over the mode axis)."""
        rng = np.random.default_rng(5)
        num_modes, n_states = 2, 3
        U = np.stack([_random_orthogonal(n_states, rng) for _ in range(num_modes)])
        expected = np.stack([U[l].T for l in range(num_modes)])
        assert np.allclose(_transpose_leaf(U, "cgf"), expected)

    def test_apply_system_basis_rotation_cgf_concrete(self):
        """Test that per-mode rotations use the transpose convention and that a
        mode whose rotation is the identity is skipped."""
        num_modes, n_states = 2, 2
        wires = list(range(num_modes * n_states))
        U0 = _random_orthogonal(n_states, np.random.default_rng(2))
        U = np.stack([U0, np.eye(n_states)])
        with qp.queuing.AnnotatedQueue() as q:
            _apply_system_basis_rotation(U, wires, "cgf")
        tape = qp.tape.QuantumScript.from_queue(q)
        assert [type(op) for op in tape.operations] == [qp.BasisRotation]
        assert list(tape.operations[0].wires) == wires[:n_states]
        assert np.allclose(tape.operations[0].parameters[0], U0.T)

    def test_apply_system_basis_rotation_cgf_abstract(self):
        """Test that the identity-skip optimization does not apply under jax tracing."""
        num_modes, n_states = 2, 2
        wires = list(range(num_modes * n_states))
        captured = {}

        def _fn(U):
            with qp.queuing.AnnotatedQueue() as q:
                _apply_system_basis_rotation(U, wires, "cgf")
            captured["ops"] = list(q.queue)
            return U

        jax.jit(_fn)(jax.numpy.stack([jax.numpy.eye(n_states)] * num_modes))
        assert [type(op) for op in captured["ops"]] == [qp.BasisRotation, qp.BasisRotation]

    def test_apply_two_body_diagonal_cgf(self):
        """Test the CGF two-body diagonal IsingZZ gates for a single mode pair."""
        num_modes, n_states = 2, 2
        wires = list(range(num_modes * n_states))
        Z = np.zeros((num_modes, num_modes, n_states, n_states))
        Z[1, 0] = np.array([[0.1, 0.2], [0.3, 0.4]])
        t = 0.25
        with qp.queuing.AnnotatedQueue() as q:
            _apply_two_body_diagonal(Z, wires, t, [], "cgf")
        tape = qp.tape.QuantumScript.from_queue(q)
        ising_ops = [op for op in tape.operations if isinstance(op, qp.IsingZZ)]
        expected = [
            (wires[1 * n_states + p], wires[0 * n_states + q], 0.5 * Z[1, 0, p, q] * t)
            for p in range(n_states)
            for q in range(n_states)
        ]
        assert len(ising_ops) == len(expected)
        for op, (w0, w1, angle) in zip(ising_ops, expected):
            assert list(op.wires) == [w0, w1]
            assert np.isclose(op.parameters[0], angle)

    def test_apply_two_body_diagonal_cgf_with_control(self):
        """Test that CNOTs sandwich each (l, p) IsingZZ block when control_wires is set."""
        num_modes, n_states = 2, 1
        wires = [0, 1]
        control_wires = [10]
        Z = np.zeros((num_modes, num_modes, n_states, n_states))
        Z[1, 0, 0, 0] = 0.3
        with qp.queuing.AnnotatedQueue() as q:
            _apply_two_body_diagonal(Z, wires, 0.2, control_wires, "cgf")
        tape = qp.tape.QuantumScript.from_queue(q)
        cnots = [op for op in tape.operations if isinstance(op, qp.CNOT)]
        # A single (l=1, m=0) mode pair with n_states=1 -> a single (l, p) block -> 2 CNOTs.
        assert len(cnots) == 2
        assert all(list(op.wires) == [10, wires[1]] for op in cnots)

    def test_apply_one_body_diagonal_cgf(self):
        """Test the CGF one-body diagonal RZ gates."""
        num_modes, n_states = 2, 2
        wires = list(range(num_modes * n_states))
        Z = np.zeros((num_modes, num_modes, n_states, n_states))
        for l in range(num_modes):
            for p in range(n_states):
                Z[l, l, p, p] = 0.1 * (l + 1) * (p + 1)
        t = 0.2
        with qp.queuing.AnnotatedQueue() as q:
            _apply_one_body_diagonal(Z, wires, t, [], "cgf")
        tape = qp.tape.QuantumScript.from_queue(q)
        rz_ops = [op for op in tape.operations if isinstance(op, qp.RZ)]
        assert len(rz_ops) == num_modes * n_states
        idx = 0
        for l in range(num_modes):
            for p in range(n_states):
                wire_lp = wires[l * n_states + p]
                assert list(rz_ops[idx].wires) == [wire_lp]
                assert np.isclose(rz_ops[idx].parameters[0], -2.0 * Z[l, l, p, p] * t)
                idx += 1

    def test_apply_one_body_diagonal_cgf_with_control(self):
        """Test that CNOTs sandwich each RZ when control_wires is set."""
        num_modes, n_states = 1, 2
        wires = [0, 1]
        control_wires = [10]
        Z = np.zeros((num_modes, num_modes, n_states, n_states))
        Z[0, 0] = np.diag([0.3, -0.1])
        with qp.queuing.AnnotatedQueue() as q:
            _apply_one_body_diagonal(Z, wires, 0.1, control_wires, "cgf")
        tape = qp.tape.QuantumScript.from_queue(q)
        cnots = [op for op in tape.operations if isinstance(op, qp.CNOT)]
        assert len(cnots) == 2 * len(wires)
        assert all(op.wires[0] == 10 for op in cnots)

    def test_energy_shift_cgf(self):
        """Test the CGF zero-of-energy shift formula (one-body diagonal only)."""
        num_modes, n_states = 2, 2
        Z0 = np.zeros((num_modes, num_modes, n_states, n_states))
        for l in range(num_modes):
            for p in range(n_states):
                Z0[l, l, p, p] = 0.1 * (l + 1) * (p + 1)
        ham = {"core_tensors": Z0[np.newaxis], "nuc_constant": 0.3}

        shift = _energy_shift(ham, "cgf")
        expected = 0.3 + sum(Z0[l, l, p, p] for l in range(num_modes) for p in range(n_states)) / 2
        assert np.isclose(shift, expected)


class TestResourceRule:
    """Direct unit tests for the registered TrotterFragmented resource function,
    following the graph-based decomposition testing convention."""

    def test_num_trotter_steps_zero_has_no_resources(self, toy_hamiltonian):
        """Test that zero Trotter steps require zero resources."""
        ham, num_modes, n_states = toy_hamiltonian
        wires = list(range(num_modes * n_states))
        rule = qp.list_decomps(qp.TrotterFragmented)[0]

        resources = rule.compute_resources(
            evolution_time=1.0,
            num_trotter_steps=0,
            hamiltonian=ham,
            wires=wires,
            control_wires=(),
        )
        assert resources == Resources({})

    def test_cdf_resources_no_control(self, toy_hamiltonian_cdf):
        """Test that CDF resource estimation runs and reports a GlobalPhase (no control)."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        rule = qp.list_decomps(qp.TrotterFragmented)[0]

        resources = rule.compute_resources(
            evolution_time=1.0,
            num_trotter_steps=2,
            hamiltonian=ham,
            wires=wires,
            control_wires=(),
        )
        assert resources.num_gates > 0
        assert qp.resource_rep(qp.GlobalPhase) in resources.gate_counts
        assert qp.CNOT(wires=Wire[2]) not in resources.gate_counts

    def test_cdf_resources_with_control(self, toy_hamiltonian_cdf):
        """Test that CDF resource estimation with control_wires reports CNOTs
        instead of a GlobalPhase."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        rule = qp.list_decomps(qp.TrotterFragmented)[0]

        resources = rule.compute_resources(
            evolution_time=1.0,
            num_trotter_steps=2,
            hamiltonian=ham,
            wires=wires,
            control_wires=(99,),
        )
        assert resources.num_gates > 0
        assert qp.resource_rep(qp.GlobalPhase) not in resources.gate_counts
        assert qp.CNOT(wires=Wire[2]) in resources.gate_counts


class TestDecomposition:
    """Structural, self-consistency tests of the full registered decomposition
    rule: its output (for a single Trotter step) is compared against calling
    the same private helper functions directly in plain Python. No independent
    physics/numerical reference is used."""

    @pytest.mark.parametrize("control_wires", [(), (10,)])
    def test_decomposition_matches_manual_step_cdf(self, toy_hamiltonian_cdf, control_wires):
        """For num_trotter_steps=1, the registered rule should emit exactly the
        ops produced by manually calling the private helpers in sequence."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        t = 0.37
        frag_scheme = _frag_scheme(ham)

        with qp.queuing.AnnotatedQueue() as q_expected:
            _manual_one_step_decomposition(ham, wires, t, control_wires, frag_scheme)
        expected_ops = qp.tape.QuantumScript.from_queue(q_expected).operations

        rule = qp.list_decomps(qp.TrotterFragmented)[0]
        with qp.queuing.AnnotatedQueue() as q_actual:
            rule(
                evolution_time=t,
                num_trotter_steps=1,
                hamiltonian=ham,
                wires=wires,
                control_wires=control_wires,
            )
        actual_ops = qp.tape.QuantumScript.from_queue(q_actual).operations

        assert len(actual_ops) == len(expected_ops)
        for actual_op, expected_op in zip(actual_ops, expected_ops):
            qp.assert_equal(actual_op, expected_op)

    @pytest.mark.parametrize("control_wires", [(), (10,)])
    def test_decomposition_matches_manual_step_cgf(self, toy_hamiltonian, control_wires):
        """Same self-consistency check as above, for the CGF branch."""
        ham, num_modes, n_states = toy_hamiltonian
        wires = list(range(num_modes * n_states))
        t = 0.41
        frag_scheme = _frag_scheme(ham)

        with qp.queuing.AnnotatedQueue() as q_expected:
            _manual_one_step_decomposition(ham, wires, t, control_wires, frag_scheme)
        expected_ops = qp.tape.QuantumScript.from_queue(q_expected).operations

        rule = qp.list_decomps(qp.TrotterFragmented)[0]
        with qp.queuing.AnnotatedQueue() as q_actual:
            rule(
                evolution_time=t,
                num_trotter_steps=1,
                hamiltonian=ham,
                wires=wires,
                control_wires=control_wires,
            )
        actual_ops = qp.tape.QuantumScript.from_queue(q_actual).operations

        assert len(actual_ops) == len(expected_ops)
        for actual_op, expected_op in zip(actual_ops, expected_ops):
            qp.assert_equal(actual_op, expected_op)


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestIntegration:
    """Integration tests that check the TrotterFragmented template executes
    correctly end-to-end. As in TestDecomposition, correctness is checked via
    self-consistency (comparing circuit execution against directly multiplying
    the matrices of a manually-obtained op list), not an independent physics
    reference."""

    @pytest.mark.parametrize("control_wires", [(), (10,)])
    def test_execution_matches_manual_decomposition_cdf(self, toy_hamiltonian_cdf, control_wires):
        """Executing TrotterFragmented (num_trotter_steps=1) should match directly
        multiplying the matrices of the ops from a manual, plain-Python call to
        the same private helper functions used internally."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        all_wires = wires + list(control_wires)
        t = 0.29
        frag_scheme = _frag_scheme(ham)

        with qp.queuing.AnnotatedQueue() as q_expected:
            _manual_one_step_decomposition(ham, wires, t, control_wires, frag_scheme)
        expected_ops = qp.tape.QuantumScript.from_queue(q_expected).operations
        expected_matrix = _matrix_from_ops(expected_ops, all_wires)

        actual_matrix = run_trotter_circuit(ham, wires, t, num_steps=1, control_wires=control_wires)

        assert np.allclose(actual_matrix, expected_matrix)

    @pytest.mark.parametrize("control_wires", [(), (10,)])
    def test_execution_matches_manual_decomposition_cgf(self, toy_hamiltonian, control_wires):
        """Same self-consistency check as above, for the CGF branch."""
        ham, num_modes, n_states = toy_hamiltonian
        wires = list(range(num_modes * n_states))
        all_wires = wires + list(control_wires)
        t = 0.31
        frag_scheme = _frag_scheme(ham)

        with qp.queuing.AnnotatedQueue() as q_expected:
            _manual_one_step_decomposition(ham, wires, t, control_wires, frag_scheme)
        expected_ops = qp.tape.QuantumScript.from_queue(q_expected).operations
        expected_matrix = _matrix_from_ops(expected_ops, all_wires)

        actual_matrix = run_trotter_circuit(ham, wires, t, num_steps=1, control_wires=control_wires)

        assert np.allclose(actual_matrix, expected_matrix)

    def test_zero_trotter_steps_is_identity(self, toy_hamiltonian):
        """Test that num_steps=0 produces the identity unitary."""
        ham, num_modes, n_states = toy_hamiltonian
        wires = list(range(num_modes * n_states))
        t = 1.0

        U = run_trotter_circuit(ham, wires, t, num_steps=0)
        I_expected = np.eye(2 ** len(wires), dtype=complex)

        assert np.allclose(U, I_expected, atol=1e-12)

    def test_zero_evolution_time(self, toy_hamiltonian):
        """Check that t=0 produces the identity regardless of the number of steps."""
        ham, num_modes, n_states = toy_hamiltonian
        wires = list(range(num_modes * n_states))

        U = run_trotter_circuit(ham, wires, t=0.0, num_steps=10)
        I_expected = np.eye(2 ** len(wires), dtype=complex)

        assert np.allclose(U, I_expected, atol=1e-12)

    @pytest.mark.catalyst
    def test_catalyst_legacy_frontend(self):
        """Test that the template runs while using the legacy catalyst frontend."""
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

            qp.TrotterFragmented(
                evolution_time=1.0,
                num_trotter_steps=10,
                hamiltonian=hamiltonian,
                wires=registers["system"],
                control_wires=registers["hadamard"],
            )

            return qp.expval(qp.X(registers["hadamard"]))

        assert not np.isclose(trotter_circuit(), 0)


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
            qp.TrotterFragmented(0.1, 1, bad_ham, wires)

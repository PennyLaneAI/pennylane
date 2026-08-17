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
Tests for the TrotterCGF template.

The private CGF helper functions are unit tested directly, the registered base and
controlled decomposition rules are checked for self-consistency, and the controlled
constructions are validated numerically against ``qp.matrix`` of the base operator.
There are two controlled variants selected by the ``double_phase`` flag: the default
(``False``) genuine controlled unitary and the (``True``) double-phase Hadamard-test
circuit of Fig. 6 of arXiv:2506.15784.
"""

# pylint: disable=too-many-arguments, redefined-outer-name, too-few-public-methods, wrong-import-position, protected-access
import numpy as np
import pytest
from scipy.linalg import block_diag, expm

jax = pytest.importorskip("jax")

import pennylane as qp
from pennylane.decomposition.resources import Resources
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.time_evolution.trotter_cgf import (
    _apply_one_body_diagonal,
    _apply_system_basis_rotation,
    _apply_two_body_diagonal,
    _cgf_resource_counts,
    _energy_shift,
    _merge_leaves,
    _transpose_leaf,
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
def toy_hamiltonian_cgf():
    """Synthetic CGF (vibrational) Hamiltonian on 2 modes x 2 modals with 1 two-body
    fragment (4 qubits, 16-dim space)."""
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
def diagonal_hamiltonian_cgf():
    """CGF Hamiltonian with identity leaf tensors. All gates are diagonal (and thus
    commute), so the base circuit and both controlled constructions are Trotter-exact,
    enabling a machine-precision correctness check."""
    rng = np.random.default_rng(12)
    num_modes = 2
    n_states = 3
    L = 2
    core = rng.normal(size=(L + 1, num_modes, num_modes, n_states, n_states)) * 0.4
    leaf = np.stack([np.stack([np.eye(n_states) for _ in range(num_modes)]) for _ in range(L + 1)])
    return {"core_tensors": core, "leaf_tensors": leaf, "nuc_constant": 0.3}, num_modes, n_states


class TestInitialization:
    """Test that TrotterCGF is initialized correctly."""

    def test_init_correctly(self, toy_hamiltonian_cgf):
        """Test that arguments and wires are stored correctly."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        wires = list(range(num_modes * n_states))
        op = qp.TrotterCGF(0.3, 5, ham, wires)

        assert op.arguments["evolution_time"] == 0.3
        assert op.arguments["num_trotter_steps"] == 5
        assert op.arguments["hamiltonian"] is ham
        assert op.wires == Wires(wires)
        # double_phase defaults to False and only affects the controlled decomposition.
        assert op.arguments["double_phase"] is False
        assert (
            qp.TrotterCGF(0.3, 5, ham, wires, double_phase=True).arguments["double_phase"] is True
        )

    def test_abstract_init(self, toy_hamiltonian_cgf):
        """Test that an abstract instance (e.g. for resource-rep purposes) is built."""
        from pennylane.typing import Float

        ham, num_modes, n_states = toy_hamiltonian_cgf
        op = qp.TrotterCGF(Float, 5, ham, Wire[num_modes * n_states])
        assert op.is_abstract


class TestValidity:
    """Basic structural validity tests for the TrotterCGF operator."""

    def test_assert_valid(self, toy_hamiltonian_cgf):
        """Run qp.ops.functions.assert_valid on a concrete CGF instance."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        wires = list(range(num_modes * n_states))
        op = qp.TrotterCGF(0.1, 3, ham, wires)
        # Differentiating through the (non-trainable) hamiltonian dict is not supported.
        qp.ops.functions.assert_valid(op, skip_differentiation=True)


class TestCGFScheme:
    """Structural unit tests for the CGF-format private helper functions, called
    directly and in isolation."""

    def test_merge_leaves(self):
        """Test the CGF merge rule: per-mode U_prev^dagger @ U_curr."""
        rng = np.random.default_rng(4)
        num_modes, n_states = 2, 3
        U_prev = np.stack([_random_orthogonal(n_states, rng) for _ in range(num_modes)])
        U_curr = np.stack([_random_orthogonal(n_states, rng) for _ in range(num_modes)])
        expected = np.stack([U_prev[l].T @ U_curr[l] for l in range(num_modes)])
        assert np.allclose(_merge_leaves(U_prev, U_curr), expected)

    def test_transpose_leaf(self):
        """Test the CGF leaf transpose (batched over the mode axis)."""
        rng = np.random.default_rng(5)
        num_modes, n_states = 2, 3
        U = np.stack([_random_orthogonal(n_states, rng) for _ in range(num_modes)])
        expected = np.stack([U[l].T for l in range(num_modes)])
        assert np.allclose(_transpose_leaf(U), expected)

    def test_apply_system_basis_rotation_concrete(self):
        """Test that per-mode rotations use the transpose convention and that a mode
        whose rotation is the identity is skipped."""
        num_modes, n_states = 2, 2
        wires = list(range(num_modes * n_states))
        U0 = _random_orthogonal(n_states, np.random.default_rng(2))
        U = np.stack([U0, np.eye(n_states)])
        with qp.queuing.AnnotatedQueue() as q:
            _apply_system_basis_rotation(U, wires)
        tape = qp.tape.QuantumScript.from_queue(q)
        assert [type(op) for op in tape.operations] == [qp.BasisRotation]
        assert list(tape.operations[0].wires) == wires[:n_states]
        assert np.allclose(tape.operations[0].parameters[0], U0.T)

    def test_apply_system_basis_rotation_abstract(self):
        """Test that the identity-skip optimization does not apply under jax tracing."""
        num_modes, n_states = 2, 2
        wires = list(range(num_modes * n_states))
        captured = {}

        def _fn(U):
            with qp.queuing.AnnotatedQueue() as q:
                _apply_system_basis_rotation(U, wires)
            captured["ops"] = list(q.queue)
            return U

        jax.jit(_fn)(jax.numpy.stack([jax.numpy.eye(n_states)] * num_modes))
        assert [type(op) for op in captured["ops"]] == [qp.BasisRotation, qp.BasisRotation]

    def test_apply_two_body_diagonal(self):
        """Test the CGF two-body diagonal IsingZZ gates for a single mode pair."""
        num_modes, n_states = 2, 2
        wires = list(range(num_modes * n_states))
        Z = np.zeros((num_modes, num_modes, n_states, n_states))
        Z[1, 0] = np.array([[0.1, 0.2], [0.3, 0.4]])
        t = 0.25
        with qp.queuing.AnnotatedQueue() as q:
            _apply_two_body_diagonal(Z, wires, t, [], False)
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

    def test_apply_two_body_diagonal_double_phase_control(self):
        """Test that CNOTs sandwich each (l, p) IsingZZ block for the double-phase control."""
        num_modes, n_states = 2, 1
        wires = [0, 1]
        control_wires = [10]
        Z = np.zeros((num_modes, num_modes, n_states, n_states))
        Z[1, 0, 0, 0] = 0.3
        with qp.queuing.AnnotatedQueue() as q:
            _apply_two_body_diagonal(Z, wires, 0.2, control_wires, True)
        tape = qp.tape.QuantumScript.from_queue(q)
        cnots = [op for op in tape.operations if isinstance(op, qp.CNOT)]
        # A single (l=1, m=0) mode pair with n_states=1 -> a single (l, p) block -> 2 CNOTs.
        assert len(cnots) == 2
        assert all(list(op.wires) == [10, wires[1]] for op in cnots)
        assert any(isinstance(op, qp.IsingZZ) for op in tape.operations)

    def test_apply_two_body_diagonal_genuine_control(self):
        """Test that the genuine control emits controlled-IsingZZ (CNOT+RZ, no bare IsingZZ)."""
        num_modes, n_states = 2, 1
        wires = [0, 1]
        control_wires = [10]
        Z = np.zeros((num_modes, num_modes, n_states, n_states))
        Z[1, 0, 0, 0] = 0.3
        with qp.queuing.AnnotatedQueue() as q:
            _apply_two_body_diagonal(Z, wires, 0.2, control_wires, False)
        tape = qp.tape.QuantumScript.from_queue(q)
        # A single IsingZZ becomes a controlled-IsingZZ = 4 CNOT + 2 RZ.
        assert not any(isinstance(op, qp.IsingZZ) for op in tape.operations)
        assert sum(isinstance(op, qp.CNOT) for op in tape.operations) == 4
        assert sum(isinstance(op, qp.RZ) for op in tape.operations) == 2

    def test_apply_one_body_diagonal(self):
        """Test the CGF one-body diagonal RZ gates."""
        num_modes, n_states = 2, 2
        wires = list(range(num_modes * n_states))
        Z = np.zeros((num_modes, num_modes, n_states, n_states))
        for l in range(num_modes):
            for p in range(n_states):
                Z[l, l, p, p] = 0.1 * (l + 1) * (p + 1)
        t = 0.2
        with qp.queuing.AnnotatedQueue() as q:
            _apply_one_body_diagonal(Z, wires, t, [], False)
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

    def test_apply_one_body_diagonal_double_phase_control(self):
        """Test that CNOTs sandwich each RZ for the double-phase control."""
        num_modes, n_states = 1, 2
        wires = [0, 1]
        control_wires = [10]
        Z = np.zeros((num_modes, num_modes, n_states, n_states))
        Z[0, 0] = np.diag([0.3, -0.1])
        with qp.queuing.AnnotatedQueue() as q:
            _apply_one_body_diagonal(Z, wires, 0.1, control_wires, True)
        tape = qp.tape.QuantumScript.from_queue(q)
        cnots = [op for op in tape.operations if isinstance(op, qp.CNOT)]
        assert len(cnots) == 2 * len(wires)
        assert all(op.wires[0] == 10 for op in cnots)

    def test_apply_one_body_diagonal_genuine_control(self):
        """Test that the genuine control emits controlled-RZ (2 CNOT + 2 RZ per rotation)."""
        num_modes, n_states = 1, 2
        wires = [0, 1]
        control_wires = [10]
        Z = np.zeros((num_modes, num_modes, n_states, n_states))
        Z[0, 0] = np.diag([0.3, -0.1])
        with qp.queuing.AnnotatedQueue() as q:
            _apply_one_body_diagonal(Z, wires, 0.1, control_wires, False)
        tape = qp.tape.QuantumScript.from_queue(q)
        # Two one-body RZ rotations -> 2 x (2 CNOT + 2 RZ).
        assert sum(isinstance(op, qp.CNOT) for op in tape.operations) == 4
        assert sum(isinstance(op, qp.RZ) for op in tape.operations) == 4

    def test_energy_shift(self):
        """Test the CGF zero-of-energy shift formula (one-body diagonal only)."""
        num_modes, n_states = 2, 2
        Z0 = np.zeros((num_modes, num_modes, n_states, n_states))
        for l in range(num_modes):
            for p in range(n_states):
                Z0[l, l, p, p] = 0.1 * (l + 1) * (p + 1)
        ham = {"core_tensors": Z0[np.newaxis], "nuc_constant": 0.3}

        shift = _energy_shift(ham)
        expected = 0.3 + sum(Z0[l, l, p, p] for l in range(num_modes) for p in range(n_states)) / 2
        assert np.isclose(shift, expected)


class TestResourceRule:
    """Direct unit tests for the registered resource functions."""

    def test_num_trotter_steps_zero_has_no_resources(self, toy_hamiltonian_cgf):
        """Test that zero Trotter steps require zero resources."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        wires = list(range(num_modes * n_states))
        rule = qp.list_decomps(qp.TrotterCGF)[0]
        resources = rule.compute_resources(
            evolution_time=1.0,
            num_trotter_steps=0,
            hamiltonian=ham,
            wires=wires,
            double_phase=False,
        )
        assert resources == Resources({})

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_controlled_zero_steps_empty_decomposition(self, toy_hamiltonian_cgf):
        """Test that C(TrotterCGF) with zero steps emits no gates."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        wires = list(range(num_modes * n_states))
        op = qp.TrotterCGF(1.0, 0, ham, wires=wires)
        assert qp.ctrl(op, control=[99]).decomposition() == []

    def test_base_resources_report_global_phase(self, toy_hamiltonian_cgf):
        """Test that the base (uncontrolled) resources report a GlobalPhase, not CNOTs."""
        ham, _, _ = toy_hamiltonian_cgf
        counts = _cgf_resource_counts(2, ham, has_control=False)
        assert qp.GlobalPhase in counts
        assert qp.CNOT not in counts
        assert qp.IsingZZ in counts

    def test_genuine_controlled_resources_have_no_isingzz(self, toy_hamiltonian_cgf):
        """The default (genuine) controlled resources report PhaseShift + CNOT/RZ, no IsingZZ."""
        ham, _, _ = toy_hamiltonian_cgf
        counts = _cgf_resource_counts(2, ham, has_control=True, double_phase=False)
        assert qp.PhaseShift in counts
        assert qp.CNOT in counts
        assert qp.IsingZZ not in counts
        assert qp.GlobalPhase not in counts

    def test_double_phase_controlled_resources_have_isingzz(self, toy_hamiltonian_cgf):
        """The double-phase controlled resources report IsingZZ + sandwiching CNOTs."""
        ham, _, _ = toy_hamiltonian_cgf
        counts = _cgf_resource_counts(2, ham, has_control=True, double_phase=True)
        assert qp.IsingZZ in counts
        assert qp.CNOT in counts
        assert qp.PhaseShift not in counts


class TestDecomposition:
    """Self-consistency tests of the registered base decomposition rule."""

    def test_decomposition_self_consistent(self, toy_hamiltonian_cgf):
        """The registered base rule is self-consistent with its resource function."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        wires = list(range(num_modes * n_states))
        op = qp.TrotterCGF(0.4, 2, ham, wires)
        for rule in qp.list_decomps(qp.TrotterCGF):
            _test_decomposition_rule(op, rule)


_GATE_SET = {
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


def _phase_free_close(A, B, atol=1e-8):
    """Compare two matrices up to a global phase (``A == e^{i.} B``)."""
    tr = np.trace(B.conj().T @ A)
    phase = tr / abs(tr) if abs(tr) > 1e-12 else 1.0
    return np.allclose(A, phase * B, atol=atol)


def _hadamard_test(ham, sys_wires, t, steps, double_phase):
    """Return (measured <X_anc>, psi) for the H-ctrl-<X> Hadamard-test circuit."""
    anc = "anc"
    dev = qp.device("default.qubit", wires=[anc] + list(sys_wires))
    rng = np.random.default_rng(2024)
    dim = 2 ** len(sys_wires)
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    psi /= np.linalg.norm(psi)

    @qp.qnode(dev)
    @qp.transforms.decompose(gate_set=_GATE_SET)
    def circ():
        qp.StatePrep(psi, wires=sys_wires)
        qp.H(anc)
        qp.ctrl(
            qp.TrotterCGF(t, steps, ham, wires=sys_wires, double_phase=double_phase), control=[anc]
        )
        return qp.expval(qp.X(anc))

    return float(circ()), psi


def _control_branches(ham, sys_wires, t, steps, double_phase):
    """Return the (control-0, control-1) branch unitaries of ctrl(TrotterCGF)."""
    anc = "anc"
    op = qp.ctrl(
        qp.TrotterCGF(t, steps, ham, wires=sys_wires, double_phase=double_phase), control=[anc]
    )
    [tape], _ = qp.transforms.decompose([qp.tape.QuantumScript([op], [])], gate_set=_GATE_SET)
    matrix = qp.matrix(tape, wire_order=[anc] + list(sys_wires))
    dim = 2 ** len(sys_wires)
    return matrix[:dim, :dim], matrix[dim:, dim:]


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestControlledDecomposition:
    """Tests for the default (genuine) C(TrotterCGF) controlled decomposition."""

    def test_controlled_decomposition_self_consistent(self, toy_hamiltonian_cgf):
        """The registered C(TrotterCGF) rule is self-consistent with its resources."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        wires = list(range(num_modes * n_states))
        op = qp.ctrl(qp.TrotterCGF(0.4, 2, ham, wires), control=[99])
        for rule in qp.list_decomps("C(TrotterCGF)"):
            _test_decomposition_rule(op, rule)

    def test_genuine_controlled_block_structure(self, toy_hamiltonian_cgf):
        """By default ctrl(TrotterCGF) is a genuine controlled unitary: its matrix is
        block-diagonal with the identity on the control-0 block and matrix(base) on the
        control-1 block."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        anc = "anc"
        t, steps = 0.5, 2
        op = qp.ctrl(qp.TrotterCGF(t, steps, ham, sys_wires), control=[anc])
        [tape], _ = qp.transforms.decompose([qp.tape.QuantumScript([op], [])], gate_set=_GATE_SET)
        matrix = qp.matrix(tape, wire_order=[anc] + sys_wires)
        u_base = qp.matrix(qp.TrotterCGF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        dim = 2 ** len(sys_wires)
        assert np.allclose(matrix, block_diag(np.eye(dim), u_base), atol=1e-9)

    def test_controlled_hadamard_test_exact_diagonal(self, diagonal_hamiltonian_cgf):
        """For a diagonal (commuting) Hamiltonian the genuine controlled Hadamard test
        reproduces Re<psi|e^{-iHt}|psi> = Re<psi|matrix(base)|psi> exactly."""
        ham, num_modes, n_states = diagonal_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        t, steps = 0.9, 3
        measured, psi = _hadamard_test(ham, sys_wires, t, steps, False)
        u_base = qp.matrix(qp.TrotterCGF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        ref = float(np.real(psi.conj() @ (u_base @ psi)))
        assert np.isclose(measured, ref, atol=1e-9)

    def test_controlled_hadamard_test_generic(self, toy_hamiltonian_cgf):
        """The genuine controlled operation is an exact controlled-matrix(base), so the
        Hadamard test matches Re<psi|matrix(base)|psi> to machine precision."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        t, steps = 0.7, 12
        measured, psi = _hadamard_test(ham, sys_wires, t, steps, False)
        u_base = qp.matrix(qp.TrotterCGF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        ref = float(np.real(psi.conj() @ (u_base @ psi)))
        assert np.isclose(measured, ref, atol=1e-9)

    def test_genuine_global_phase_is_phaseshift(self, toy_hamiltonian_cgf):
        """The genuine controlled global phase is a PhaseShift(-phi) on the control wire
        (controlled-GlobalPhase), not an RZ."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        anc = 99
        t = 0.5
        op = qp.ctrl(qp.TrotterCGF(t, 1, ham, sys_wires), control=[anc])
        [tape], _ = qp.transforms.decompose([qp.tape.QuantumScript([op], [])], gate_set=_GATE_SET)
        phase_shifts = [
            o for o in tape.operations if isinstance(o, qp.PhaseShift) and list(o.wires) == [anc]
        ]
        assert len(phase_shifts) == 1
        phi = float((_energy_shift(ham) * t) % (4 * np.pi))
        assert np.isclose(phase_shifts[0].parameters[0], -phi)


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestDoublePhaseControlledDecomposition:
    """Tests for the opt-in double-phase (Fig. 6) C(TrotterCGF) controlled decomposition."""

    def test_controlled_decomposition_self_consistent(self, toy_hamiltonian_cgf):
        """The double-phase C(TrotterCGF) rule is self-consistent with its resources."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        wires = list(range(num_modes * n_states))
        op = qp.ctrl(qp.TrotterCGF(0.4, 2, ham, wires, double_phase=True), control=[99])
        for rule in qp.list_decomps("C(TrotterCGF)"):
            _test_decomposition_rule(op, rule)

    def test_double_phase_branches(self, diagonal_hamiltonian_cgf):
        """The double-phase control-0 / control-1 blocks evolve by the full-time e^{-iHt} /
        e^{+iHt} (up to a per-branch global phase), which for a diagonal (Trotter-exact)
        Hamiltonian equal matrix(TrotterCGF(+/-t)). This matches the original
        ``trotter_fragmented`` decomposition."""
        ham, num_modes, n_states = diagonal_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        t, steps = 0.9, 3
        block0, block1 = _control_branches(ham, sys_wires, t, steps, True)
        dim = 2 ** len(sys_wires)
        v_pos = qp.matrix(qp.TrotterCGF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        v_neg = qp.matrix(qp.TrotterCGF(-t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        assert not np.allclose(block0, np.eye(dim))  # not a genuine controlled unitary
        assert _phase_free_close(block0, v_pos)
        assert _phase_free_close(block1, v_neg)

    def test_double_phase_hadamard_test_matches_branches(self, diagonal_hamiltonian_cgf):
        """The double-phase Hadamard test realizes Re<psi|V0^dag V1|psi>, where V0 / V1 are
        the (full-time) control-0 / control-1 branch unitaries."""
        ham, num_modes, n_states = diagonal_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        t, steps = 0.9, 3
        measured, psi = _hadamard_test(ham, sys_wires, t, steps, True)
        block0, block1 = _control_branches(ham, sys_wires, t, steps, True)
        ref = float(np.real(psi.conj() @ (block0.conj().T @ block1 @ psi)))
        assert np.isclose(measured, ref, atol=1e-9)

    def test_double_phase_global_phase_is_rz(self, toy_hamiltonian_cgf):
        """The double-phase controlled global phase is an RZ(-phi) on the control wire
        (matching the original ``trotter_fragmented`` decomposition)."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        anc = 99
        t = 0.5
        op = qp.ctrl(qp.TrotterCGF(t, 1, ham, sys_wires, double_phase=True), control=[anc])
        [tape], _ = qp.transforms.decompose([qp.tape.QuantumScript([op], [])], gate_set=_GATE_SET)
        rz_on_control = [
            o for o in tape.operations if isinstance(o, qp.RZ) and list(o.wires) == [anc]
        ]
        assert len(rz_on_control) == 1
        phi = float((_energy_shift(ham) * t) % (4 * np.pi))
        assert np.isclose(rz_on_control[0].parameters[0], -phi)


class TestIntegration:
    """Integration tests via the graph-based decomposition system."""

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("t, num_steps", [(1.0, 0), (0.0, 10)])
    def test_identity_edge_cases(self, toy_hamiltonian_cgf, t, num_steps):
        """Test that zero Trotter steps, or zero evolution time, produce the identity."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        wires = list(range(num_modes * n_states))

        def _circuit():
            qp.TrotterCGF(t, num_steps, ham, wires)

        U = qp.matrix(_circuit, wire_order=wires)()
        assert np.allclose(U, np.eye(2 ** len(wires), dtype=complex), atol=1e-12)

    @pytest.mark.catalyst
    def test_catalyst_legacy_frontend(self):
        """Test that the controlled template runs with the legacy catalyst frontend."""
        L = 2
        M = 2
        N = 2
        hamiltonian = {
            "core_tensors": np.random.rand(L, M, M, N, N),
            "leaf_tensors": np.random.rand(L, M, N, N),
            "nuc_constant": 0.5,
        }
        registers = qp.registers({"hadamard": 1, "system": M * N})
        target_gates = {"Hadamard", "BasisRotation", "RZ", "CNOT", "PhaseShift", "ForLoop"}

        @qp.qjit
        @qp.transforms.decompose(gate_set=target_gates)
        @qp.qnode(qp.device("lightning.qubit"))
        def trotter_circuit():
            qp.H(registers["hadamard"])
            qp.ctrl(
                qp.TrotterCGF(1.0, 10, hamiltonian, wires=registers["system"]),
                control=registers["hadamard"],
            )
            return qp.expval(qp.X(registers["hadamard"]))

        assert not np.isclose(trotter_circuit(), 0)


class TestInputValidation:
    """Test that invalid inputs raise appropriate errors."""

    def test_rejects_cdf_hamiltonian(self):
        """Test that a CDF-shaped Hamiltonian raises a ValueError."""
        bad_ham = {
            "core_tensors": np.zeros((2, 3, 3)),  # CDF core (ndim 3)
            "leaf_tensors": np.zeros((2, 3, 3)),  # CDF leaf (ndim 3)
            "nuc_constant": 0.0,
        }
        with pytest.raises(ValueError, match="TrotterCGF expects a CGF Hamiltonian"):
            qp.TrotterCGF(0.1, 1, bad_ham, list(range(6)))

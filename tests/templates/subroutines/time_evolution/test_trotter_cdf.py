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
Tests for the TrotterCDF template.

The private CDF helper functions are unit tested directly, the registered base and
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
from pennylane.templates.subroutines.time_evolution.trotter_cdf import (
    _apply_one_body_diagonal,
    _apply_system_basis_rotation,
    _apply_two_body_diagonal,
    _cdf_resource_counts,
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


@pytest.fixture(scope="module")
def diagonal_hamiltonian_cdf():
    """CDF Hamiltonian with identity leaf tensors. All gates are diagonal (and thus
    commute), so the base circuit and both controlled constructions are Trotter-exact,
    enabling a machine-precision correctness check."""
    rng = np.random.default_rng(11)
    num_orbitals = 2
    L = 2
    core = rng.normal(size=(L + 1, num_orbitals, num_orbitals)) * 0.4
    core = 0.5 * (core + np.transpose(core, (0, 2, 1)))
    leaf = np.stack([np.eye(num_orbitals) for _ in range(L + 1)])
    return {"core_tensors": core, "leaf_tensors": leaf, "nuc_constant": 0.3}, num_orbitals


class TestInitialization:
    """Test that TrotterCDF is initialized correctly."""

    def test_init_correctly(self, toy_hamiltonian_cdf):
        """Test that arguments and wires are stored correctly."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        op = qp.TrotterCDF(0.3, 5, ham, wires)

        assert op.arguments["evolution_time"] == 0.3
        assert op.arguments["num_trotter_steps"] == 5
        assert op.arguments["hamiltonian"] is ham
        assert op.wires == Wires(wires)
        # double_phase defaults to False and only affects the controlled decomposition.
        assert op.arguments["double_phase"] is False
        assert (
            qp.TrotterCDF(0.3, 5, ham, wires, double_phase=True).arguments["double_phase"] is True
        )

    def test_abstract_init(self, toy_hamiltonian_cdf):
        """Test that an abstract instance (e.g. for resource-rep purposes) is built."""
        from pennylane.typing import Float

        ham, num_orbitals = toy_hamiltonian_cdf
        op = qp.TrotterCDF(Float, 5, ham, Wire[2 * num_orbitals])
        assert op.is_abstract


class TestValidity:
    """Basic structural validity tests for the TrotterCDF operator."""

    def test_assert_valid(self, toy_hamiltonian_cdf):
        """Run qp.ops.functions.assert_valid on a concrete CDF instance."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        op = qp.TrotterCDF(0.1, 3, ham, wires)
        # Differentiating through the (non-trainable) hamiltonian dict is not supported.
        qp.ops.functions.assert_valid(op, skip_differentiation=True)


class TestCDFScheme:
    """Structural unit tests for the CDF-format private helper functions, called
    directly and in isolation."""

    def test_merge_leaves(self):
        """Test the CDF merge rule: U_prev^dagger @ U_curr."""
        rng = np.random.default_rng(0)
        U_prev = _random_orthogonal(3, rng)
        U_curr = _random_orthogonal(3, rng)
        assert np.allclose(_merge_leaves(U_prev, U_curr), U_prev.T @ U_curr)

    def test_transpose_leaf(self):
        """Test the CDF leaf transpose."""
        U = _random_orthogonal(3, np.random.default_rng(0))
        assert np.allclose(_transpose_leaf(U), U.T)

    def test_apply_system_basis_rotation_concrete(self):
        """Test that a non-identity rotation is applied to both spin channels."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        U = _random_orthogonal(num_cas, np.random.default_rng(1))
        with qp.queuing.AnnotatedQueue() as q:
            _apply_system_basis_rotation(U, wires)
        tape = qp.tape.QuantumScript.from_queue(q)
        assert [type(op) for op in tape.operations] == [qp.BasisRotation, qp.BasisRotation]
        assert list(tape.operations[0].wires) == wires[::2]
        assert list(tape.operations[1].wires) == wires[1::2]

    def test_apply_system_basis_rotation_identity_skipped(self):
        """Test that an identity rotation is skipped for concrete (non-traced) data."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        with qp.queuing.AnnotatedQueue() as q:
            _apply_system_basis_rotation(np.eye(num_cas), wires)
        tape = qp.tape.QuantumScript.from_queue(q)
        assert len(tape.operations) == 0

    def test_apply_system_basis_rotation_abstract(self):
        """Test that the identity-skip optimization does not apply under jax tracing."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        captured = {}

        def _fn(U):
            with qp.queuing.AnnotatedQueue() as q:
                _apply_system_basis_rotation(U, wires)
            captured["ops"] = list(q.queue)
            return U

        jax.jit(_fn)(jax.numpy.eye(num_cas))
        assert [type(op) for op in captured["ops"]] == [qp.BasisRotation, qp.BasisRotation]

    def test_apply_two_body_diagonal(self):
        """Test the CDF two-body diagonal IsingZZ gates (all spin-orbital pairs)."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        Z = np.array([[0.0, 0.5], [0.5, 0.0]])
        t = 0.3
        with qp.queuing.AnnotatedQueue() as q:
            _apply_two_body_diagonal(Z, wires, t, [], False)
        tape = qp.tape.QuantumScript.from_queue(q)
        ising_ops = [op for op in tape.operations if isinstance(op, qp.IsingZZ)]
        expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert len(ising_ops) == len(expected_pairs)
        for op, (i, j) in zip(ising_ops, expected_pairs):
            assert list(op.wires) == [wires[i], wires[j]]
            assert np.isclose(op.parameters[0], -0.25 * Z[i // 2, j // 2] * t)

    def test_apply_two_body_diagonal_double_phase_control(self):
        """Test that CNOTs sandwich the IsingZZ block for the double-phase control."""
        wires = [0, 1]
        control_wires = [10]
        Z = np.array([[0.0]])
        with qp.queuing.AnnotatedQueue() as q:
            _apply_two_body_diagonal(Z, wires, 0.3, control_wires, True)
        tape = qp.tape.QuantumScript.from_queue(q)
        cnots = [op for op in tape.operations if isinstance(op, qp.CNOT)]
        assert len(cnots) == 2
        assert all(list(op.wires) == [10, 0] for op in cnots)
        assert any(isinstance(op, qp.IsingZZ) for op in tape.operations)

    def test_apply_two_body_diagonal_genuine_control(self):
        """Test that the genuine control emits controlled-IsingZZ (CNOT+RZ, no bare IsingZZ)."""
        wires = [0, 1]
        control_wires = [10]
        Z = np.array([[0.5]])
        with qp.queuing.AnnotatedQueue() as q:
            _apply_two_body_diagonal(Z, wires, 0.3, control_wires, False)
        tape = qp.tape.QuantumScript.from_queue(q)
        # No bare IsingZZ; each rotation becomes a controlled-IsingZZ = 4 CNOT + 2 RZ.
        assert not any(isinstance(op, qp.IsingZZ) for op in tape.operations)
        assert sum(isinstance(op, qp.CNOT) for op in tape.operations) == 4
        assert sum(isinstance(op, qp.RZ) for op in tape.operations) == 2

    def test_apply_one_body_diagonal(self):
        """Test the CDF one-body diagonal RZ gates."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        Z = np.diag([0.3, -0.2])
        t = 0.1
        with qp.queuing.AnnotatedQueue() as q:
            _apply_one_body_diagonal(Z, wires, t, [], False)
        tape = qp.tape.QuantumScript.from_queue(q)
        rz_ops = [op for op in tape.operations if isinstance(op, qp.RZ)]
        assert len(rz_ops) == 2 * num_cas
        for wire_idx, op in enumerate(rz_ops):
            assert list(op.wires) == [wires[wire_idx]]
            assert np.isclose(op.parameters[0], Z[wire_idx // 2, wire_idx // 2] * t)

    def test_apply_one_body_diagonal_double_phase_control(self):
        """Test that CNOTs sandwich each RZ for the double-phase control."""
        wires = [0, 1]
        control_wires = [10]
        Z = np.diag([0.3])
        with qp.queuing.AnnotatedQueue() as q:
            _apply_one_body_diagonal(Z, wires, 0.1, control_wires, True)
        tape = qp.tape.QuantumScript.from_queue(q)
        cnots = [op for op in tape.operations if isinstance(op, qp.CNOT)]
        assert len(cnots) == 2 * len(wires)
        assert all(op.wires[0] == 10 for op in cnots)

    def test_apply_one_body_diagonal_genuine_control(self):
        """Test that the genuine control emits controlled-RZ (2 CNOT + 2 RZ per rotation)."""
        wires = [0, 1]
        control_wires = [10]
        Z = np.diag([0.3])  # num_cas == 1 -> 2 * num_cas == 2 one-body RZ rotations
        with qp.queuing.AnnotatedQueue() as q:
            _apply_one_body_diagonal(Z, wires, 0.1, control_wires, False)
        tape = qp.tape.QuantumScript.from_queue(q)
        assert sum(isinstance(op, qp.CNOT) for op in tape.operations) == 4
        assert sum(isinstance(op, qp.RZ) for op in tape.operations) == 4

    def test_energy_shift(self):
        """Test the CDF zero-of-energy shift formula (Eq. A29, first line)."""
        rng = np.random.default_rng(3)
        num_cas = 2
        Z0 = np.diag(rng.normal(size=num_cas))
        Z_frag = rng.normal(size=(1, num_cas, num_cas))
        core_tensors = np.concatenate([Z0[np.newaxis], Z_frag], axis=0)
        ham = {"core_tensors": core_tensors, "nuc_constant": 0.42}

        shift = _energy_shift(ham)
        expected = (
            0.42
            + np.trace(Z0)
            + (-np.sum(Z_frag) / 2 + np.sum(np.trace(Z_frag, axis1=1, axis2=2)) / 4)
        )
        assert np.isclose(shift, expected)


class TestResourceRule:
    """Direct unit tests for the registered resource functions, following the
    graph-based decomposition testing convention."""

    def test_num_trotter_steps_zero_has_no_resources(self, toy_hamiltonian_cdf):
        """Test that zero Trotter steps require zero resources."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        rule = qp.list_decomps(qp.TrotterCDF)[0]
        resources = rule.compute_resources(
            evolution_time=1.0,
            num_trotter_steps=0,
            hamiltonian=ham,
            wires=wires,
            double_phase=False,
        )
        assert resources == Resources({})

    def test_base_resources_report_global_phase(self, toy_hamiltonian_cdf):
        """Test that the base (uncontrolled) resources report a GlobalPhase, not CNOTs."""
        ham, _ = toy_hamiltonian_cdf
        counts = _cdf_resource_counts(2, ham, has_control=False)
        assert qp.GlobalPhase in counts
        assert qp.CNOT not in counts
        assert qp.IsingZZ in counts

    def test_genuine_controlled_resources_have_no_isingzz(self, toy_hamiltonian_cdf):
        """The default (genuine) controlled resources report PhaseShift + CNOT/RZ, no IsingZZ."""
        ham, _ = toy_hamiltonian_cdf
        counts = _cdf_resource_counts(2, ham, has_control=True, double_phase=False)
        assert qp.PhaseShift in counts
        assert qp.CNOT in counts
        assert qp.IsingZZ not in counts
        assert qp.GlobalPhase not in counts

    def test_double_phase_controlled_resources_have_isingzz(self, toy_hamiltonian_cdf):
        """The double-phase controlled resources report IsingZZ + sandwiching CNOTs."""
        ham, _ = toy_hamiltonian_cdf
        counts = _cdf_resource_counts(2, ham, has_control=True, double_phase=True)
        assert qp.IsingZZ in counts
        assert qp.CNOT in counts
        assert qp.PhaseShift not in counts


class TestDecomposition:
    """Self-consistency tests of the registered base decomposition rule."""

    def test_decomposition_self_consistent(self, toy_hamiltonian_cdf):
        """The registered base rule is self-consistent with its resource function."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        op = qp.TrotterCDF(0.4, 2, ham, wires)
        for rule in qp.list_decomps(qp.TrotterCDF):
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
            qp.TrotterCDF(t, steps, ham, wires=sys_wires, double_phase=double_phase), control=[anc]
        )
        return qp.expval(qp.X(anc))

    return float(circ()), psi


def _control_branches(ham, sys_wires, t, steps, double_phase):
    """Return the (control-0, control-1) branch unitaries of ctrl(TrotterCDF)."""
    anc = "anc"
    op = qp.ctrl(
        qp.TrotterCDF(t, steps, ham, wires=sys_wires, double_phase=double_phase), control=[anc]
    )
    [tape], _ = qp.transforms.decompose([qp.tape.QuantumScript([op], [])], gate_set=_GATE_SET)
    matrix = qp.matrix(tape, wire_order=[anc] + list(sys_wires))
    dim = 2 ** len(sys_wires)
    return matrix[:dim, :dim], matrix[dim:, dim:]


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestControlledDecomposition:
    """Tests for the default (genuine) C(TrotterCDF) controlled decomposition."""

    def test_controlled_decomposition_self_consistent(self, toy_hamiltonian_cdf):
        """The registered C(TrotterCDF) rule is self-consistent with its resources."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        op = qp.ctrl(qp.TrotterCDF(0.4, 2, ham, wires), control=[99])
        for rule in qp.list_decomps("C(TrotterCDF)"):
            _test_decomposition_rule(op, rule)

    def test_genuine_controlled_block_structure(self, toy_hamiltonian_cdf):
        """By default ctrl(TrotterCDF) is a genuine controlled unitary: its matrix is
        block-diagonal with the identity on the control-0 block and matrix(base) on the
        control-1 block."""
        ham, num_orbitals = toy_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        anc = "anc"
        t, steps = 0.5, 2
        op = qp.ctrl(qp.TrotterCDF(t, steps, ham, sys_wires), control=[anc])
        [tape], _ = qp.transforms.decompose([qp.tape.QuantumScript([op], [])], gate_set=_GATE_SET)
        matrix = qp.matrix(tape, wire_order=[anc] + sys_wires)
        u_base = qp.matrix(qp.TrotterCDF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        dim = 2 ** len(sys_wires)
        assert np.allclose(matrix, block_diag(np.eye(dim), u_base), atol=1e-9)

    def test_controlled_hadamard_test_exact_diagonal(self, diagonal_hamiltonian_cdf):
        """For a diagonal (commuting) Hamiltonian the genuine controlled Hadamard test
        reproduces Re<psi|e^{-iHt}|psi> = Re<psi|matrix(base)|psi> exactly."""
        ham, num_orbitals = diagonal_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        t, steps = 0.9, 3
        measured, psi = _hadamard_test(ham, sys_wires, t, steps, False)
        u_base = qp.matrix(qp.TrotterCDF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        ref = float(np.real(psi.conj() @ (u_base @ psi)))
        assert np.isclose(measured, ref, atol=1e-9)

    def test_controlled_hadamard_test_generic(self, toy_hamiltonian_cdf):
        """The genuine controlled operation is an exact controlled-matrix(base), so the
        Hadamard test matches Re<psi|matrix(base)|psi> to machine precision."""
        ham, num_orbitals = toy_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        t, steps = 0.7, 12
        measured, psi = _hadamard_test(ham, sys_wires, t, steps, False)
        u_base = qp.matrix(qp.TrotterCDF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        ref = float(np.real(psi.conj() @ (u_base @ psi)))
        assert np.isclose(measured, ref, atol=1e-9)

    def test_genuine_global_phase_is_phaseshift(self, toy_hamiltonian_cdf):
        """The genuine controlled global phase is a PhaseShift(-phi) on the control wire
        (controlled-GlobalPhase), not an RZ."""
        ham, num_orbitals = toy_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        anc = 99
        t = 0.5
        op = qp.ctrl(qp.TrotterCDF(t, 1, ham, sys_wires), control=[anc])
        [tape], _ = qp.transforms.decompose([qp.tape.QuantumScript([op], [])], gate_set=_GATE_SET)
        phase_shifts = [
            o for o in tape.operations if isinstance(o, qp.PhaseShift) and list(o.wires) == [anc]
        ]
        assert len(phase_shifts) == 1
        phi = float((_energy_shift(ham) * t) % (4 * np.pi))
        assert np.isclose(phase_shifts[0].parameters[0], -phi)


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestDoublePhaseControlledDecomposition:
    """Tests for the opt-in double-phase (Fig. 6) C(TrotterCDF) controlled decomposition."""

    def test_controlled_decomposition_self_consistent(self, toy_hamiltonian_cdf):
        """The double-phase C(TrotterCDF) rule is self-consistent with its resources."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        op = qp.ctrl(qp.TrotterCDF(0.4, 2, ham, wires, double_phase=True), control=[99])
        for rule in qp.list_decomps("C(TrotterCDF)"):
            _test_decomposition_rule(op, rule)

    def test_double_phase_branches(self, diagonal_hamiltonian_cdf):
        """The double-phase control-0 / control-1 blocks evolve by the full-time e^{-iHt} /
        e^{+iHt} (up to a per-branch global phase), which for a diagonal (Trotter-exact)
        Hamiltonian equal matrix(TrotterCDF(+/-t)). This matches the original
        ``trotter_fragmented`` decomposition."""
        ham, num_orbitals = diagonal_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        t, steps = 0.9, 3
        block0, block1 = _control_branches(ham, sys_wires, t, steps, True)
        dim = 2 ** len(sys_wires)
        v_pos = qp.matrix(qp.TrotterCDF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        v_neg = qp.matrix(qp.TrotterCDF(-t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        assert not np.allclose(block0, np.eye(dim))  # not a genuine controlled unitary
        assert _phase_free_close(block0, v_pos)
        assert _phase_free_close(block1, v_neg)

    def test_double_phase_hadamard_test_matches_branches(self, diagonal_hamiltonian_cdf):
        """The double-phase Hadamard test realizes Re<psi|V0^dag V1|psi>, where V0 / V1 are
        the (full-time) control-0 / control-1 branch unitaries."""
        ham, num_orbitals = diagonal_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        t, steps = 0.9, 3
        measured, psi = _hadamard_test(ham, sys_wires, t, steps, True)
        block0, block1 = _control_branches(ham, sys_wires, t, steps, True)
        ref = float(np.real(psi.conj() @ (block0.conj().T @ block1 @ psi)))
        assert np.isclose(measured, ref, atol=1e-9)

    def test_double_phase_global_phase_is_rz(self, toy_hamiltonian_cdf):
        """The double-phase controlled global phase is an RZ(-phi) on the control wire
        (matching the original ``trotter_fragmented`` decomposition)."""
        ham, num_orbitals = toy_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        anc = 99
        t = 0.5
        op = qp.ctrl(qp.TrotterCDF(t, 1, ham, sys_wires, double_phase=True), control=[anc])
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
    def test_identity_edge_cases(self, toy_hamiltonian_cdf, t, num_steps):
        """Test that zero Trotter steps, or zero evolution time, produce the identity."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))

        def _circuit():
            qp.TrotterCDF(t, num_steps, ham, wires)

        U = qp.matrix(_circuit, wire_order=wires)()
        assert np.allclose(U, np.eye(2 ** len(wires), dtype=complex), atol=1e-12)

    @pytest.mark.catalyst
    def test_catalyst_legacy_frontend(self):
        """Test that the controlled template runs with the legacy catalyst frontend."""
        N = 2
        L = 1
        rng = np.random.default_rng(0)
        hamiltonian = {
            "core_tensors": rng.random((L + 1, N, N)),
            "leaf_tensors": rng.random((L + 1, N, N)),
            "nuc_constant": 0.5,
        }
        registers = qp.registers({"hadamard": 1, "system": 2 * N})
        target_gates = {"Hadamard", "BasisRotation", "RZ", "CNOT", "PhaseShift", "ForLoop"}

        @qp.qjit
        @qp.transforms.decompose(gate_set=target_gates)
        @qp.qnode(qp.device("lightning.qubit"))
        def trotter_circuit():
            qp.H(registers["hadamard"])
            qp.ctrl(
                qp.TrotterCDF(1.0, 10, hamiltonian, wires=registers["system"]),
                control=registers["hadamard"],
            )
            return qp.expval(qp.X(registers["hadamard"]))

        assert not np.isclose(trotter_circuit(), 0)


class TestInputValidation:
    """Test that invalid inputs raise appropriate errors."""

    def test_rejects_cgf_hamiltonian(self):
        """Test that a CGF-shaped Hamiltonian raises a ValueError."""
        bad_ham = {
            "core_tensors": np.zeros((2, 2, 2, 3, 3)),  # CGF core (ndim 5)
            "leaf_tensors": np.zeros((2, 2, 3, 3)),  # CGF leaf (ndim 4)
            "nuc_constant": 0.0,
        }
        with pytest.raises(ValueError, match="TrotterCDF expects a CDF Hamiltonian"):
            qp.TrotterCDF(0.1, 1, bad_ham, list(range(6)))

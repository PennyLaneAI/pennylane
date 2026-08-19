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

Correctness is checked definitionally: for an identity-leaf (Trotter-exact) Hamiltonian the
base and both controlled constructions are compared against ``expm(-i H t)`` of the
Hamiltonian implied by the CDF definition. Only the leaf-handling helpers, which those
identity-leaf checks do not exercise, are additionally unit tested. There are two controlled
variants selected by the ``double_phase`` flag: the default (``False``) genuine controlled
unitary and the (``True``) double-phase Hadamard-test circuit of Fig. 6 of arXiv:2506.15784.
"""

# pylint: disable=too-many-arguments, redefined-outer-name, too-few-public-methods, wrong-import-position, protected-access
import numpy as np
import pytest
from scipy.linalg import expm

jax = pytest.importorskip("jax")

import pennylane as qp
from pennylane.decomposition.resources import Resources
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.time_evolution.trotter_cdf import (
    _apply_system_basis_rotation,
    _merge_leaves,
)
from pennylane.typing import Wire
from pennylane.wires import Wires
from tests.templates.subroutines.time_evolution.fermi_tools import (  # pylint: disable=no-name-in-module
    one_body_matrix,
    permute_qubits,
)
from tests.templates.subroutines.time_evolution.trotter_test_helpers import (  # pylint: disable=no-name-in-module
    CATALYST_GATE_SET_DOUBLE_PHASE,
    CATALYST_GATE_SET_GENUINE,
    _single_z,
    cdf_reference_hamiltonian,
    control_branches,
    hadamard_test,
    random_orthogonal,
)

pytestmark = pytest.mark.jax


def _basis_rotation_matrix(U, n_wires):
    """Matrix of ``BasisRotation(U)`` applied to the alpha and beta spin channels."""
    ops = [
        qp.BasisRotation(unitary_matrix=U, wires=range(0, n_wires, 2)),
        qp.BasisRotation(unitary_matrix=U, wires=range(1, n_wires, 2)),
    ]
    return qp.matrix(qp.tape.QuantumScript(ops), wire_order=range(n_wires))


def cdf_reference_hamiltonian_leaves(ham):
    """Exact Hamiltonian matrix implied by a CDF Hamiltonian dict with arbitrary (real
    orthogonal) leaves, built independently from the template.

    Each fragment is diagonal in its own orbital basis, so its lab-frame generator is
    ``B_l^dag D_l B_l`` with ``B_l = BasisRotation(leaf_tensors[l])`` (applied on both spin
    channels) and ``D_l`` the diagonal generator from the Implementation Details:
    ``D_0 = sum_wire (-Z0[p, p] / 2) Z_wire`` for the one-body fragment and
    ``D_l = sum_{i<j} (Z[l][p, q] / 4) Z_i Z_j`` for the two-body fragments. The scalar identity
    part is basis independent and equals ``s = _energy_shift(ham)``. Unlike the identity-leaf case
    this is only reproduced by ``matrix(TrotterCDF)`` in the many-step limit (second-order Trotter
    error ``~ 1 / steps^2``).
    """
    from pennylane.templates.subroutines.time_evolution.trotter_cdf import (  # pylint: disable=import-outside-toplevel
        _energy_shift,
    )

    Z = np.asarray(ham["core_tensors"], dtype=float)
    U = np.asarray(ham["leaf_tensors"], dtype=float)
    num_cas = Z.shape[-1]
    n_wires = 2 * num_cas
    dim = 2**n_wires

    z_ops = [_single_z(w, n_wires) for w in range(n_wires)]
    H = _energy_shift(ham) * np.eye(dim, dtype=complex)

    B0 = _basis_rotation_matrix(U[0], n_wires)
    D0 = np.zeros((dim, dim), dtype=complex)
    for wire in range(n_wires):
        D0 += (-Z[0][wire // 2, wire // 2] / 2) * z_ops[wire]
    H += B0.conj().T @ D0 @ B0

    for frag in range(1, Z.shape[0]):
        Bl = _basis_rotation_matrix(U[frag], n_wires)
        Dl = np.zeros((dim, dim), dtype=complex)
        for i in range(n_wires):
            for j in range(i + 1, n_wires):
                Dl += (Z[frag][i // 2, j // 2] / 4) * (z_ops[i] @ z_ops[j])
        H += Bl.conj().T @ Dl @ Bl
    return H


@pytest.fixture
def toy_hamiltonian_cdf(seed):
    """Synthetic CDF (electronic-structure) Hamiltonian: 2 spatial orbitals
    (4 qubits, alpha/beta interleaved), 1 two-body fragment."""
    rng = np.random.default_rng(seed)
    num_orbitals = 2

    eps = rng.normal(size=num_orbitals) * 0.4
    one_body_core = np.diag(eps)
    one_body_leaf = random_orthogonal(num_orbitals, rng)

    lam = rng.normal(size=(num_orbitals, num_orbitals)) * 0.3
    core_2b = np.expand_dims(lam, axis=0)
    leaf_2b = np.expand_dims(random_orthogonal(num_orbitals, rng), axis=0)

    core_tensors = np.concatenate([np.expand_dims(one_body_core, axis=0), core_2b], axis=0)
    leaf_tensors = np.concatenate([np.expand_dims(one_body_leaf, axis=0), leaf_2b], axis=0)
    hamiltonian = {
        "core_tensors": core_tensors,
        "leaf_tensors": leaf_tensors,
        "nuc_constant": 0.6,
    }
    return hamiltonian, num_orbitals


@pytest.fixture
def diagonal_hamiltonian_cdf(seed):
    """CDF Hamiltonian with identity leaf tensors. All gates are diagonal (and thus
    commute), so the base circuit and both controlled constructions are Trotter-exact,
    enabling a machine-precision correctness check."""
    rng = np.random.default_rng(seed)
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
    """Unit tests for the leaf-handling helpers, which the identity-leaf definitional
    tests below do not exercise numerically (angles, pair enumeration, the energy shift,
    and both controlled structures are all covered by the ``*_matches_expm`` tests)."""

    def test_merge_leaves(self, seed):
        """Test the CDF merge rule that combines consecutive fragment rotations as the
        conjugate transpose of the previous leaf times the current one."""
        rng = np.random.default_rng(seed)
        U_prev = random_orthogonal(3, rng)
        U_curr = random_orthogonal(3, rng)
        assert np.allclose(_merge_leaves(U_prev, U_curr), U_prev.conj().T @ U_curr)
        # complex leaves use the adjoint (conj().T), not the plain transpose
        C_prev = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        C_curr = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        assert np.allclose(_merge_leaves(C_prev, C_curr), C_prev.conj().T @ C_curr)

    def test_apply_system_basis_rotation(self, seed):
        """Test that a non-identity leaf is applied as a BasisRotation on both spin channels."""
        num_cas = 2
        wires = list(range(2 * num_cas))
        U = random_orthogonal(num_cas, np.random.default_rng(seed))
        tape = qp.tape.make_qscript(_apply_system_basis_rotation)(U, wires)
        assert [type(op) for op in tape.operations] == [qp.BasisRotation, qp.BasisRotation]
        assert list(tape.operations[0].wires) == wires[::2]
        assert list(tape.operations[1].wires) == wires[1::2]


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

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_controlled_zero_steps_empty_decomposition(self, toy_hamiltonian_cdf):
        """Test that C(TrotterCDF) with zero steps emits no gates."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        op = qp.TrotterCDF(1.0, 0, ham, wires=wires)
        assert qp.ctrl(op, control=[99]).decomposition() == []


class TestDecomposition:
    """Tests of the registered base decomposition rule."""

    def test_decomposition_self_consistent(self, toy_hamiltonian_cdf):
        """The registered base rule is self-consistent with its resource function."""
        ham, num_orbitals = toy_hamiltonian_cdf
        wires = list(range(2 * num_orbitals))
        op = qp.TrotterCDF(0.4, 2, ham, wires)
        for rule in qp.list_decomps(qp.TrotterCDF):
            _test_decomposition_rule(op, rule)

    def test_base_matches_expm(self, diagonal_hamiltonian_cdf):
        """For an identity-leaf (Trotter-exact) Hamiltonian, matrix(TrotterCDF) equals the
        exact evolution expm(-i H t) of the Hamiltonian implied by the CDF definition. This
        is the definitional check on the Trotter angle prefactors (and the energy shift)."""
        ham, num_orbitals = diagonal_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        t, steps = 0.9, 3
        u = qp.matrix(qp.TrotterCDF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        expected = expm(-1j * cdf_reference_hamiltonian(ham) * t)
        assert np.allclose(u, expected, atol=1e-9)

    @pytest.mark.slow
    def test_base_matches_expm_nonidentity_leaves(self, seed):
        """With random real-orthogonal leaves the circuit is no longer Trotter-exact, but
        matrix(TrotterCDF) converges to the exact expm(-i H t) at second order (error
        ~ 1 / steps^2). Here H is built independently from the fragment basis rotations,
        H = sum_l B_l^dag D_l B_l, which pins down the leaf conjugation direction and the
        consecutive-fragment merge that the identity-leaf checks do not exercise."""
        rng = np.random.default_rng(seed)
        num_orbitals, L = 2, 2
        core = rng.normal(size=(L + 1, num_orbitals, num_orbitals)) * 0.4
        core = 0.5 * (core + np.transpose(core, (0, 2, 1)))
        leaf = np.stack([random_orthogonal(num_orbitals, rng) for _ in range(L + 1)])
        ham = {"core_tensors": core, "leaf_tensors": leaf, "nuc_constant": 0.3}
        sys_wires = list(range(2 * num_orbitals))
        t = 0.6
        expected = expm(-1j * cdf_reference_hamiltonian_leaves(ham) * t)

        diffs = [
            float(
                np.linalg.norm(
                    qp.matrix(qp.TrotterCDF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
                    - expected
                )
            )
            for steps in (2, 4, 8)
        ]
        # second-order Trotter: halving the step size quarters the error
        assert diffs[0] > diffs[1] > diffs[2]
        assert diffs[1] / diffs[2] > 3.0
        assert diffs[2] < 5e-3

    def test_mixed_determinant_leaves_gauge_invariant(self, seed):
        """Negating an orbital column flips a leaf's determinant but leaves the projector
        ``|v><v|`` -- and hence the physical fragment -- unchanged. The template normalizes
        leaf determinants internally, so the circuit is invariant under this flip even when it
        yields mixed determinants (as ``qp.qchem.factorize`` does for many molecules, e.g. an
        ``eigh`` one-body leaf with ``det = -1`` next to ``expm`` two-body leaves). Without the
        normalization ``BasisRotation``'s real-orthogonal sign gauge would realize a different
        Hamiltonian."""
        rng = np.random.default_rng(seed)
        num_orbitals, L = 3, 2
        core = rng.normal(size=(L + 1, num_orbitals, num_orbitals)) * 0.4
        core = 0.5 * (core + np.transpose(core, (0, 2, 1)))
        leaf = np.stack([random_orthogonal(num_orbitals, rng) for _ in range(L + 1)])
        ham = {"core_tensors": core, "leaf_tensors": leaf, "nuc_constant": 0.3}

        flipped = leaf.copy()
        flipped[0][:, 0] *= -1.0  # negate one orbital column -> det(leaf[0]) flips (now mixed)
        assert np.linalg.det(leaf[0]) * np.linalg.det(flipped[0]) < 0
        ham_flipped = {**ham, "leaf_tensors": flipped}

        sys_wires = list(range(2 * num_orbitals))
        t, steps = 0.6, 3
        u = qp.matrix(qp.TrotterCDF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        u_flipped = qp.matrix(
            qp.TrotterCDF(t, steps, ham_flipped, wires=sys_wires), wire_order=sys_wires
        )
        assert np.allclose(u, u_flipped, atol=1e-12)

    def test_energy_shift_matches_literal(self):
        """The global phase equals ``exp(-i s t)`` with ``s`` the identity content of the CDF
        Hamiltonian, checked against a hard-coded literal computed here from the definition
        (Eq. A29) rather than by calling ``_energy_shift``. The ``RZ``/``IsingZZ`` layers are
        traceless generators (``det = 1``), so ``det(U) = exp(-i * dim * s * t)`` isolates ``s``
        independently of the circuit's basis-rotation content. Guards constant/energy-shift
        regressions that the (circular) identity-leaf check reuses ``_energy_shift`` for."""
        nuc = 0.2
        core0 = np.diag([0.1, 0.3])  # one-body: trace = 0.4
        core2b = np.array([[0.2, 0.1], [0.1, 0.4]])  # one two-body fragment
        ham = {
            "core_tensors": np.stack([core0, core2b]),
            "leaf_tensors": np.stack([np.eye(2), np.eye(2)]),  # identity leaves -> Trotter-exact
            "nuc_constant": nuc,
        }
        # s = nuc + tr(Z0) - sum(Z_l)/2 + sum(tr(Z_l))/4
        #   = 0.2 + 0.4 - 0.8/2 + 0.6/4 = 0.35
        s_literal = 0.35
        num_orbitals = 2
        wires = list(range(2 * num_orbitals))
        dim = 2 ** len(wires)
        t = 0.3
        u = qp.matrix(qp.TrotterCDF(t, 2, ham, wires=wires), wire_order=wires)
        # det(expm(-i H t)) = exp(-i t Tr H) and Tr H = dim * s (Z/ZZ terms are traceless)
        assert np.isclose(np.linalg.det(u), np.exp(-1j * t * dim * s_literal), atol=1e-9)

    def test_matches_fermionic_reference(self, seed):
        """The two-body layer realizes the genuine fermionic operator
        ``1/2 sum_pq lam_pq n~_p n~_q``, checked against an independent Jordan-Wigner reference
        built from occupation-number matrix elements (no ``BasisRotation`` anywhere in the
        reference). This pins the fermionic meaning of the CDF fragments and the blocked
        spin-orbital -> interleaved-wire mode ordering, which the self-referential leaf check
        cannot. ``matrix(TrotterCDF)`` converges to ``expm(-i H_phys t)`` at second order."""
        rng = np.random.default_rng(seed)
        num_orbitals, t = 2, 0.5
        n_wires = 2 * num_orbitals
        wires = list(range(n_wires))
        dim = 2**n_wires
        # alpha modes 0..N-1 then beta N..2N-1, placed on interleaved wires (2p alpha, 2p+1 beta)
        mode_on_wire = [i + s * num_orbitals for i in range(num_orbitals) for s in (0, 1)]
        # BasisRotation's real-orthogonal sign gauge is diag((-1)^k) for det = +1 leaves (all
        # leaves here, and after the template's determinant normalization, are det = +1)
        pi_gauge = np.diag([(-1.0) ** k for k in range(num_orbitals)])

        def onebody_wire_matrix(mat_orb):
            full = np.zeros((n_wires, n_wires), dtype=complex)
            for s in (0, 1):
                block = range(s * num_orbitals, s * num_orbitals + num_orbitals)
                full[np.ix_(block, block)] = mat_orb
            return permute_qubits(one_body_matrix(full), mode_on_wire)

        # physical fermionic data: H = C + one-body(h) + 1/2 sum_pq lam_pq n~_p n~_q
        nuc = 0.37
        h_raw = rng.normal(size=(num_orbitals, num_orbitals)) * 0.5
        h = 0.5 * (h_raw + h_raw.T)
        lam_raw = rng.normal(size=(num_orbitals, num_orbitals)) * 0.5
        lam = 0.5 * (lam_raw + lam_raw.T)
        leaf2 = random_orthogonal(num_orbitals, rng)
        vecs = [pi_gauge @ leaf2[:, p] for p in range(num_orbitals)]  # realized fragment orbitals

        h_phys = nuc * np.eye(dim, dtype=complex) + onebody_wire_matrix(h)
        num_ops = [onebody_wire_matrix(np.outer(vecs[p], vecs[p])) for p in range(num_orbitals)]
        for p in range(num_orbitals):
            for q in range(num_orbitals):
                h_phys = h_phys + 0.5 * lam[p, q] * (num_ops[p] @ num_ops[q])

        # documented regrouping: the two-body single-site terms fold into the one-body fragment
        mu = lam.sum(axis=1)
        vg = np.stack(vecs, axis=1)
        h_eff = h + vg @ np.diag(mu) @ vg.T
        eps, ve = np.linalg.eigh(h_eff)
        if np.linalg.det(pi_gauge @ ve) < 0:  # keep det(leaf_0) = +1 so its gauge is pi_gauge too
            ve[:, 0] = -ve[:, 0]
        leaf0 = pi_gauge @ ve
        ham = {
            "core_tensors": np.stack([np.diag(eps), lam]),
            "leaf_tensors": np.stack([leaf0, leaf2]),
            "nuc_constant": nuc,
        }

        expected = expm(-1j * h_phys * t)
        diffs = [
            float(
                np.linalg.norm(
                    qp.matrix(qp.TrotterCDF(t, steps, ham, wires=wires), wire_order=wires)
                    - expected
                )
            )
            for steps in (2, 4, 8)
        ]
        assert diffs[0] > diffs[1] > diffs[2]
        assert diffs[1] / diffs[2] > 3.0


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

    def test_genuine_controlled_matches_expm(self, diagonal_hamiltonian_cdf):
        """By default ctrl(TrotterCDF) is a genuine controlled unitary: for an identity-leaf
        (Trotter-exact) Hamiltonian its matrix is exactly block_diag(I, expm(-i H t))."""
        ham, num_orbitals = diagonal_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        t, steps = 0.9, 3
        block0, block1 = control_branches(qp.TrotterCDF, ham, sys_wires, t, steps, False)
        dim = 2 ** len(sys_wires)
        expected = expm(-1j * cdf_reference_hamiltonian(ham) * t)
        assert np.allclose(block0, np.eye(dim), atol=1e-9)
        assert np.allclose(block1, expected, atol=1e-9)


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

    def test_double_phase_controlled_matches_expm(self, diagonal_hamiltonian_cdf):
        """With double_phase=True, ctrl(TrotterCDF) realizes diag(U, U^dagger): for an
        identity-leaf (Trotter-exact) Hamiltonian the control-0 / control-1 blocks are
        exactly expm(-i H t) / expm(+i H t) (explicit global phases, no per-block phase)."""
        ham, num_orbitals = diagonal_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        t, steps = 0.9, 3
        block0, block1 = control_branches(qp.TrotterCDF, ham, sys_wires, t, steps, True)
        u = expm(-1j * cdf_reference_hamiltonian(ham) * t)
        assert np.allclose(block0, u, atol=1e-9)
        assert np.allclose(block1, u.conj().T, atol=1e-9)

    def test_double_phase_hadamard_invariant(self, diagonal_hamiltonian_cdf, seed):
        """The double-phase Hadamard test measures <X> = Re<psi|block0^dag block1|psi>, which
        for the exact blocks equals Re<psi|expm(+2 i H t)|psi>."""
        ham, num_orbitals = diagonal_hamiltonian_cdf
        sys_wires = list(range(2 * num_orbitals))
        t, steps = 0.9, 3
        measured, psi = hadamard_test(qp.TrotterCDF, ham, sys_wires, t, steps, True, seed)
        block0, block1 = control_branches(qp.TrotterCDF, ham, sys_wires, t, steps, True)
        ref_blocks = float(np.real(psi.conj() @ (block0.conj().T @ block1 @ psi)))
        ref_expm = float(
            np.real(psi.conj() @ (expm(2j * cdf_reference_hamiltonian(ham) * t) @ psi))
        )
        assert np.isclose(measured, ref_blocks, atol=1e-9)
        assert np.isclose(measured, ref_expm, atol=1e-9)


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
    @pytest.mark.parametrize(
        ("double_phase", "target_gates"),
        [
            (False, CATALYST_GATE_SET_GENUINE),
            (True, CATALYST_GATE_SET_DOUBLE_PHASE),
        ],
    )
    def test_catalyst_legacy_frontend(self, double_phase, target_gates, seed):
        """Test that the controlled template runs with the legacy catalyst frontend."""
        N = 2
        L = 1
        rng = np.random.default_rng(seed)
        hamiltonian = {
            "core_tensors": rng.random((L + 1, N, N)),
            "leaf_tensors": rng.random((L + 1, N, N)),
            "nuc_constant": 0.5,
        }
        registers = qp.registers({"hadamard": 1, "system": 2 * N})

        @qp.qjit
        @qp.transforms.decompose(gate_set=target_gates)
        @qp.qnode(qp.device("lightning.qubit"))
        def trotter_circuit():
            qp.H(registers["hadamard"])
            qp.ctrl(
                qp.TrotterCDF(
                    1.0, 10, hamiltonian, wires=registers["system"], double_phase=double_phase
                ),
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

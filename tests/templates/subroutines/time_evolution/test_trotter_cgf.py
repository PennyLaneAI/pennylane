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

Correctness is checked definitionally: for an identity-leaf (Trotter-exact) Hamiltonian the
base and both controlled constructions are compared against ``expm(-i H t)`` of the
Hamiltonian implied by the CGF definition. Only the leaf-handling helpers, which those
identity-leaf checks do not exercise, are additionally unit tested. There are two controlled
variants selected by the ``double_phase`` flag: the default (``False``) genuine controlled
unitary and the (``True``) double-phase Hadamard-test circuit of Fig. 6 of arXiv:2506.15784.
"""

# pylint: disable=too-many-arguments, redefined-outer-name, too-few-public-methods, wrong-import-position, protected-access
import itertools
import warnings

import numpy as np
import pytest
from scipy.linalg import expm

jax = pytest.importorskip("jax")

import pennylane as qp
from pennylane.decomposition.resources import Resources
from pennylane.exceptions import CaptureWarning
from pennylane.numeric_hamiltonians import CGFHamiltonian
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.time_evolution.trotter_cgf import (
    _apply_system_basis_rotation,
    _merge_leaves,
)
from pennylane.typing import Wire, Float
from pennylane.wires import Wires
from tests.templates.subroutines.time_evolution.trotter_test_helpers import (  # pylint: disable=no-name-in-module
    CATALYST_GATE_SET_DOUBLE_PHASE,
    CATALYST_GATE_SET_GENUINE,
    _single_z,
    cgf_reference_hamiltonian,
    control_branches,
    hadamard_test,
    random_orthogonal,
)

pytestmark = pytest.mark.jax


def _cgf_basis_rotation_matrix(A, n_states):
    """Matrix of a fragment's per-mode ``BasisRotation``\\ s ``A[l]`` on the mode-major
    unary register (mode ``l`` occupies wires ``[l * n_states : (l + 1) * n_states]``)."""
    num_modes = A.shape[0]
    n_wires = num_modes * n_states
    ops = [
        qp.BasisRotation(unitary_matrix=A[l], wires=range(l * n_states, (l + 1) * n_states))
        for l in range(num_modes)
    ]
    return qp.matrix(qp.tape.QuantumScript(ops), wire_order=range(n_wires))


def cgf_reference_hamiltonian_leaves(ham):
    """Exact Hamiltonian matrix implied by a CGF Hamiltonian dict with arbitrary (real
    orthogonal) leaves, built independently from the template.

    Each fragment is diagonal in its own per-mode basis, so its lab-frame generator is
    ``B_frag^dag D_frag B_frag`` with ``B_frag`` the product of the per-mode
    :class:`~.BasisRotation`\\ s and ``D_frag`` the diagonal generator from the Implementation
    Details. The one-body and two-body leaves follow *opposite* modal conventions: the one-body
    leaf stores its eigenvectors as columns, so ``B_0`` uses ``leaf_tensors[0][l]`` directly,
    while each two-body leaf stores the modal index on its rows, so ``B_frag`` uses
    ``leaf_tensors[frag][l]^T``. The diagonal generators are
    ``D_0 = sum_{l,p} (-Z0[l,l,p,p] / 2) Z_{lp}`` for the one-body fragment and
    ``D_frag = sum_{l>m} sum_{p,q} (Z[frag][l,m][p,q] / 4) Z_{lp} Z_{mq}`` for the two-body
    fragments, with wire index ``l * n_states + p``. The scalar identity part is basis
    independent and equals ``s = _energy_shift(ham)``. Unlike the identity-leaf case this is only
    reproduced by ``matrix(TrotterCGF)`` in the many-step limit (second-order Trotter error
    ``~ 1 / steps^2``).
    """
    from pennylane.templates.subroutines.time_evolution.trotter_cgf import (  # pylint: disable=import-outside-toplevel
        _energy_shift,
    )

    Z = np.asarray(ham.core_tensors, dtype=float)
    U = np.asarray(ham.leaf_tensors, dtype=float)
    num_modes = Z.shape[1]
    n_states = Z.shape[-1]
    n_wires = num_modes * n_states
    dim = 2**n_wires

    def wire(l, p):
        return l * n_states + p

    z_ops = [_single_z(w, n_wires) for w in range(n_wires)]
    H = _energy_shift(ham) * np.eye(dim, dtype=complex)

    B0 = _cgf_basis_rotation_matrix(U[0], n_states)
    D0 = np.zeros((dim, dim), dtype=complex)
    for l in range(num_modes):
        for p in range(n_states):
            D0 += (-Z[0][l, l, p, p] / 2) * z_ops[wire(l, p)]
    H += B0.conj().T @ D0 @ B0

    for frag in range(1, Z.shape[0]):
        Bl = _cgf_basis_rotation_matrix(np.swapaxes(U[frag], -2, -1), n_states)
        Dl = np.zeros((dim, dim), dtype=complex)
        for l in range(num_modes):
            for m in range(l):
                for p in range(n_states):
                    for q in range(n_states):
                        Dl += (Z[frag][l, m][p, q] / 4) * (z_ops[wire(l, p)] @ z_ops[wire(m, q)])
        H += Bl.conj().T @ Dl @ Bl
    return H


def toy_hamiltonian_cgf_generator(seed, abstract=False):
    """Synthetic CGF (vibrational) Hamiltonian on 2 modes x 2 modals with 1 two-body
    fragment (4 qubits, 16-dim space)."""
    rng = np.random.default_rng(seed)
    num_modes = 2
    n_states = 2

    eps = rng.normal(size=(num_modes, n_states)) * 0.4
    one_body_core_full = np.zeros((num_modes, num_modes, n_states, n_states))
    for l in range(num_modes):
        one_body_core_full[l, l] = np.diag(eps[l])
    one_body_leaf = np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])

    lam = rng.normal(size=(n_states, n_states)) * 0.35
    core_2b = np.zeros((1, num_modes, num_modes, n_states, n_states))
    core_2b[0, 1, 0] = lam
    leaf_2b = np.stack([np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])])

    core_tensors = np.concatenate([np.expand_dims(one_body_core_full, axis=0), core_2b], axis=0)
    leaf_tensors = np.concatenate([np.expand_dims(one_body_leaf, axis=0), leaf_2b], axis=0)
    nuc_constant = 0.7

    if abstract:
        core_tensors = Float[core_tensors.shape]
        leaf_tensors = Float[leaf_tensors.shape]
        nuc_constant = Float

    return core_tensors, leaf_tensors, nuc_constant, num_modes, n_states


@pytest.fixture
def toy_hamiltonian_cgf_concrete(seed):
    core_tensors, leaf_tensors, nuc_constant, num_modes, n_states = toy_hamiltonian_cgf_generator(
        seed
    )

    hamiltonian = CGFHamiltonian(
        core_tensors=core_tensors, leaf_tensors=leaf_tensors, nuc_constant=nuc_constant
    )

    return hamiltonian, num_modes, n_states


@pytest.fixture
def toy_hamiltonian_cgf_abstract(seed):
    core_tensors, leaf_tensors, nuc_constant, num_modes, n_states = toy_hamiltonian_cgf_generator(
        seed, abstract=True
    )

    abs_hamiltonian = CGFHamiltonian(
        core_tensors=core_tensors, leaf_tensors=leaf_tensors, nuc_constant=nuc_constant
    )

    return abs_hamiltonian, num_modes, n_states


@pytest.fixture
def diagonal_hamiltonian_cgf(seed):
    """CGF Hamiltonian with identity leaf tensors. All gates are diagonal (and thus
    commute), so the base circuit and both controlled constructions are Trotter-exact,
    enabling a machine-precision correctness check."""
    rng = np.random.default_rng(seed)
    num_modes = 2
    n_states = 3
    L = 2
    core = rng.normal(size=(L + 1, num_modes, num_modes, n_states, n_states)) * 0.4
    leaf = np.stack([np.stack([np.eye(n_states) for _ in range(num_modes)]) for _ in range(L + 1)])
    ham = CGFHamiltonian(core_tensors=core, leaf_tensors=leaf, nuc_constant=0.3)
    return ham, num_modes, n_states


class TestInitialization:
    """Test that TrotterCGF is initialized correctly."""

    def test_init_correctly(self, toy_hamiltonian_cgf_concrete):
        """Test that arguments and wires are stored correctly."""
        ham, num_modes, n_states = toy_hamiltonian_cgf_concrete
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

    def test_abstract_init(self, toy_hamiltonian_cgf_abstract):
        """Test that an abstract instance (e.g. for resource-rep purposes) is built."""
        from pennylane.typing import Float

        abs_ham, num_modes, n_states = toy_hamiltonian_cgf_abstract
        op = qp.TrotterCGF(Float, 5, abs_ham, Wire[num_modes * n_states])
        assert op.is_abstract

    def test_input_hamiltonian_type(self):
        """Test that anything but a CGFHamiltonian being given to the hamiltonian argument throws
        an error."""
        ham = [0.1, 0.2, 0.3, 0.4]
        match = (
            f"TrotterCGF expects a CGFHamiltonian for the hamiltonian argument. Got {type(ham)}."
        )

        with pytest.raises(ValueError, match=match):
            qp.TrotterCGF(evolution_time=0.1, num_trotter_steps=123, hamiltonian=ham, wires=(0, 1))


class TestValidity:
    """Basic structural validity tests for the TrotterCGF operator."""

    def test_assert_valid(self, toy_hamiltonian_cgf_concrete):
        """Run qp.ops.functions.assert_valid on a concrete CGF instance."""
        ham, num_modes, n_states = toy_hamiltonian_cgf_concrete
        wires = list(range(num_modes * n_states))
        op = qp.TrotterCGF(0.1, 3, ham, wires)
        # Differentiating through the (non-trainable) hamiltonian dict is not supported.
        qp.ops.functions.assert_valid(op, skip_differentiation=True)


class TestCGFScheme:
    """Unit tests for the leaf-handling helpers, which the identity-leaf definitional
    tests below do not exercise numerically (angles, pair enumeration, the energy shift,
    and both controlled structures are all covered by the ``*_matches_expm`` tests)."""

    def test_merge_leaves(self, seed):
        """Test the CGF per-mode merge rule ``U_curr @ U_prev^T`` that combines consecutive
        fragment rotations, so the ``leaf_prev`` un-rotation and ``leaf_curr`` rotation
        telescope into a single (transposed) BasisRotation per mode."""
        rng = np.random.default_rng(seed)
        num_modes, n_states = 2, 3
        U_prev = np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])
        U_curr = np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])
        expected = np.stack([U_curr[l] @ U_prev[l].T for l in range(num_modes)])
        assert np.allclose(_merge_leaves(U_prev, U_curr), expected)

    def test_align_one_body_leaf(self, seed):
        """The one-body leaf (eigenvectors stored as columns) is transposed per mode to match
        the two-body row convention; the two-body leaves are returned untouched."""
        from pennylane.templates.subroutines.time_evolution.trotter_cgf import (  # pylint: disable=import-outside-toplevel
            _align_one_body_leaf,
        )

        rng = np.random.default_rng(seed)
        num_modes, n_states, L = 2, 3, 2
        leaf = np.stack(
            [
                np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])
                for _ in range(L + 1)
            ]
        )

        ham = CGFHamiltonian(
            core_tensors=np.zeros((L + 1, num_modes, num_modes, n_states, n_states)),
            leaf_tensors=leaf,
            nuc_constant=0.0,
        )
        aligned = _align_one_body_leaf(ham).leaf_tensors
        assert np.allclose(aligned[0], np.swapaxes(leaf[0], -2, -1))
        assert np.allclose(aligned[1:], leaf[1:])

    def test_apply_system_basis_rotation(self, seed):
        """Test that per-mode leaves are applied as (transposed) BasisRotations and that a
        mode whose rotation is the identity is skipped."""
        num_modes, n_states = 2, 2
        wires = list(range(num_modes * n_states))
        U0 = random_orthogonal(n_states, np.random.default_rng(seed))
        U = np.stack([U0, np.eye(n_states)])
        tape = qp.tape.make_qscript(_apply_system_basis_rotation)(U, wires)
        assert [type(op) for op in tape.operations] == [qp.BasisRotation]
        assert list(tape.operations[0].wires) == wires[:n_states]
        assert np.allclose(tape.operations[0].parameters[0], U0.T)


class TestResourceRule:
    """Direct unit tests for the registered resource functions."""

    def test_num_trotter_steps_zero_has_no_resources(self, toy_hamiltonian_cgf_concrete):
        """Test that zero Trotter steps require zero resources."""
        ham, num_modes, n_states = toy_hamiltonian_cgf_concrete
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
    def test_controlled_zero_steps_empty_decomposition(self, toy_hamiltonian_cgf_concrete):
        """Test that C(TrotterCGF) with zero steps emits no gates."""
        ham, num_modes, n_states = toy_hamiltonian_cgf_concrete
        wires = list(range(num_modes * n_states))
        op = qp.TrotterCGF(1.0, 0, ham, wires=wires)
        assert qp.ctrl(op, control=[99]).decomposition() == []


class TestDecomposition:
    """Tests of the registered base decomposition rule."""

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_decomposition_self_consistent(self, toy_hamiltonian_cgf_concrete):
        """The registered base rule is self-consistent with its resource function."""
        ham, num_modes, n_states = toy_hamiltonian_cgf_concrete
        wires = list(range(num_modes * n_states))
        op = qp.TrotterCGF(0.4, 2, ham, wires)
        for rule in qp.list_decomps(qp.TrotterCGF):
            _test_decomposition_rule(op, rule)

    def test_base_matches_expm(self, diagonal_hamiltonian_cgf):
        """For an identity-leaf (Trotter-exact) Hamiltonian, matrix(TrotterCGF) equals the
        exact evolution expm(-i H t) of the Hamiltonian implied by the CGF definition. This
        is the definitional check on the Trotter angle prefactors (and the energy shift)."""
        ham, num_modes, n_states = diagonal_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        t, steps = 0.9, 3
        u = qp.matrix(qp.TrotterCGF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        expected = expm(-1j * cgf_reference_hamiltonian(ham) * t)
        assert np.allclose(u, expected, atol=1e-9)

    @pytest.mark.slow
    def test_base_matches_expm_nonidentity_leaves(self, seed):
        """With random real-orthogonal leaves the circuit is no longer Trotter-exact, but
        matrix(TrotterCGF) converges to the exact expm(-i H t) at second order (error
        ~ 1 / steps^2). Here H is built independently from the fragment basis rotations,
        H = sum_frag B_frag^dag D_frag B_frag, which pins down the (opposite) one-body /
        two-body leaf conventions and the consecutive-fragment merge that the identity-leaf
        checks do not exercise."""
        rng = np.random.default_rng(seed)
        num_modes, n_states, L = 2, 2, 2
        core = rng.normal(size=(L + 1, num_modes, num_modes, n_states, n_states)) * 0.4
        leaf = np.stack(
            [
                np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])
                for _ in range(L + 1)
            ]
        )
        ham = CGFHamiltonian(core_tensors=core, leaf_tensors=leaf, nuc_constant=0.3)
        sys_wires = list(range(num_modes * n_states))
        t = 0.6
        expected = expm(-1j * cgf_reference_hamiltonian_leaves(ham) * t)

        diffs = [
            float(
                np.linalg.norm(
                    qp.matrix(qp.TrotterCGF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
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
        """Negating an orbital line flips a leaf's determinant but leaves the projector
        ``|v><v|`` -- and hence the physical fragment -- unchanged (the orbital is on the
        columns of the one-body leaf and the rows of the two-body leaves). The template
        normalizes leaf determinants internally, so the circuit is invariant under this flip
        even when it yields mixed determinants (as an ``eigh`` one-body leaf with ``det = -1``
        next to ``expm`` two-body leaves would). Without the normalization ``BasisRotation``'s
        real-orthogonal sign gauge would realize a different Hamiltonian."""
        rng = np.random.default_rng(seed)
        num_modes, n_states, L = 2, 3, 2
        core = rng.normal(size=(L + 1, num_modes, num_modes, n_states, n_states)) * 0.4
        leaf = np.stack(
            [
                np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])
                for _ in range(L + 1)
            ]
        )
        ham = CGFHamiltonian(core_tensors=core, leaf_tensors=leaf, nuc_constant=0.3)

        flipped = leaf.copy()
        flipped[0][0][:, 0] *= -1.0  # one-body: negate an orbital column -> det flips (mixed)
        flipped[1][1][0, :] *= -1.0  # two-body: negate an orbital row -> det flips (mixed)
        assert np.linalg.det(leaf[0][0]) * np.linalg.det(flipped[0][0]) < 0
        assert np.linalg.det(leaf[1][1]) * np.linalg.det(flipped[1][1]) < 0
        ham_flipped = CGFHamiltonian(
            core_tensors=ham.core_tensors, leaf_tensors=flipped, nuc_constant=ham.nuc_constant
        )

        sys_wires = list(range(num_modes * n_states))
        t, steps = 0.6, 3
        u = qp.matrix(qp.TrotterCGF(t, steps, ham, wires=sys_wires), wire_order=sys_wires)
        u_flipped = qp.matrix(
            qp.TrotterCGF(t, steps, ham_flipped, wires=sys_wires), wire_order=sys_wires
        )
        assert np.allclose(u, u_flipped, atol=1e-12)

    def test_energy_shift_matches_literal(self):
        """The global phase equals ``exp(-i s t)`` with ``s = nuc + (1/2) sum eps`` the identity
        content of the CGF Hamiltonian, checked against a hard-coded literal computed here from
        the definition rather than by calling ``_energy_shift``. The ``RZ``/``IsingZZ`` layers are
        traceless generators (``det = 1``), so ``det(U) = exp(-i * dim * s * t)`` isolates ``s``
        independently of the circuit's basis-rotation content. Guards constant/energy-shift
        regressions that the (circular) identity-leaf check reuses ``_energy_shift`` for."""
        num_modes, n_states, L = 2, 2, 1
        core = np.zeros((L + 1, num_modes, num_modes, n_states, n_states))
        core[0, 0, 0] = np.diag([0.1, 0.3])  # eps of mode 0
        core[0, 1, 1] = np.diag([0.2, 0.4])  # eps of mode 1
        core[1, 1, 0] = np.array([[0.15, -0.05], [0.2, 0.1]])  # two-body: no effect on s
        leaf = np.stack(
            [np.stack([np.eye(n_states) for _ in range(num_modes)]) for _ in range(L + 1)]
        )
        ham = CGFHamiltonian(core_tensors=core, leaf_tensors=leaf, nuc_constant=0.2)
        # s = nuc + (1/2) sum_{l,p} eps^l_p = 0.2 + (0.1 + 0.3 + 0.2 + 0.4) / 2 = 0.7
        s_literal = 0.7
        wires = list(range(num_modes * n_states))
        dim = 2 ** len(wires)
        t = 0.3
        u = qp.matrix(qp.TrotterCGF(t, 2, ham, wires=wires), wire_order=wires)
        # det(expm(-i H t)) = exp(-i t Tr H) and Tr H = dim * s (Z/ZZ terms are traceless)
        assert np.isclose(np.linalg.det(u), np.exp(-1j * t * dim * s_literal), atol=1e-9)

    def test_preserves_unary_subspace(self, seed):
        """CGF encodes each mode's modal occupation in a unary (one-hot) register, so the circuit
        must never move amplitude out of the one-excitation-per-mode subspace: ``BasisRotation``
        conserves particle number within each mode block and the diagonal layers are
        occupation-preserving, giving exactly zero leakage."""
        rng = np.random.default_rng(seed)
        num_modes, n_states, L = 2, 3, 1
        core = rng.normal(size=(L + 1, num_modes, num_modes, n_states, n_states)) * 0.4
        leaf = np.stack(
            [
                np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])
                for _ in range(L + 1)
            ]
        )
        ham = CGFHamiltonian(core_tensors=core, leaf_tensors=leaf, nuc_constant=0.3)
        n_wires = num_modes * n_states
        wires = list(range(n_wires))
        u = qp.matrix(qp.TrotterCGF(0.5, 2, ham, wires=wires), wire_order=wires)
        # physical basis states: exactly one excitation per mode block (wire l*N + p, big-endian)
        physical = [
            sum(1 << (n_wires - 1 - (mode * n_states + occ[mode])) for mode in range(num_modes))
            for occ in itertools.product(range(n_states), repeat=num_modes)
        ]
        mask = np.zeros(2**n_wires, dtype=bool)
        mask[physical] = True
        # no amplitude leaves the physical subspace (and by unitarity none enters it either)
        assert np.allclose(u[np.ix_(~mask, physical)], 0.0, atol=1e-12)


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestControlledDecomposition:
    """Tests for the default (genuine) C(TrotterCGF) controlled decomposition."""

    def test_controlled_decomposition_self_consistent(self, toy_hamiltonian_cgf_concrete):
        """The registered C(TrotterCGF) rule is self-consistent with its resources."""
        ham, num_modes, n_states = toy_hamiltonian_cgf_concrete
        wires = list(range(num_modes * n_states))
        op = qp.ctrl(qp.TrotterCGF(0.4, 2, ham, wires), control=[99])
        for rule in qp.list_decomps("C(TrotterCGF)"):
            _test_decomposition_rule(op, rule)

    @pytest.mark.capture
    def test_controlled_decomposition_capture(self, toy_hamiltonian_cgf_concrete):
        """The C(TrotterCGF) rule captures cleanly. Operators can't be passed as capture inputs
        yet - they surface as ``ArgInfo`` wires (the "ArgInfo issue" tracked by ``test_Controlled``
        in ``tests/capture/test_operators.py``) - so the base is built inside the traced function.
        Once that is resolved this can use ``_test_decomposition_rule`` under capture directly."""
        ham, num_modes, n_states = toy_hamiltonian_cgf_concrete
        n = num_modes * n_states
        rule = qp.list_decomps("C(TrotterCGF)")[0]

        def circuit(t, *wires):
            with qp.capture.pause():  # the base itself should not be captured
                base = qp.TrotterCGF(t, 2, ham, wires=list(wires[:n]))
            rule(
                base=base,
                control_wires=list(wires[n:]),
                control_values=[1],
                work_wires=[],
                work_wire_type="borrowed",
            )

        with warnings.catch_warnings():  # no fall back to an unrolled Python loop
            warnings.simplefilter("error", CaptureWarning)
            jaxpr = jax.make_jaxpr(circuit)(jax.numpy.array(0.4), *range(n + 1))

        ops = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 0.4, *range(n + 1)).operations
        assert {type(op).__name__ for op in ops} == {"BasisRotation", "CNOT", "RZ", "PhaseShift"}

    def test_genuine_controlled_matches_expm(self, diagonal_hamiltonian_cgf):
        """By default ctrl(TrotterCGF) is a genuine controlled unitary: for an identity-leaf
        (Trotter-exact) Hamiltonian its matrix is exactly block_diag(I, expm(-i H t))."""
        ham, num_modes, n_states = diagonal_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        t, steps = 0.9, 3
        block0, block1 = control_branches(qp.TrotterCGF, ham, sys_wires, t, steps, False)
        dim = 2 ** len(sys_wires)
        expected = expm(-1j * cgf_reference_hamiltonian(ham) * t)
        assert np.allclose(block0, np.eye(dim), atol=1e-9)
        assert np.allclose(block1, expected, atol=1e-9)


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestDoublePhaseControlledDecomposition:
    """Tests for the opt-in double-phase (Fig. 6) C(TrotterCGF) controlled decomposition."""

    def test_controlled_decomposition_self_consistent(self, toy_hamiltonian_cgf_concrete):
        """The double-phase C(TrotterCGF) rule is self-consistent with its resources."""
        ham, num_modes, n_states = toy_hamiltonian_cgf_concrete
        wires = list(range(num_modes * n_states))
        op = qp.ctrl(qp.TrotterCGF(0.4, 2, ham, wires, double_phase=True), control=[99])
        for rule in qp.list_decomps("C(TrotterCGF)"):
            _test_decomposition_rule(op, rule)

    @pytest.mark.capture
    def test_controlled_decomposition_capture(self, toy_hamiltonian_cgf_concrete):
        """The double-phase C(TrotterCGF) rule captures cleanly. Operators can't be passed as
        capture inputs yet - they surface as ``ArgInfo`` wires (the "ArgInfo issue" tracked by
        ``test_Controlled`` in ``tests/capture/test_operators.py``) - so the base is built inside
        the traced function; once resolved this can use ``_test_decomposition_rule`` directly."""
        ham, num_modes, n_states = toy_hamiltonian_cgf_concrete
        n = num_modes * n_states
        rule = qp.list_decomps("C(TrotterCGF)")[0]

        def circuit(t, *wires):
            with qp.capture.pause():  # the base itself should not be captured
                base = qp.TrotterCGF(t, 2, ham, wires=list(wires[:n]), double_phase=True)
            rule(
                base=base,
                control_wires=list(wires[n:]),
                control_values=[1],
                work_wires=[],
                work_wire_type="borrowed",
            )

        with warnings.catch_warnings():  # no fall back to an unrolled Python loop
            warnings.simplefilter("error", CaptureWarning)
            jaxpr = jax.make_jaxpr(circuit)(jax.numpy.array(0.4), *range(n + 1))

        ops = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 0.4, *range(n + 1)).operations
        assert {type(op).__name__ for op in ops} == {"BasisRotation", "CNOT", "RZ", "IsingZZ"}

    def test_double_phase_controlled_matches_expm(self, diagonal_hamiltonian_cgf):
        """With double_phase=True, ctrl(TrotterCGF) realizes diag(U, U^dagger): for an
        identity-leaf (Trotter-exact) Hamiltonian the control-0 / control-1 blocks are
        exactly expm(-i H t) / expm(+i H t) (explicit global phases, no per-block phase)."""
        ham, num_modes, n_states = diagonal_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        t, steps = 0.9, 3
        block0, block1 = control_branches(qp.TrotterCGF, ham, sys_wires, t, steps, True)
        u = expm(-1j * cgf_reference_hamiltonian(ham) * t)
        assert np.allclose(block0, u, atol=1e-9)
        assert np.allclose(block1, u.conj().T, atol=1e-9)

    def test_double_phase_hadamard_invariant(self, diagonal_hamiltonian_cgf, seed):
        """The double-phase Hadamard test measures <X> = Re<psi|block0^dag block1|psi>, which
        for the exact blocks equals Re<psi|expm(+2 i H t)|psi>."""
        ham, num_modes, n_states = diagonal_hamiltonian_cgf
        sys_wires = list(range(num_modes * n_states))
        t, steps = 0.9, 3
        measured, psi = hadamard_test(qp.TrotterCGF, ham, sys_wires, t, steps, True, seed)
        block0, block1 = control_branches(qp.TrotterCGF, ham, sys_wires, t, steps, True)
        ref_blocks = float(np.real(psi.conj() @ (block0.conj().T @ block1 @ psi)))
        ref_expm = float(
            np.real(psi.conj() @ (expm(2j * cgf_reference_hamiltonian(ham) * t) @ psi))
        )
        assert np.isclose(measured, ref_blocks, atol=1e-9)
        assert np.isclose(measured, ref_expm, atol=1e-9)


class TestIntegration:
    """Integration tests via the graph-based decomposition system."""

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("t, num_steps", [(1.0, 0), (0.0, 10)])
    def test_identity_edge_cases(self, toy_hamiltonian_cgf_concrete, t, num_steps):
        """Test that zero Trotter steps, or zero evolution time, produce the identity."""
        ham, num_modes, n_states = toy_hamiltonian_cgf_concrete
        wires = list(range(num_modes * n_states))

        def _circuit():
            qp.TrotterCGF(t, num_steps, ham, wires)

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
        L = 2
        M = 2
        N = 2
        rng = np.random.default_rng(seed)
        hamiltonian = CGFHamiltonian(
            core_tensors=rng.random((L, M, M, N, N)),
            leaf_tensors=rng.random((L, M, N, N)),
            nuc_constant=0.5,
        )
        registers = qp.registers({"hadamard": 1, "system": M * N})

        @qp.qjit
        @qp.transforms.decompose(gate_set=target_gates)
        @qp.qnode(qp.device("lightning.qubit"))
        def trotter_circuit():
            qp.H(registers["hadamard"])
            qp.ctrl(
                qp.TrotterCGF(
                    1.0, 10, hamiltonian, wires=registers["system"], double_phase=double_phase
                ),
                control=registers["hadamard"],
            )
            return qp.expval(qp.X(registers["hadamard"]))

        assert not np.isclose(trotter_circuit(), 0)

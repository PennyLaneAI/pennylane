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
import numpy as np
import pytest
from scipy.linalg import expm

jax = pytest.importorskip("jax")

import pennylane as qp
from pennylane.decomposition.resources import Resources
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.time_evolution.trotter_cgf import (
    _apply_system_basis_rotation,
    _merge_leaves,
)
from pennylane.typing import Wire
from pennylane.wires import Wires
from tests.templates.subroutines.time_evolution.trotter_test_helpers import (  # pylint: disable=no-name-in-module
    CATALYST_GATE_SET_DOUBLE_PHASE,
    CATALYST_GATE_SET_GENUINE,
    cgf_reference_hamiltonian,
    control_branches,
    hadamard_test,
    random_orthogonal,
)

pytestmark = pytest.mark.jax


@pytest.fixture
def toy_hamiltonian_cgf(seed):
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
    hamiltonian = {
        "core_tensors": core_tensors,
        "leaf_tensors": leaf_tensors,
        "nuc_constant": 0.7,
    }
    return hamiltonian, num_modes, n_states


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
    """Unit tests for the leaf-handling helpers, which the identity-leaf definitional
    tests below do not exercise numerically (angles, pair enumeration, the energy shift,
    and both controlled structures are all covered by the ``*_matches_expm`` tests)."""

    def test_merge_leaves(self, seed):
        """Test the CGF per-mode merge rule that combines consecutive fragment rotations."""
        rng = np.random.default_rng(seed)
        num_modes, n_states = 2, 3
        U_prev = np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])
        U_curr = np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])
        expected = np.stack([U_prev[l].T @ U_curr[l] for l in range(num_modes)])
        assert np.allclose(_merge_leaves(U_prev, U_curr), expected)

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


class TestDecomposition:
    """Tests of the registered base decomposition rule."""

    def test_decomposition_self_consistent(self, toy_hamiltonian_cgf):
        """The registered base rule is self-consistent with its resource function."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
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

    def test_controlled_decomposition_self_consistent(self, toy_hamiltonian_cgf):
        """The double-phase C(TrotterCGF) rule is self-consistent with its resources."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
        wires = list(range(num_modes * n_states))
        op = qp.ctrl(qp.TrotterCGF(0.4, 2, ham, wires, double_phase=True), control=[99])
        for rule in qp.list_decomps("C(TrotterCGF)"):
            _test_decomposition_rule(op, rule)

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
    def test_identity_edge_cases(self, toy_hamiltonian_cgf, t, num_steps):
        """Test that zero Trotter steps, or zero evolution time, produce the identity."""
        ham, num_modes, n_states = toy_hamiltonian_cgf
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
        hamiltonian = {
            "core_tensors": rng.random((L, M, M, N, N)),
            "leaf_tensors": rng.random((L, M, N, N)),
            "nuc_constant": 0.5,
        }
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

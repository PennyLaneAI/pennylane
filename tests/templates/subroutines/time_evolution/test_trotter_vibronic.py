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
"""Tests for the ``TrotterVibronic`` template."""

from itertools import combinations_with_replacement

import numpy as np
import pytest

import pennylane as qp
from pennylane.templates.subroutines import AQFT, QROM, SemiAdder
from pennylane.templates.subroutines.arithmetic import (
    OutMultiplier,
    SignedOutMultiplier,
    SignedOutSquare,
)
from pennylane.templates.subroutines.time_evolution._trotter_vibronic_utils import (
    _derive_diag_keys,
    _diagonalization_matrix,
    _diagonalize_vibronic_circuit,
    _half_signed_out_multiplier,
    _momentum_coefficients,
    _position_coefficients,
)

vibronic_fragments = pytest.importorskip(
    "pennylane.labs.trotter_error.fragments"
).vibronic_fragments


# ---------------------------------------------------------------------------
# --------------------------- Test data helpers -----------------------------
# ---------------------------------------------------------------------------


def fragment_to_dense(fragment, op_type):
    """Convert the ``op_type`` coefficients of a ``RealspaceMatrix`` fragment to a dense array."""
    n_states, n_modes, order = fragment.states, fragment.modes, len(op_type)
    dense = np.zeros((n_states, n_states) + (n_modes,) * order)
    for elec_key, val in fragment.get_coefficients().items():
        terms = val.get(op_type, None)
        if terms is None:
            continue
        if order == 0:
            dense[elec_key] = terms.get((), 0.0)
            continue
        for modes in combinations_with_replacement(range(n_modes), r=order):
            dense[elec_key][modes] = terms.get(modes, 0.0)
    return dense


def build_hamiltonian(fragments):
    """Build the dense vibronic Hamiltonian dictionary from a list of fragments."""
    position, kinetic = fragments[:-1], fragments[-1]
    return {
        "constant": np.stack([fragment_to_dense(f, ()) for f in position]),
        "linear": np.stack([fragment_to_dense(f, ("Q",)) for f in position]),
        "quadratic": np.stack([fragment_to_dense(f, ("Q", "Q")) for f in position]),
        "kinetic": fragment_to_dense(kinetic, ("P", "P")),
    }


def fragment_list(n_states=2, n_modes=2, seed=42):
    """Create a list of vibronic fragments with random coefficients."""
    rng = np.random.default_rng(seed)
    freqs = rng.random(n_modes)
    coeffs = [rng.random((n_states, n_states)), rng.random((n_states, n_states, n_modes))]
    return vibronic_fragments(n_states, n_modes, freqs, coeffs)


def make_wires(n_states, n_modes, k=2, b=3):
    """Create the wire registers required by :class:`~.TrotterVibronic`."""
    n = int(qp.math.ceil_log2(n_states))
    sizes = {
        "electronic": n,
        "vib_wires": n_modes * k,
        "cache": 2 * k,
        "coefficients": b,
        "phase_gradient": b,
        "work": max(n - 1, 2 * k, 2 * b + 2),
    }
    return qp.registers(sizes)


def make_op(hamiltonian, wires, evolution_time=0.52, num_trotter_steps=1, aqft_order=1, **kwargs):
    """Construct a :class:`~.TrotterVibronic` operator from a Hamiltonian and wire registers."""
    return qp.TrotterVibronic(
        evolution_time=evolution_time,
        num_trotter_steps=num_trotter_steps,
        hamiltonian=hamiltonian,
        electronic=wires["electronic"],
        vib_wires=wires["vib_wires"],
        cache=wires["cache"],
        coefficients=wires["coefficients"],
        phase_gradient=wires["phase_gradient"],
        work=wires["work"],
        aqft_order=aqft_order,
        **kwargs,
    )


def decomposition_queue(op):
    """Return the queued operations of an operator's decomposition."""
    with qp.queuing.AnnotatedQueue() as q:
        op.compute_decomposition(**op.arguments)
    return q.queue


def count_ops(queue, *op_types):
    return sum(isinstance(op, op_types) for op in queue)


# ---------------------------------------------------------------------------
# --------------------------------- Helpers ---------------------------------
# ---------------------------------------------------------------------------


class TestDiagonalization:
    """Tests for the electronic-diagonalization helpers."""

    @pytest.mark.parametrize(
        "n_states, key, support",
        [(2, (0, 1), [0]), (4, (0, 3), [1, 0]), (8, (0, 7), [2, 1, 0])],
    )
    def test_expected_circuit(self, n_states, key, support):
        """Test that the correct Clifford circuit is queued."""
        wires = list(range(int(qp.math.ceil_log2(n_states))))
        with qp.queuing.AnnotatedQueue() as q:
            _diagonalize_vibronic_circuit(key=key, wires=wires)
        control = wires[support[0]]
        expected = [qp.Hadamard(control)] + [qp.CNOT([control, wires[i]]) for i in support[1:]]
        assert q.queue == expected

    def test_diagonal_key_is_identity(self):
        """Test that a diagonal key queues no operations and yields the identity matrix."""
        with qp.queuing.AnnotatedQueue() as q:
            _diagonalize_vibronic_circuit(key=(1, 1), wires=[0, 1])
        assert q.queue == []
        assert np.allclose(_diagonalization_matrix((1, 1), 2), np.eye(4))

    @pytest.mark.parametrize("n, key", [(1, (0, 1)), (2, (0, 3)), (2, (1, 2)), (3, (0, 7))])
    def test_matrix_matches_circuit(self, n, key):
        """Test that the dense diagonalization matrix matches the circuit unitary."""
        wires = list(range(n))
        circuit_matrix = qp.matrix(_diagonalize_vibronic_circuit, wires)(key=key, wires=wires)
        assert np.allclose(_diagonalization_matrix(key, n), circuit_matrix)

    def test_matrix_is_orthogonal(self):
        """Test that the diagonalization matrix is orthogonal."""
        matrix = _diagonalization_matrix((0, 3), 2)
        assert np.allclose(matrix @ matrix.T, np.eye(4))


class TestCoefficientReadout:
    """Tests for the dense coefficient-extraction helpers."""

    def test_position_coefficients(self, seed):
        """Test position coefficient extraction against a direct reference computation."""
        fragments = fragment_list(n_states=3, n_modes=2, seed=seed)
        hamiltonian = build_hamiltonian(fragments)
        diag_keys = _derive_diag_keys(hamiltonian)

        n_states, n_modes = 3, 2
        n = int(qp.math.ceil_log2(n_states))
        matrix = _diagonalization_matrix(diag_keys[0], n)[:n_states, :n_states]
        constant, linear, quadratic, bilinear = _position_coefficients(
            matrix,
            hamiltonian["constant"][0],
            hamiltonian["linear"][0],
            hamiltonian["quadratic"][0],
            n_states,
            n_modes,
        )
        assert constant.shape == (n_states,)
        assert linear.shape == (n_modes, n_states)
        assert quadratic.shape == (n_modes, n_states)
        assert bilinear.shape == (n_modes * (n_modes - 1) // 2, n_states)

        # Reference: constant diagonal of M^T C M
        ref_constant = np.diag(matrix.T @ hamiltonian["constant"][0] @ matrix)
        assert np.allclose(constant, ref_constant)

    def test_momentum_coefficients(self, seed):
        """Test momentum coefficient extraction against the injected diagonal values."""
        n_states, n_modes = 2, 3
        rng = np.random.default_rng(seed)
        p_quad = rng.random(n_modes)
        kinetic = np.einsum("ab,cd->abcd", np.eye(n_states), np.diag(p_quad))
        assert np.allclose(_momentum_coefficients(kinetic, n_modes), p_quad)


@pytest.mark.parametrize(
    "x, y, z, expected",
    [(3, -3, 5, (5 + 3 * (-3)) % 64), (2, 3, 1, (1 + 2 * 3) % 64), (4, -1, 0, (-4) % 64)],
)
def test_half_signed_out_multiplier(x, y, z, expected):
    """Test that the half-signed multiplier computes ``(z + x*y) mod 2^k`` for a signed ``y``."""
    x_wires, y_wires = [0, 1, 2], [3, 4, 5]
    output_wires, work_wires = [6, 7, 8, 9, 10, 11], [12, 13, 14, 15]
    dev = qp.device("default.qubit", wires=16)

    @qp.qnode(dev)
    def circuit():
        qp.BasisEmbedding(x, wires=x_wires)
        qp.BasisEmbedding(y % (2 ** len(y_wires)), wires=y_wires)
        qp.BasisEmbedding(z, wires=output_wires)
        _half_signed_out_multiplier(x_wires, y_wires, output_wires, work_wires)
        return qp.probs(wires=output_wires)

    # the circuit permutes basis states, so the output is a single basis state
    assert int(np.argmax(circuit())) == expected


def test_derive_diag_keys():
    """Test that diagonal and off-diagonal fragments yield the expected diagonalization keys."""
    n_states, n_modes = 2, 2
    # fragment 0: diagonal (harmonic); fragment 1: off-diagonal coupling
    constant = np.zeros((2, n_states, n_states))
    constant[0, 0, 0] = 1.0
    linear = np.zeros((2, n_states, n_states, n_modes))
    linear[1, 0, 1, 0] = 0.5
    linear[1, 1, 0, 0] = 0.5
    quadratic = np.zeros((2, n_states, n_states, n_modes, n_modes))
    kinetic = np.einsum("ab,cd->abcd", np.eye(n_states), np.diag(np.ones(n_modes)))
    hamiltonian = {
        "constant": constant,
        "linear": linear,
        "quadratic": quadratic,
        "kinetic": kinetic,
    }
    assert _derive_diag_keys(hamiltonian) == ((0, 0), (0, 1))


# ---------------------------------------------------------------------------
# ------------------------------ Construction -------------------------------
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for the construction and validation of ``TrotterVibronic``."""

    def test_basic_construction(self):
        """Test basic construction, argument categorization and wires."""
        fragments = fragment_list()
        hamiltonian = build_hamiltonian(fragments)
        wires = make_wires(2, 2)
        op = make_op(hamiltonian, wires)
        assert op.name == "TrotterVibronic"
        assert op.arguments["num_trotter_steps"] == 1
        assert "hamiltonian" in op.hybrid_args
        assert "evolution_time" in op.dynamic_args
        # every register wire is part of op.wires
        all_wires = set()
        for reg in wires.values():
            all_wires |= set(reg)
        assert set(op.wires) == all_wires

    def test_diag_keys_derived(self):
        """Test that diag_keys are derived from the Hamiltonian when not provided."""
        fragments = fragment_list()
        hamiltonian = build_hamiltonian(fragments)
        op = make_op(hamiltonian, make_wires(2, 2))
        assert op.arguments["diag_keys"] == _derive_diag_keys(hamiltonian)

    def test_diag_keys_explicit(self):
        """Test that explicit diag_keys are respected."""
        fragments = fragment_list()
        hamiltonian = build_hamiltonian(fragments)
        diag_keys = ((0, 0), (0, 0))
        op = make_op(hamiltonian, make_wires(2, 2), diag_keys=diag_keys)
        assert op.arguments["diag_keys"] == diag_keys

    @pytest.mark.parametrize(
        "hamiltonian, match",
        [
            ("not a dict", "dictionary"),
            ({}, "keys in `hamiltonian`"),
            ({"constant": 1, "linear": 1, "quadratic": 1, "extra": 1}, "keys in `hamiltonian`"),
        ],
    )
    def test_rejects_bad_hamiltonian_container(self, hamiltonian, match):
        """Test validation of the Hamiltonian container."""
        with pytest.raises(ValueError, match=match):
            make_op(hamiltonian, make_wires(2, 2))

    def test_rejects_bad_hamiltonian_ndim(self):
        """Test validation of the Hamiltonian tensor dimensions."""
        hamiltonian = build_hamiltonian(fragment_list())
        hamiltonian["constant"] = hamiltonian["constant"][0]  # drop the fragment axis
        with pytest.raises(ValueError, match="3-dimensional"):
            make_op(hamiltonian, make_wires(2, 2))

    @pytest.mark.parametrize("num_steps", [0, -1, 1.5])
    def test_rejects_invalid_num_trotter_steps(self, num_steps):
        """Test that an invalid number of Trotter steps raises an error."""
        hamiltonian = build_hamiltonian(fragment_list())
        with pytest.raises(ValueError, match="positive integer"):
            make_op(hamiltonian, make_wires(2, 2), num_trotter_steps=num_steps)

    @pytest.mark.parametrize(
        "register, match",
        [
            ("electronic", "electronic states"),
            ("cache", "cache qubits"),
            ("phase_gradient", "phase-gradient"),
            ("work", "work qubits"),
        ],
    )
    def test_rejects_bad_register_sizes(self, register, match):
        """Test that wrongly-sized registers raise an error during decomposition."""
        hamiltonian = build_hamiltonian(fragment_list(n_states=4, n_modes=2))
        wires = make_wires(4, 2)
        wires[register] = wires[register][:-1]
        op = make_op(hamiltonian, wires)
        with pytest.raises(ValueError, match=match):
            op.compute_decomposition(**op.arguments)


# ---------------------------------------------------------------------------
# ------------------------------ Decomposition ------------------------------
# ---------------------------------------------------------------------------


class TestDecomposition:
    """Tests for the decomposition of ``TrotterVibronic``."""

    def test_zero_time_skips_arithmetic(self, seed):
        """Test that arithmetic is skipped when the evolution time is zero."""
        hamiltonian = build_hamiltonian(fragment_list(n_modes=3, seed=seed))
        op = make_op(hamiltonian, make_wires(2, 3), evolution_time=0.0)
        queue = decomposition_queue(op)
        arithmetic = (SemiAdder, OutMultiplier, SignedOutSquare, SignedOutMultiplier, AQFT)
        assert count_ops(queue, *arithmetic) == 0
        assert all(isinstance(o, (qp.Hadamard, qp.CNOT, QROM)) for o in queue)

    @pytest.mark.parametrize("num_steps", [1, 2, 3])
    def test_num_steps_scales(self, num_steps, seed):
        """Test that the number of queued operations scales linearly with the Trotter steps."""
        hamiltonian = build_hamiltonian(fragment_list(seed=seed))
        wires = make_wires(2, 2)
        one = decomposition_queue(make_op(hamiltonian, wires, num_trotter_steps=1))
        many = decomposition_queue(make_op(hamiltonian, wires, num_trotter_steps=num_steps))
        assert len(many) == num_steps * len(one)

    def test_constant_term_queues_semiadder(self, seed):
        """Test that a constant position term queues a SemiAdder."""
        fragments = [
            _single_fragment(2, 2, [()], seed),
            _single_fragment(2, 2, [("P", "P")], seed + 1),
        ]
        hamiltonian = build_hamiltonian(fragments)
        queue = decomposition_queue(make_op(hamiltonian, make_wires(2, 2), evolution_time=0.1))
        assert count_ops(queue, SemiAdder) > 0

    def test_kinetic_queues_aqft_and_basisstate(self, seed):
        """Test that BasisState and AQFT appear only at non-zero evolution time."""
        hamiltonian = build_hamiltonian(fragment_list(seed=seed))
        wires = make_wires(2, 2)
        zero = decomposition_queue(make_op(hamiltonian, wires, evolution_time=0.0))
        assert count_ops(zero, AQFT) == 0
        nonzero = decomposition_queue(make_op(hamiltonian, wires, evolution_time=0.1))
        assert count_ops(nonzero, AQFT) > 0
        assert count_ops(nonzero, qp.BasisState) > 0

    def test_bilinear_queues_signed_out_multiplier(self):
        """Test that a bilinear fragment queues a SignedOutMultiplier."""
        fragments = [
            _single_fragment(2, 3, [("Q", "Q")], seed=0, skip_quadratic=True),
            _single_fragment(2, 3, [("P", "P")], seed=1),
        ]
        hamiltonian = build_hamiltonian(fragments)
        queue = decomposition_queue(make_op(hamiltonian, make_wires(2, 3), evolution_time=0.1))
        assert count_ops(queue, SignedOutMultiplier) > 0


# ---------------------------------------------------------------------------
# ------------------------------- Execution ---------------------------------
# ---------------------------------------------------------------------------


def test_default_qubit_execution():
    """Test that a small vibronic Trotter circuit executes on default.qubit."""
    n_states, n_modes, k, b = 2, 1, 2, 2
    n = int(qp.math.ceil_log2(n_states))
    hamiltonian = {
        "constant": np.zeros((1, n_states, n_states)),
        "linear": np.zeros((1, n_states, n_states, n_modes)),
        "quadratic": np.zeros((1, n_states, n_states, n_modes, n_modes)),
        "kinetic": np.einsum("ab,cd->abcd", np.eye(n_states), np.diag(0.3 * np.ones(n_modes))),
    }
    wires = qp.registers(
        {
            "electronic": n,
            "vib_wires": n_modes * k,
            "cache": 2 * k,
            "coefficients": b,
            "phase_gradient": b,
            "work": max(n - 1, 2 * k, 2 * b + 2),
        }
    )
    num_wires = max(w for reg in wires.values() for w in reg) + 1
    dev = qp.device("default.qubit", wires=num_wires)

    @qp.qnode(dev)
    def circuit():
        make_op(hamiltonian, wires, evolution_time=1.0, num_trotter_steps=1)
        return qp.state()

    state = circuit()
    assert np.all(np.isfinite(state))
    assert np.isclose(np.linalg.norm(state), 1.0)


# ---------------------------------------------------------------------------
# --------------------------- Extra data helpers ----------------------------
# ---------------------------------------------------------------------------


def _single_fragment(n_states, n_modes, include_op_types, seed=0, skip_quadratic=False):
    """Build a single random position or kinetic fragment for targeted tests."""
    # pylint: disable=import-outside-toplevel
    from pennylane.labs.trotter_error.realspace import (
        RealspaceCoeffs,
        RealspaceMatrix,
        RealspaceOperator,
        RealspaceSum,
    )

    rng = np.random.default_rng(seed)

    if include_op_types == [("P", "P")]:
        op = RealspaceOperator(
            n_modes, ("P", "P"), RealspaceCoeffs(np.diag(rng.random(n_modes)), "label")
        )
        blocks = {(i, i): RealspaceSum(n_modes, [op]) for i in range(n_states)}
        return RealspaceMatrix(n_states, n_modes, blocks)

    blocks = {}
    m = rng.integers(0, n_states)
    elec_ids = [(i, int(i ^ m)) for i in range(n_states) if i ^ m < n_states]
    for elec_idx in elec_ids:
        ops = []
        for op_type in include_op_types:
            tensor = rng.random((n_modes,) * len(op_type))
            if len(op_type) == 2:
                diagonal = 0 if skip_quadratic else -1
                tensor[np.tril_indices(n_modes, k=diagonal)] = 0.0
            ops.append(RealspaceOperator(n_modes, op_type, RealspaceCoeffs(tensor, "label")))
        blocks[elec_idx] = blocks[elec_idx[::-1]] = RealspaceSum(n_modes, ops)
    return RealspaceMatrix(n_states, n_modes, blocks)

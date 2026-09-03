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

import numpy as np
import pytest

import pennylane as qp
from pennylane.allocation import Allocate, Deallocate
from pennylane.templates.subroutines import AQFT, QROM, SemiAdder
from pennylane.templates.subroutines.arithmetic import (
    OutMultiplier,
    SignedOutMultiplier,
    SignedOutSquare,
)
from pennylane.templates.subroutines.time_evolution.trotter_vibronic import (
    _diagonalization_matrix,
    _diagonalize_vibronic_circuit,
    _half_signed_out_multiplier,
    _momentum_coefficients,
    _position_coefficients,
    _validate_registers,
    _wires_are_concrete,
)
from pennylane.typing import AbstractWires

# ---------------------------------------------------------------------------
# --------------------------- Test data helpers -----------------------------
# ---------------------------------------------------------------------------
#
# Dense vibronic Hamiltonians in the "XOR" fragmentation format expected by ``TrotterVibronic``.


def _next_pow_2(k):
    """Return the smallest power of 2 greater than or equal to ``k``."""
    return 2 ** (k - 1).bit_length()


def _zero_fragment(n_states, n_modes):
    """Return a fragment of all-zero dense coefficient tensors."""
    return {
        "constant": np.zeros((n_states, n_states)),
        "linear": np.zeros((n_states, n_states, n_modes)),
        "quadratic": np.zeros((n_states, n_states, n_modes, n_modes)),
        "kinetic": np.zeros((n_states, n_states, n_modes, n_modes)),
    }


def build_hamiltonian(fragments):
    """Build the dense vibronic Hamiltonian dictionary from a list of dense fragments.

    The leading ``fragments[:-1]`` are the position fragments (stacked along a new leading
    fragment axis) and the last entry contributes the single kinetic fragment.
    """
    position, kinetic = fragments[:-1], fragments[-1]
    return {
        "constant": np.stack([f["constant"] for f in position]),
        "linear": np.stack([f["linear"] for f in position]),
        "quadratic": np.stack([f["quadratic"] for f in position]),
        "kinetic": kinetic["kinetic"],
    }


def fragment_list(n_states=2, n_modes=2, seed=42):
    """Create the position + kinetic fragments of a random vibronic Hamiltonian.

    Reproduces the "blocks" fragmentation scheme: the harmonic frequencies and the constant and
    linear Taylor coefficients are drawn from a seeded generator, position fragment ``i`` collects
    the ``(j, i ^ j)`` electronic blocks, and the diagonal blocks additionally carry the harmonic
    ``diag(freqs) / 2`` quadratic term (which also forms the kinetic fragment).
    """
    rng = np.random.default_rng(seed)
    freqs = rng.random(n_modes)
    # A physical (Hermitian, here real-valued) Hamiltonian must be symmetric in the electronic
    # indices, which is what the Clifford diagonalization below relies on.
    constant_coeffs = rng.random((n_states, n_states))
    constant_coeffs = (constant_coeffs + constant_coeffs.T) / 2
    linear_coeffs = rng.random((n_states, n_states, n_modes))
    linear_coeffs = (linear_coeffs + linear_coeffs.transpose(1, 0, 2)) / 2

    num_fragments = _next_pow_2(n_states)
    harmonic = np.diag(freqs) / 2

    fragments = []
    for i in range(num_fragments):
        fragment = _zero_fragment(n_states, n_modes)
        for j in range(num_fragments):
            col = i ^ j
            if j >= n_states or col >= n_states:
                continue
            fragment["constant"][j, col] = constant_coeffs[j, col]
            fragment["linear"][j, col] = linear_coeffs[j, col]
            if j == col:
                fragment["quadratic"][j, j] = harmonic
        fragments.append(fragment)

    kinetic = _zero_fragment(n_states, n_modes)
    for i in range(n_states):
        kinetic["kinetic"][i, i] = harmonic
    fragments.append(kinetic)
    return fragments


def make_wires(n_states, n_modes, k=3, b=3):
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


def make_op(
    hamiltonian, wires, evolution_time=0.52, num_trotter_steps=1, aqft_order=None, **kwargs
):
    """Construct a :class:`~.TrotterVibronic` operator from a Hamiltonian and wire registers."""
    if aqft_order is None and isinstance(hamiltonian, dict) and "linear" in hamiltonian:
        linear = hamiltonian["linear"]
        if hasattr(linear, "shape"):
            n_modes = linear.shape[-1]
            k = len(wires["vib_wires"]) // n_modes
            # ``None`` resolves to ``k - 1`` in the template, which triggers AQFT's QFT-equivalence
            # warning; use a genuine approximate order on the usual ``k >= 3`` test grids.
            if k >= 3:
                aqft_order = 1
    return qp.TrotterVibronic(
        evolution_time=evolution_time,
        num_trotter_steps=num_trotter_steps,
        hamiltonian=hamiltonian,
        electronic_wires=wires["electronic"],
        vib_wires=wires["vib_wires"],
        cache_wires=wires["cache"],
        coefficient_wires=wires["coefficients"],
        phase_gradient_wires=wires["phase_gradient"],
        work_wires=wires["work"],
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


def count_allocations(op):
    """Return the ``(num_allocate, num_deallocate)`` operations in ``op``'s decomposition.

    ``TrotterVibronic`` allocates its dynamic wires in a single top-level ``with allocate(...)``
    block, so under capture (where a jaxpr containing dynamic qubit allocation cannot be eagerly
    executed) the ``allocate``/``deallocate`` primitives are counted directly in the traced
    jaxpr's top-level equations instead of in the (executed) queue.
    """
    if qp.capture.enabled():
        import jax  # pylint: disable=import-outside-toplevel

        from pennylane.allocation import (  # pylint: disable=import-outside-toplevel
            allocate_prim,
            deallocate_prim,
        )

        decomposition = qp.list_decomps(qp.TrotterVibronic)[0]
        jaxpr = jax.make_jaxpr(lambda: decomposition(**op.arguments))().jaxpr
        return (
            sum(eqn.primitive == allocate_prim for eqn in jaxpr.eqns),
            sum(eqn.primitive == deallocate_prim for eqn in jaxpr.eqns),
        )
    queue = decomposition_queue(op)
    return count_ops(queue, Allocate), count_ops(queue, Deallocate)


# ---------------------------------------------------------------------------
# --------------------------------- Helpers ---------------------------------
# ---------------------------------------------------------------------------


class TestDiagonalization:
    """Tests for the electronic-diagonalization helpers."""

    @pytest.mark.parametrize(
        "n_states, fragment_idx, support",
        [(2, 1, [0]), (4, 3, [1, 0]), (8, 7, [2, 1, 0])],
    )
    def test_expected_circuit(self, n_states, fragment_idx, support):
        """Test that the correct Clifford circuit is queued for the ``(0, fragment_idx)`` key."""
        wires = list(range(int(qp.math.ceil_log2(n_states))))
        with qp.queuing.AnnotatedQueue() as q:
            _diagonalize_vibronic_circuit(fragment_idx=fragment_idx, wires=wires)
        control = wires[support[0]]
        expected = [qp.Hadamard(control)] + [qp.CNOT([control, wires[i]]) for i in support[1:]]
        assert q.queue == expected

    def test_zero_fragment_idx_is_identity(self):
        """Test that fragment index 0 (key ``(0, 0)``) queues no operations and yields identity."""
        with qp.queuing.AnnotatedQueue() as q:
            _diagonalize_vibronic_circuit(fragment_idx=0, wires=[0, 1])
        assert q.queue == []
        assert np.allclose(_diagonalization_matrix((0, 0), 2), np.eye(4))

    @pytest.mark.parametrize("n, fragment_idx", [(1, 1), (2, 3), (3, 7)])
    def test_matrix_matches_circuit(self, n, fragment_idx):
        """Test that the dense diagonalization matrix matches the circuit unitary."""
        wires = list(range(n))
        circuit_matrix = qp.matrix(_diagonalize_vibronic_circuit, wires)(
            fragment_idx=fragment_idx, wires=wires
        )
        assert np.allclose(_diagonalization_matrix((0, fragment_idx), n), circuit_matrix)

    def test_matrix_is_orthogonal(self):
        """Test that the diagonalization matrix is orthogonal."""
        matrix = _diagonalization_matrix((0, 3), 2)
        assert np.allclose(matrix @ matrix.T, np.eye(4))


class TestCoefficientReadout:
    """Tests for the dense coefficient-extraction helpers."""

    def test_position_coefficients(self, seed):
        """Test position coefficient extraction against a direct reference computation.

        Uses fragment 1 of the blocks scheme, which is genuinely off-diagonal.
        """
        fragments = fragment_list(n_states=4, n_modes=2, seed=seed)
        hamiltonian = build_hamiltonian(fragments)

        n_states, n_modes = 4, 2
        n = int(qp.math.ceil_log2(n_states))
        # Fragment 1 is diagonalized with key ``(0, 1)``.
        matrix = _diagonalization_matrix((0, 1), n)[:n_states, :n_states]
        constant, linear, quadratic, bilinear = _position_coefficients(
            matrix,
            hamiltonian["constant"][1],
            hamiltonian["linear"][1],
            hamiltonian["quadratic"][1],
            n_states,
            n_modes,
        )
        assert constant.shape == (n_states,)
        assert linear.shape == (n_modes, n_states)
        assert quadratic.shape == (n_modes, n_states)
        assert bilinear.shape == (n_modes * (n_modes - 1) // 2, n_states)

        # Check the rotated constant term is diagonal, not just that the diagonal matches itself.
        rotated_constant = matrix.T @ hamiltonian["constant"][1] @ matrix
        assert np.allclose(rotated_constant, np.diag(np.diag(rotated_constant)))
        assert np.allclose(constant, np.diag(rotated_constant))

    def test_momentum_coefficients(self, seed):
        """Test momentum coefficient extraction against the injected diagonal values."""
        n_states, n_modes = 2, 3
        rng = np.random.default_rng(seed)
        p_quad = rng.random(n_modes)
        kinetic = np.einsum("ab,cd->abcd", np.eye(n_states), np.diag(p_quad))
        assert np.allclose(_momentum_coefficients(kinetic), p_quad)


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
        qp.BasisState(qp.math.int_to_binary(x, len(x_wires)), wires=x_wires)
        qp.BasisState(qp.math.int_to_binary(y % (2 ** len(y_wires)), len(y_wires)), wires=y_wires)
        qp.BasisState(qp.math.int_to_binary(z, len(output_wires)), wires=output_wires)
        _half_signed_out_multiplier(x_wires, y_wires, output_wires, work_wires)
        return qp.probs(wires=output_wires)

    # the circuit permutes basis states, so the output is a single basis state
    assert int(np.argmax(circuit())) == expected


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
        # ``work_wires`` are auxiliary scratch and excluded from ``op.wires``.
        algorithmic_wires = set()
        for name, reg in wires.items():
            if name != "work":
                algorithmic_wires |= set(reg)
        assert set(op.wires) == algorithmic_wires
        assert set(wires["work"]).isdisjoint(op.wires)

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

    @pytest.mark.parametrize("num_steps", [0, -1, 1.5, True, np.int64(0)])
    def test_rejects_invalid_num_trotter_steps(self, num_steps):
        """Test that invalid ``num_trotter_steps`` values are rejected (including ``bool``)."""
        hamiltonian = build_hamiltonian(fragment_list())
        with pytest.raises(ValueError, match="positive integer"):
            make_op(hamiltonian, make_wires(2, 2), num_trotter_steps=num_steps)

    @pytest.mark.parametrize("num_steps", [np.int64(2), np.int32(3)])
    def test_accepts_numpy_integer_num_trotter_steps(self, num_steps):
        """Test that numpy integers (the natural output of ``len()`` arithmetic) are accepted."""
        hamiltonian = build_hamiltonian(fragment_list())
        op = make_op(hamiltonian, make_wires(2, 2), num_trotter_steps=num_steps)
        assert op.arguments["num_trotter_steps"] == num_steps

    @pytest.mark.parametrize(
        "register, match",
        [
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

    def test_rejects_oversized_cache(self):
        """Test that an over-sized ``cache`` register is rejected (resource fn assumes ``2k``)."""
        hamiltonian = build_hamiltonian(fragment_list(n_states=4, n_modes=2))
        wires = make_wires(4, 2)
        wires["cache"] = list(wires["cache"]) + [max(wires["work"]) + 1]  # 2k + 1 wires
        op = make_op(hamiltonian, wires)
        with pytest.raises(ValueError, match="cache qubits"):
            op.compute_decomposition(**op.arguments)

    def test_accepts_list_hamiltonian(self):
        """Test that nested list/tuple Hamiltonian leaves are coerced to arrays."""
        hamiltonian = build_hamiltonian(fragment_list(n_states=2, n_modes=2))
        hamiltonian = {key: value.tolist() for key, value in hamiltonian.items()}
        op = make_op(hamiltonian, make_wires(2, 2))
        assert all(isinstance(v, np.ndarray) for v in op.arguments["hamiltonian"].values())

    def test_rejects_bad_electronic_size_at_construction(self):
        """Test that a wrongly-sized electronic register is rejected at construction time."""
        hamiltonian = build_hamiltonian(fragment_list(n_states=4, n_modes=2))
        wires = make_wires(4, 2)
        wires["electronic"] = wires["electronic"][:-1]
        with pytest.raises(ValueError, match="electronic states"):
            make_op(hamiltonian, wires)

    def test_rejects_non_divisible_vib_wires(self):
        """Test that a vibrational register not divisible by the number of modes is rejected at
        construction time."""
        hamiltonian = build_hamiltonian(fragment_list(n_states=2, n_modes=2))
        wires = make_wires(2, 2)
        wires["vib_wires"] = wires["vib_wires"][:-1]  # 3 wires, 2 modes -> not divisible
        with pytest.raises(ValueError, match="divisible by the number of modes"):
            make_op(hamiltonian, wires)

    def test_rejects_non_power_of_2_n_states(self):
        """Test that a non-power-of-2 number of electronic states is rejected."""
        hamiltonian = build_hamiltonian(fragment_list(n_states=3, n_modes=2))
        wires = make_wires(3, 2)
        with pytest.raises(ValueError, match="power of 2"):
            make_op(hamiltonian, wires)

    def test_rejects_unequal_mode_register_sizes(self):
        """Test that _validate_registers rejects vibrational-mode registers of differing sizes."""
        registers = {
            "electronic": [0],
            "cache": [1, 2, 3, 4],
            "coefficients": [5],
            "phase_gradient": [6],
            "work": [7, 8, 9, 10],
        }
        # The first mode has two wires while the second has one, which is invalid.
        mode_registers = [[11, 12], [13]]
        with pytest.raises(ValueError, match="same size"):
            _validate_registers(registers, mode_registers, n_modes=2, n_states=2)

    def test_wires_are_concrete_rejects_abstract_wires(self):
        """Test that ``_wires_are_concrete`` rejects ``AbstractWires`` placeholders.

        ``Wires(AbstractWires(n))`` would otherwise look concrete.
        """
        assert _wires_are_concrete(AbstractWires(2)) is False
        assert _wires_are_concrete([0, 1, 2]) is True

    def test_validate_registers_rejects_bad_electronic_size(self):
        """Test that ``_validate_registers`` rejects a wrongly-sized electronic register."""
        registers = {
            "electronic": [0],  # 1 wire, but n_states=4 needs 2
            "cache": [1, 2, 3, 4],
            "coefficients": [5],
            "phase_gradient": [6],
            "work": [7, 8, 9, 10],
        }
        mode_registers = [[11, 12], [13, 14]]
        with pytest.raises(ValueError, match="electronic states"):
            _validate_registers(registers, mode_registers, n_modes=2, n_states=4)

    def test_init_skips_validation_with_traced_wire_label(self):
        """Test that register-size validation is skipped when a wire label is traced.

        Uses an invalid ``vib_wires`` size to confirm the check is genuinely skipped.
        """
        jax = pytest.importorskip("jax")
        hamiltonian = build_hamiltonian(fragment_list())
        wires = make_wires(2, 2)

        def make_traced(w0):
            op = qp.TrotterVibronic(
                evolution_time=0.5,
                num_trotter_steps=1,
                hamiltonian=hamiltonian,
                electronic_wires=[w0],
                vib_wires=wires["vib_wires"][:-1],  # invalid size, skipped since w0 is traced
                coefficient_wires=wires["coefficients"],
                phase_gradient_wires=wires["phase_gradient"],
                cache_wires=wires["cache"],
                work_wires=wires["work"],
            )
            assert op.name == "TrotterVibronic"
            return 0

        jax.make_jaxpr(make_traced)(np.asarray(wires["electronic"][0]))

    def test_init_accepts_traced_hamiltonian(self):
        """Test that a traced Hamiltonian is accepted and uses the ``(0, j)`` blocks structure."""
        jax = pytest.importorskip("jax")
        hamiltonian = build_hamiltonian(fragment_list(n_states=2, n_modes=1))
        wires = make_wires(2, 1)

        def make_traced(constant):
            traced = dict(hamiltonian)
            traced["constant"] = constant
            op = qp.TrotterVibronic(
                evolution_time=0.5,
                num_trotter_steps=1,
                hamiltonian=traced,
                electronic_wires=wires["electronic"],
                vib_wires=wires["vib_wires"],
                coefficient_wires=wires["coefficients"],
                phase_gradient_wires=wires["phase_gradient"],
                cache_wires=wires["cache"],
                work_wires=wires["work"],
            )
            assert op.name == "TrotterVibronic"
            return 0

        jax.make_jaxpr(make_traced)(np.asarray(hamiltonian["constant"]))


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

    def test_kinetic_queues_aqft_no_basisstate(self, seed):
        """Test that the kinetic fragment queues an AQFT only at non-zero evolution time."""
        hamiltonian = build_hamiltonian(fragment_list(seed=seed))
        wires = make_wires(2, 2)
        zero = decomposition_queue(make_op(hamiltonian, wires, evolution_time=0.0))
        assert count_ops(zero, AQFT) == 0
        nonzero = decomposition_queue(make_op(hamiltonian, wires, evolution_time=0.1))
        assert count_ops(nonzero, AQFT) > 0
        assert count_ops(nonzero, qp.BasisState) == 0  # momentum coeffs loaded via ``PauliX``

    def test_basis_loading_uses_multix(self):
        """Test that non-zero momentum coefficients are loaded with ``MultiX``."""
        n_states, n_modes = 2, 1
        position = _zero_fragment(n_states, n_modes)
        kinetic = _zero_fragment(n_states, n_modes)
        for i in range(n_states):
            kinetic["kinetic"][i, i] = np.diag([3.0] * n_modes)
        hamiltonian = build_hamiltonian([position, kinetic])
        queue = decomposition_queue(make_op(hamiltonian, make_wires(n_states, n_modes), 1.0))
        assert count_ops(queue, qp.BasisState) == 0
        assert count_ops(queue, qp.MultiX) > 0

    def test_bilinear_queues_signed_out_multiplier(self):
        """Test that a bilinear fragment queues a SignedOutMultiplier."""
        fragments = [
            _single_fragment(2, 3, [("Q", "Q")], seed=0, skip_quadratic=True),
            _single_fragment(2, 3, [("P", "P")], seed=1),
        ]
        hamiltonian = build_hamiltonian(fragments)
        queue = decomposition_queue(make_op(hamiltonian, make_wires(2, 3), evolution_time=0.1))
        assert count_ops(queue, SignedOutMultiplier) > 0

    def test_resource_function_is_graph_compatible(self):
        """Test that ``compute_resources`` succeeds with abstractifiable operator keys."""
        hamiltonian = build_hamiltonian(fragment_list(seed=1))
        op = make_op(hamiltonian, make_wires(2, 2))
        rule = qp.list_decomps(qp.TrotterVibronic)[0]
        resources = rule.compute_resources(**op.arguments)
        assert resources.num_gates > 0

    @pytest.mark.jax
    @pytest.mark.usefixtures("enable_and_disable_graph_decomp", "enable_and_disable_capture")
    def test_assert_valid(self):
        """Test that ``TrotterVibronic`` passes ``assert_valid``."""
        hamiltonian = build_hamiltonian(fragment_list(n_states=2, n_modes=2, seed=1))
        op = make_op(hamiltonian, make_wires(2, 2), evolution_time=0.5)
        qp.ops.functions.assert_valid(
            op, skip_differentiation=True, skip_decomp_matrix_check=True, skip_wire_mapping=True
        )

    @pytest.mark.jax
    @pytest.mark.usefixtures("enable_and_disable_graph_decomp", "enable_and_disable_capture")
    def test_optional_work_wires_are_allocated(self):
        """Test that ``cache`` and ``work`` are optional and dynamically allocated when omitted."""
        hamiltonian = build_hamiltonian(fragment_list(n_states=2, n_modes=2, seed=1))
        wires = make_wires(2, 2)
        op = qp.TrotterVibronic(
            evolution_time=0.5,
            num_trotter_steps=1,
            hamiltonian=hamiltonian,
            electronic_wires=wires["electronic"],
            vib_wires=wires["vib_wires"],
            coefficient_wires=wires["coefficients"],
            phase_gradient_wires=wires["phase_gradient"],
            aqft_order=1,
        )
        assert len(op.arguments["cache_wires"]) == 0
        assert len(op.arguments["work_wires"]) == 0

        num_allocate, num_deallocate = count_allocations(op)
        assert num_allocate == 1
        assert num_deallocate == 1

        qp.ops.functions.assert_valid(
            op, skip_differentiation=True, skip_decomp_matrix_check=True, skip_wire_mapping=True
        )

    @pytest.mark.jax
    @pytest.mark.usefixtures("enable_and_disable_graph_decomp", "enable_and_disable_capture")
    def test_optional_coefficient_wires_are_allocated(self):
        """Test that ``coefficient_wires`` is optional and dynamically allocated (sized to match
        ``phase_gradient_wires``) when omitted, alongside ``cache_wires``/``work_wires``."""
        hamiltonian = build_hamiltonian(fragment_list(n_states=2, n_modes=2, seed=1))
        wires = make_wires(2, 2)
        op = qp.TrotterVibronic(
            evolution_time=0.5,
            num_trotter_steps=1,
            hamiltonian=hamiltonian,
            electronic_wires=wires["electronic"],
            vib_wires=wires["vib_wires"],
            phase_gradient_wires=wires["phase_gradient"],
            aqft_order=1,
        )
        assert len(op.arguments["coefficient_wires"]) == 0
        assert len(op.arguments["cache_wires"]) == 0
        assert len(op.arguments["work_wires"]) == 0

        num_allocate, num_deallocate = count_allocations(op)
        assert num_allocate == 1
        assert num_deallocate == 1

        qp.ops.functions.assert_valid(
            op, skip_differentiation=True, skip_decomp_matrix_check=True, skip_wire_mapping=True
        )

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_explicit_work_wires_are_not_allocated(self):
        """Test that providing ``cache``/``work`` explicitly skips the dynamic allocation."""
        hamiltonian = build_hamiltonian(fragment_list(n_states=2, n_modes=2, seed=1))
        op = make_op(hamiltonian, make_wires(2, 2))
        assert count_ops(decomposition_queue(op), Allocate) == 0

    @pytest.mark.capture
    def test_position_fragments_loop_is_traced(self, seed):
        """Test that the position-fragment loop is a traced ``for_loop`` under capture, rather
        than unrolled by the fragment count: the number of traced ``for_loop`` primitives is the
        same regardless of the number of position fragments."""
        import jax  # pylint: disable=import-outside-toplevel
        from jax._src.core import ClosedJaxpr, Jaxpr  # pylint: disable=import-outside-toplevel

        from pennylane.capture.primitives import (  # pylint: disable=import-outside-toplevel
            for_loop_prim,
        )

        def count_for_loops(jaxpr):
            count = sum(eqn.primitive == for_loop_prim for eqn in jaxpr.eqns)
            for eqn in jaxpr.eqns:
                for val in eqn.params.values():
                    if isinstance(val, ClosedJaxpr):
                        val = val.jaxpr
                    if isinstance(val, Jaxpr):
                        count += count_for_loops(val)
            return count

        def num_for_loops(n_states):
            hamiltonian = build_hamiltonian(fragment_list(n_states=n_states, n_modes=1, seed=seed))
            op = make_op(hamiltonian, make_wires(n_states, 1), evolution_time=0.3)
            decomposition = qp.list_decomps(qp.TrotterVibronic)[0]
            jaxpr = jax.make_jaxpr(lambda: decomposition(**op.arguments))()
            return count_for_loops(jaxpr.jaxpr)

        # ``num_position_fragments == n_states`` here; if the fragment loop were unrolled, the
        # traced ``for_loop`` count would scale with it instead of staying constant.
        assert num_for_loops(2) == num_for_loops(8)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp", "enable_and_disable_capture")
    def test_decomposition_resource_consistency(self, seed):
        """Test resource/decomposition consistency via ``_test_decomposition_rule``."""
        from pennylane.ops.functions.assert_valid import (  # pylint: disable=import-outside-toplevel
            _test_decomposition_rule,
        )

        hamiltonian = build_hamiltonian(fragment_list(n_states=2, n_modes=2, seed=seed))
        op = make_op(hamiltonian, make_wires(2, 2), evolution_time=0.7)
        rule = qp.list_decomps(qp.TrotterVibronic)[0]
        _test_decomposition_rule(op, rule, skip_decomp_matrix_check=True)


# ---------------------------------------------------------------------------
# ------------------------------- Execution ---------------------------------
# ---------------------------------------------------------------------------


def test_default_qubit_execution():
    """Test that a small vibronic Trotter circuit executes on default.qubit."""
    n_states, n_modes, k, b = 2, 1, 3, 2
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


def _binary_decimals_int(value, precision):
    """Integer that ``binary_decimals(value, precision, unit=2*pi)`` encodes (big-endian)."""
    bits = qp.math.binary_decimals(value, precision, unit=2 * np.pi)
    return int(np.dot(np.asarray(bits), 2 ** np.arange(precision)[::-1]))


def _prepare_phase_gradient(pg_wires):
    """Prepare PennyLane's phase-gradient state |∇n> = (1/sqrt(N)) sum_m e^{-2 pi i m/N} |m>."""
    for i, w in enumerate(pg_wires):
        qp.H(w)
        qp.PhaseShift(-np.pi / 2**i, w)


def _phase_gradient_int(hamiltonian, wires, mode_value=0, electronic_state=0, evolution_time=1.0):
    """Run ``TrotterVibronic`` with the phase-gradient register prepared in ``|0>`` and return the
    integer it holds afterwards.

    Preparing the phase-gradient register (and everything else) in a computational basis state
    turns the phase-gradient arithmetic into an exact integer accumulation: every sub-operation
    permutes basis states, so the final register content is a single basis state whose integer can
    be predicted exactly from ``binary_decimals`` -- no phase-gradient resource state, no
    truncation error, no statevector comparison (see `Motlagh et al, arXiv:2411.13669`).
    """
    k = len(wires["vib_wires"])
    n = len(wires["electronic"])
    num_wires = max(w for reg in wires.values() for w in reg) + 1
    dev = qp.device("default.qubit", wires=num_wires)

    @qp.qnode(dev)
    def circuit():
        if mode_value:
            qp.BasisState(qp.math.int_to_binary(mode_value, k), wires=wires["vib_wires"])
        if electronic_state:
            qp.BasisState(qp.math.int_to_binary(electronic_state, n), wires=wires["electronic"])
        make_op(hamiltonian, wires, evolution_time=evolution_time, num_trotter_steps=1)
        return qp.probs(wires=wires["phase_gradient"])

    probs = circuit()
    return int(np.argmax(probs))


class TestNumericalCorrectness:
    """Exact checks of the accumulated phase-gradient integer (mod ``2**b``).

    Predicts the result from ``binary_decimals``; fails if the wrong phases are applied.
    """

    @pytest.mark.parametrize("electronic_state, m", [(0, 5), (1, 3)])
    def test_constant_term_phase(self, electronic_state, m):
        """A diagonal constant fragment adds ``binary_decimals(c/2)`` per pass (twice per step)."""
        n_states, n_modes, k, b = 2, 1, 3, 4
        # Choose the constant as an exact fraction of 2*pi so ``binary_decimals`` is exact:
        # ``c / 2 = 2*pi * m / 2**b``.
        c = 2 * np.pi * m / 2 ** (b - 1)
        constant = np.zeros((1, n_states, n_states))
        constant[0, electronic_state, electronic_state] = c
        hamiltonian = {
            "constant": constant,
            "linear": np.zeros((1, n_states, n_states, n_modes)),
            "quadratic": np.zeros((1, n_states, n_states, n_modes, n_modes)),
            "kinetic": np.zeros((n_states, n_states, n_modes, n_modes)),
        }
        wires = make_wires(n_states, n_modes, k=k, b=b)
        got = _phase_gradient_int(hamiltonian, wires, electronic_state=electronic_state)
        # Coefficients are negated before loading (phase-gradient sign convention), so the
        # accumulated integer is ``-2 * m`` rather than ``2 * m`` (both mod ``2**b``).
        expected = (2 * _binary_decimals_int(-c / 2, b)) % (2**b)
        assert got == expected == (-2 * m) % (2**b)

    def test_linear_term_phase(self):
        """A linear fragment multiplies ``binary_decimals(lambda/2)`` by the (signed) mode value,
        exercising the half-signed multiplier."""
        n_states, n_modes, k, b = 2, 1, 3, 5
        m, q = 2, 3  # linear coefficient -> bd integer ``m``; positive mode value ``q``
        lam = 2 * np.pi * m / 2 ** (b - 1)
        linear = np.zeros((1, n_states, n_states, n_modes))
        linear[0, 0, 0, 0] = lam
        hamiltonian = {
            "constant": np.zeros((1, n_states, n_states)),
            "linear": linear,
            "quadratic": np.zeros((1, n_states, n_states, n_modes, n_modes)),
            "kinetic": np.zeros((n_states, n_states, n_modes, n_modes)),
        }
        wires = make_wires(n_states, n_modes, k=k, b=b)
        got = _phase_gradient_int(hamiltonian, wires, mode_value=q, electronic_state=0)
        # Negated coefficient (see ``test_constant_term_phase``) times the (signed) mode value.
        expected = (2 * _binary_decimals_int(-lam / 2, b) * q) % (2**b)
        assert got == expected == (-2 * m * q) % (2**b)

    @pytest.mark.parametrize("mode_value", [0, 1, 2])
    def test_sign_matches_exact_evolution(self, mode_value):
        """With a genuine phase-gradient state the template realizes ``e^{-iHt}`` (not
        ``e^{+iHt}``): the global sign that the integer-accumulation tests above cannot see.

        A single position fragment (kinetic zero) is diagonal, so one Trotter step is exact and
        the electronic register picks up the diagonal phase ``e^{-i theta_a}`` per electronic
        state ``a``. Comparing the measured relative phase against the exact one pins the sign.
        """
        n_states, n_modes, k, b = 2, 1, 3, 3
        t = 1.0
        # Exact binary fractions of 2*pi (so ``binary_decimals`` is exact): coeff = 2*pi*m/2**(b-1).
        scale = 2 * np.pi / 2 ** (b - 1)
        const_m, lin_m, quad_m = [1, -1], [1, -1], [1, 0]
        constant = np.zeros((1, n_states, n_states))
        linear = np.zeros((1, n_states, n_states, n_modes))
        quadratic = np.zeros((1, n_states, n_states, n_modes, n_modes))
        for a in range(n_states):
            constant[0, a, a] = const_m[a] * scale
            linear[0, a, a, 0] = lin_m[a] * scale
            quadratic[0, a, a, 0, 0] = quad_m[a] * scale
        hamiltonian = {
            "constant": constant,
            "linear": linear,
            "quadratic": quadratic,
            "kinetic": np.zeros((n_states, n_states, n_modes, n_modes)),
        }
        wires = make_wires(n_states, n_modes, k=k, b=b)
        num_wires = max(w for reg in wires.values() for w in reg) + 1
        dev = qp.device("default.qubit", wires=num_wires)

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(wires["electronic"][0])  # electronic |+>
            if mode_value:
                qp.BasisState(qp.math.int_to_binary(mode_value, k), wires=wires["vib_wires"])
            _prepare_phase_gradient(wires["phase_gradient"])
            make_op(hamiltonian, wires, evolution_time=t, num_trotter_steps=1)
            return qp.density_matrix(wires=wires["electronic"])

        measured = np.angle(circuit()[1, 0])

        def theta(a):
            q = mode_value
            return t * (
                constant[0, a, a] + linear[0, a, a, 0] * q + quadratic[0, a, a, 0, 0] * q**2
            )

        expected = -(theta(1) - theta(0))  # arg(e^{-i theta_1} conj(e^{-i theta_0}))
        assert np.isclose(np.exp(1j * measured), np.exp(1j * expected))


# ---------------------------------------------------------------------------
# --------------------------- Extra data helpers ----------------------------
# ---------------------------------------------------------------------------


# pylint: disable-next=too-many-arguments
def _single_fragment(n_states, n_modes, include_op_types, seed=0, skip_quadratic=False, m=0):
    """Build a single random position or kinetic dense fragment for targeted tests.

    ``m`` is the XOR shift of the populated electronic blocks; as position fragment ``i``, it must
    equal ``i`` to match the ``(0, i)`` diagonalization convention.
    """
    rng = np.random.default_rng(seed)
    fragment = _zero_fragment(n_states, n_modes)

    if include_op_types == [("P", "P")]:
        diagonal = np.diag(rng.random(n_modes))
        for i in range(n_states):
            fragment["kinetic"][i, i] = diagonal
        return fragment

    op_type_to_key = {(): "constant", ("Q",): "linear", ("Q", "Q"): "quadratic"}
    elec_ids = [(i, i ^ m) for i in range(n_states) if i ^ m < n_states]
    for i, j in elec_ids:
        for op_type in include_op_types:
            tensor = rng.random((n_modes,) * len(op_type))
            if len(op_type) == 2:
                tril = 0 if skip_quadratic else -1
                tensor[np.tril_indices(n_modes, k=tril)] = 0.0
            key = op_type_to_key[op_type]
            fragment[key][i, j] = fragment[key][j, i] = tensor
    return fragment

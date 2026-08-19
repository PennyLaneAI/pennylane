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
from pennylane.templates.subroutines import AQFT, QROM, SemiAdder
from pennylane.templates.subroutines.arithmetic import (
    OutMultiplier,
    SignedOutMultiplier,
    SignedOutSquare,
)
from pennylane.templates.subroutines.time_evolution.trotter_vibronic import (
    _derive_diag_keys,
    _diagonalization_matrix,
    _diagonalize_vibronic_circuit,
    _half_signed_out_multiplier,
    _momentum_coefficients,
    _position_coefficients,
    _validate_registers,
)
from pennylane.typing import AbstractWires

# ---------------------------------------------------------------------------
# --------------------------- Test data helpers -----------------------------
# ---------------------------------------------------------------------------
#
# The dense vibronic Hamiltonians used throughout these tests are exactly the data that
# ``pennylane.labs.trotter_error.vibronic_fragments`` (with the default "blocks" fragmentation
# scheme) followed by a dense conversion would produce. They are reconstructed here directly in
# NumPy so the test suite does not depend on ``pennylane.labs``. Each "fragment" is a dictionary
# of dense coefficient tensors; ``build_hamiltonian`` stacks the position fragments and appends
# the kinetic fragment, matching the stacked dense output the template consumes.


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
    constant_coeffs = rng.random((n_states, n_states))
    linear_coeffs = rng.random((n_states, n_states, n_modes))

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

    def test_rejects_unequal_mode_register_sizes(self):
        """Test that _validate_registers rejects vibrational-mode registers of differing sizes."""
        registers = {
            "electronic": [0],
            "cache": [1, 2, 3, 4],
            "coefficients": [5],
            "phase gradient": [6],
            "work": [7, 8, 9, 10],
        }
        # The first mode has two wires while the second has one, which is invalid.
        mode_registers = [[11, 12], [13]]
        with pytest.raises(ValueError, match="same size"):
            _validate_registers(registers, mode_registers, n_modes=2, n_states=2)

    def test_abstract_init_derives_diag_keys(self):
        """Test that abstract construction (here triggered by an abstract wire register) derives
        the diagonalization keys from the concrete Hamiltonian via ``__abstract_init__``."""
        hamiltonian = build_hamiltonian(fragment_list())
        wires = make_wires(2, 2)
        op = qp.TrotterVibronic(
            evolution_time=0.5,
            num_trotter_steps=1,
            hamiltonian=hamiltonian,
            electronic=AbstractWires(len(wires["electronic"])),
            vib_wires=wires["vib_wires"],
            cache=wires["cache"],
            coefficients=wires["coefficients"],
            phase_gradient=wires["phase_gradient"],
            work=wires["work"],
            aqft_order=1,
        )
        assert op.is_abstract
        assert op.arguments["diag_keys"] == _derive_diag_keys(hamiltonian)

    @pytest.mark.capture
    def test_abstract_init_requires_diag_keys_for_traced_hamiltonian(self):
        """Test that abstract construction with a traced Hamiltonian and no ``diag_keys`` errors,
        since the diagonalization keys cannot be derived from an abstract Hamiltonian."""
        jax = pytest.importorskip("jax")
        wires = make_wires(2, 1)

        def make_traced(constant, linear, quadratic, kinetic):
            hamiltonian = {
                "constant": constant,
                "linear": linear,
                "quadratic": quadratic,
                "kinetic": kinetic,
            }
            return qp.TrotterVibronic(
                evolution_time=0.5,
                num_trotter_steps=1,
                hamiltonian=hamiltonian,
                electronic=AbstractWires(len(wires["electronic"])),
                vib_wires=wires["vib_wires"],
                cache=wires["cache"],
                coefficients=wires["coefficients"],
                phase_gradient=wires["phase_gradient"],
                work=wires["work"],
                aqft_order=1,
                diag_keys=None,
            )

        hamiltonian = build_hamiltonian(fragment_list(n_states=2, n_modes=1))
        with pytest.raises(ValueError, match="diag_keys"):
            jax.make_jaxpr(make_traced)(
                hamiltonian["constant"],
                hamiltonian["linear"],
                hamiltonian["quadratic"],
                hamiltonian["kinetic"],
            )


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

    def test_resource_function_is_graph_compatible(self):
        """Test that the registered decomposition's resource keys can be abstractified.

        This guards the resource function against regressing to bare operator classes (which the
        decomposition graph cannot abstractify for :class:`~.Operator2` sub-operations such as QROM).
        """
        hamiltonian = build_hamiltonian(fragment_list(seed=1))
        op = make_op(hamiltonian, make_wires(2, 2))
        rule = qp.list_decomps(qp.TrotterVibronic)[0]
        # ``compute_resources`` abstractifies every key internally and raises for bare Operator2
        # classes, so a successful call with positive gate count is the assertion of interest.
        resources = rule.compute_resources(**op.arguments)
        assert resources.num_gates > 0

    @pytest.mark.capture
    def test_decomposition_captures_into_plxpr(self, seed):
        """Test that the decomposition can be captured into plxpr, exercising the program-capture
        branches that build jax arrays for the mode registers and Hamiltonian coefficients."""
        jax = pytest.importorskip("jax")
        # Two modes so the bilinear terms (and thus every coefficient branch) are traced.
        hamiltonian = build_hamiltonian(fragment_list(n_states=2, n_modes=2, seed=seed))
        op = make_op(hamiltonian, make_wires(2, 2), evolution_time=0.7)
        jaxpr = jax.make_jaxpr(lambda: op.compute_decomposition(**op.arguments))()
        assert len(jaxpr.eqns) > 0


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
    """Build a single random position or kinetic dense fragment for targeted tests.

    This mirrors the fragments the labs ``vibronic_fragments`` helpers would produce for the
    requested operator types: a diagonal kinetic fragment for ``[("P", "P")]``, otherwise a
    position fragment populating the ``(i, i ^ m)`` electronic blocks (and their transpose) with
    the requested constant/linear/quadratic coefficients. Quadratic tensors keep only the strict
    upper triangle (``skip_quadratic=True``) or the upper triangle including the diagonal.
    """
    rng = np.random.default_rng(seed)
    fragment = _zero_fragment(n_states, n_modes)

    if include_op_types == [("P", "P")]:
        diagonal = np.diag(rng.random(n_modes))
        for i in range(n_states):
            fragment["kinetic"][i, i] = diagonal
        return fragment

    op_type_to_key = {(): "constant", ("Q",): "linear", ("Q", "Q"): "quadratic"}
    m = int(rng.integers(0, n_states))
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

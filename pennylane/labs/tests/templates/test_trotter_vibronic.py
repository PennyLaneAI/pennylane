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
"""Contains tests for the Trotter template for vibronic Hamiltonians."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates.trotter_vibronic import (
    _extract_registers,
    _preprocess_data,
    _validate_fragments,
    _validate_registers,
    diagonalize_vibronic_mat,
    diagonalize_vibronic_qjit,
    fragment_to_dense,
    get_momentum_coefficients,
    get_position_coefficients,
    load_coefficients,
    trotter_vibronic,
)
from pennylane.labs.trotter_error.fragments import vibronic_fragments
from pennylane.labs.trotter_error.realspace import (
    RealspaceCoeffs,
    RealspaceMatrix,
    RealspaceOperator,
    RealspaceSum,
)
from pennylane.ops.qubit.state_preparation import BasisState
from pennylane.templates.subroutines import AQFT, QROM, SemiAdder
from pennylane.templates.subroutines.arithmetic import (
    OutMultiplier,
    SignedOutMultiplier,
    SignedOutSquare,
)

N_STATES, N_MODES, K, B = 2, 2, 2, 3
_ARITHMETIC_OPS = (SemiAdder, OutMultiplier, SignedOutSquare, SignedOutMultiplier, BasisState)


def _random_vibronic_elec_ids(n_states, rng):
    m = rng.integers(0, n_states)
    return [(i, int(i ^ m)) for i in range(n_states) if i ^ m < n_states]


def random_vibronic_fragment(
    n_states, n_modes, include_op_types=None, seed=0, skip_quadratic=False
):
    """Build a random position or kinetic fragment."""
    rng = np.random.default_rng(seed)
    include_op_types = include_op_types or [(), ("Q",), ("Q", "Q")]

    if include_op_types == [("P", "P")]:
        op = RealspaceOperator(
            n_modes, ("P", "P"), RealspaceCoeffs(np.diag(rng.random(n_modes)), "label")
        )
        blocks = {(i, i): RealspaceSum(n_modes, [op]) for i in range(n_states)}
        return RealspaceMatrix(n_states, n_modes, blocks)

    blocks = {}
    for elec_idx in _random_vibronic_elec_ids(n_states, rng):
        ops = []
        for op_type in include_op_types:
            tensor = rng.random((n_modes,) * len(op_type))
            if len(op_type) == 2:
                diagonal = 0 if skip_quadratic else -1
                tensor[np.tril_indices(n_modes, k=diagonal)] = 0.0
            ops.append(RealspaceOperator(n_modes, op_type, RealspaceCoeffs(tensor, "label")))
        blocks[elec_idx] = blocks[elec_idx[::-1]] = RealspaceSum(n_modes, ops)
    return RealspaceMatrix(n_states, n_modes, blocks)


def _vibronic_fragment_list(n_states=N_STATES, n_modes=N_MODES, seed=42):
    """Create a list of regular fragments via `vibronic_fragments` with random numbers."""
    rng = np.random.default_rng(seed)
    freqs = rng.random(n_modes)
    coeffs = [rng.random((n_states, n_states)), rng.random((n_states, n_states, n_modes))]
    return vibronic_fragments(n_states, n_modes, freqs, coeffs)


def _make_registers(n_states, n_modes, k=K, b=B):
    """Make standard registers for the `trotter_vibronic` template."""
    n = qp.math.ceil_log2(n_states)
    sizes = {
        "electronic": n,
        "cache": 2 * k,
        "coefficients": b,
        "phase gradient": b,
        "work": max(n - 1, 2 * k, 2 * b + 2),
        **{f"mode {i}": k for i in range(n_modes)},
    }
    return qp.registers(sizes)


def _queue_trotter(fragments, registers, time, *, num_steps=1):
    """Queue Trotterized vibronic Hamiltonian time evolution. Either calls the full template
    or just the internal Trotter step function."""
    with qp.queuing.AnnotatedQueue() as q:
        trotter_vibronic(time, num_steps, fragments, registers, aqft_order=1)
    return q.queue


def _count_ops(queue, *op_types):
    return sum(isinstance(op, op_types) for op in queue)


class TestDiagonalizeVibronic:
    """Tests for diagonalize_vibronic(_mat/_qjit), the diagonalization on
    the electronic register at the matrix and circuit level, respectively."""

    @pytest.mark.parametrize(
        "n_states, key, support",
        [(2, (0, 1), [0]), (4, (0, 3), [1, 0]), (8, (0, 7), [2, 1, 0])],
    )
    @pytest.mark.parametrize("fn", [diagonalize_vibronic_mat, diagonalize_vibronic_qjit])
    def test_expected_circuit(self, n_states, key, support, fn):
        """Test that the correct circuit is queued by the diagonalizing functions."""
        wires = list(range(qp.math.ceil_log2(n_states)))
        with qp.queuing.AnnotatedQueue() as q:
            fn(key=key, wires=wires)
        if not support:
            assert not q.queue
            return
        c = wires[support[0]]
        expected = [qp.Hadamard(c)] + [qp.CNOT([c, wires[i]]) for i in support[1:]]
        assert q.queue == expected

    def test_is_orthogonal(self):
        """Test that the matrix computed by diagonalize_vibronic_mat is orthogonal and actually
        diagonalizes the fragment matrix correctly."""
        n_states, wires = 4, list(range(2))
        rng = np.random.default_rng(0)
        matrix = np.zeros((n_states, n_states))
        m = 1
        for i, val in enumerate(rng.random(n_states)):
            if i < (j := i ^ m) < n_states:
                matrix[i, j] = matrix[j, i] = val
        U = qp.matrix(diagonalize_vibronic_mat, wires)(key=(0, 1), wires=wires)[
            :n_states, :n_states
        ]
        assert np.allclose(U @ U.T, np.eye(n_states))
        assert np.allclose(np.diag(np.diag(U.T @ matrix @ U)), U.T @ matrix @ U)


class TestFragmentReadout:
    """Test that fragments are read out correctly into numerical arrays."""

    @pytest.mark.parametrize("op_type", [(), ("Q",), ("Q", "Q"), ("P", "P")])
    def test_fragment_to_dense_roundtrip(self, op_type, seed):
        """Roundtrip test that a fragment is correctly read out into a dense array, by packing
        it back into a real space matrix and comparing with the original fragment."""
        fragment = random_vibronic_fragment(3, 2, [op_type], seed=seed)
        dense = fragment_to_dense(fragment, op_type)
        assert dense.shape == (3, 3) + (2,) * len(op_type)
        assert np.allclose(np.moveaxis(dense, 1, 0), dense)

        blocks = {}
        where = np.abs(dense) > 1e-12
        for _ in range(len(op_type)):
            where = np.any(where, axis=-1)
        for idx in zip(*np.where(where)):
            idx = (int(idx[0]), int(idx[1]))
            op = RealspaceOperator(2, op_type, RealspaceCoeffs(dense[idx], "label"))
            blocks[idx] = blocks[idx[::-1]] = RealspaceSum(2, [op])
        assert RealspaceMatrix(3, 2, blocks) == fragment

    def test_position_and_momentum_coefficients(self, seed):
        """Test that position and momentum coefficients are read out correctly."""
        position = random_vibronic_fragment(3, 2, seed=seed)
        kinetic = random_vibronic_fragment(3, 2, [("P", "P")], seed=seed + 52)
        wires = list(range(qp.math.ceil_log2(3)))
        key = next(k for k, v in position.get_coefficients().items() if v)
        M = qp.matrix(diagonalize_vibronic_mat, wires)(key=key, wires=wires)[:3, :3]

        c, lin, quad, bil = get_position_coefficients(position)
        assert c.shape == (3,) and lin.shape == (2, 3) and quad.shape == (2, 3)
        assert bil.shape == (2, 2, 3)
        assert np.allclose(M.T @ fragment_to_dense(position, ()) @ M, np.diag(c))

        p_quad = get_momentum_coefficients(kinetic)
        assert p_quad.shape == (2,)
        exp = fragment_to_dense(kinetic, ("P", "P"))
        assert np.allclose(np.einsum("ab,cd->abcd", np.eye(3), np.diag(p_quad)), exp)

    def test_preprocess_data(self, seed):
        """Test that _preprocess_data self-consistently processes position fragments data."""
        fragments = _vibronic_fragment_list(n_modes=3, seed=seed)
        (c, lin, quad, bil), bi_idx, _ = _preprocess_data(0.8, fragments)
        exp_c, exp_lin, exp_quad, exp_bil = get_position_coefficients(fragments[0])
        scale = 0.4
        assert np.allclose(c[0], exp_c * scale)
        assert np.allclose(lin[0], exp_lin * scale)
        assert np.allclose(quad[0], exp_quad * scale)
        assert np.allclose(bil[0], exp_bil[*bi_idx] * scale)


class TestExtractRegisters:
    """Tests for _extract_registers."""

    def setup(self):
        """Create registers and split them into mode and non-mode registers."""
        registers = _make_registers(N_STATES, N_MODES)
        mode_regs = [registers[f"mode {i}"] for i in range(N_MODES)]
        non_mode = {k: v for k, v in registers.items() if not k.startswith("mode ")}
        return non_mode, mode_regs

    def test_single_dict_terms(self):
        """Test the cases that yield a single dict of registers: constant, QROM and linear."""
        registers, mode_regs = self.setup()
        constant = _extract_registers(registers, mode_regs, "constant")
        assert set(constant) == {"x_wires", "y_wires", "work_wires"}
        assert constant["x_wires"] == registers["coefficients"]
        assert constant["y_wires"] == registers["phase gradient"]
        assert constant["work_wires"] == registers["work"]

        qrom = _extract_registers(registers, mode_regs, "QROM")
        assert set(qrom) == {"control_wires", "target_wires", "work_wires"}
        assert qrom["control_wires"] == registers["electronic"]
        assert qrom["target_wires"] == registers["coefficients"]
        assert qrom["work_wires"] == registers["work"][: len(registers["electronic"]) - 1]

        linear = _extract_registers(registers, mode_regs, "linear", 1)
        assert set(linear) == {"x_wires", "y_wires", "output_wires", "work_wires"}
        assert linear["x_wires"] == registers["coefficients"]
        assert linear["y_wires"] == mode_regs[1]
        assert linear["output_wires"] == registers["phase gradient"]
        assert linear["work_wires"] == registers["work"]

    def test_quadratic_and_bilinear(self):
        """Test the cases that yield two dicts of registers: quadratic and bilinear."""
        registers, mode_regs = self.setup()
        square, mult = _extract_registers(registers, mode_regs, "quadratic", 0)
        assert square["x_wires"] == mode_regs[0]
        assert square["output_wires"] == registers["cache"][1:]
        assert square["work_wires"] == registers["work"]
        assert mult["x_wires"] == registers["coefficients"]
        assert mult["y_wires"] == registers["cache"][1:]
        assert mult["output_wires"] == registers["phase gradient"]
        assert mult["work_wires"] == registers["work"]

        mode_mult, coeff_mult = _extract_registers(registers, mode_regs, "bilinear", 0, 1)
        assert mode_mult["x_wires"] == mode_regs[0]
        assert mode_mult["y_wires"] == mode_regs[1]
        assert mode_mult["output_wires"] == registers["cache"]
        assert mode_mult["work_wires"] == registers["work"]
        assert coeff_mult["x_wires"] == registers["coefficients"]
        assert coeff_mult["y_wires"] == registers["cache"]
        assert coeff_mult["output_wires"] == registers["phase gradient"]
        assert coeff_mult["work_wires"] == registers["work"]


class TestValidation:
    """Tests for register and fragment validation."""

    def test_accepts_defaults(self):
        """Test that default valid inputs are accepted by the validation functions."""
        fragments = _vibronic_fragment_list()
        registers = _make_registers(fragments[0].states, fragments[0].modes)
        _validate_fragments(fragments)
        _validate_registers(registers, fragments[0].modes, fragments[0].states)

    @pytest.mark.parametrize(
        "args, match",
        [
            ([(2, 2, [("P", "P")])], "potential and one kinetic"),
            (
                [
                    (2, 2),
                    (3, 2),
                    (2, 2, [("P", "P")]),
                ],
                "same number of electronic states",
            ),
            (
                [
                    (2, 2),
                    (2, 3),
                    (2, 2, [("P", "P")]),
                ],
                "same number of vibrational modes",
            ),
        ],
    )
    def test_rejects_bad_fragments(self, args, match, seed):
        """Tests that invalid (combinations of) fragments raise an error."""
        fragments = [random_vibronic_fragment(*_args, seed=seed) for _args in args]
        with pytest.raises(ValueError, match=match):
            _validate_fragments(fragments)

    def test_rejects_bad_fragment_roles(self):
        """Tests that a non-trailing kinetic fragment and a trailing non-kinetic
        fragment raise errors."""
        fragments = _vibronic_fragment_list()
        bad = fragments.copy()
        bad[0] = bad[-1]
        with pytest.raises(ValueError, match="position terms"):
            _validate_fragments(bad)
        bad = fragments.copy()
        bad[-1] = bad[0]
        with pytest.raises(ValueError, match="kinetic terms only"):
            _validate_fragments(bad)

    @pytest.mark.parametrize(
        "registers, match",
        [
            ("not a dict", "dictionary"),
            ({}, "keys in `registers`"),
        ],
    )
    def test_rejects_bad_register_container(self, registers, match):
        """Tests that an error is raised for wrong container types or empty dict."""
        with pytest.raises(ValueError, match=match):
            _validate_registers(registers, N_MODES, N_STATES)

    @pytest.mark.parametrize(
        "register, match",
        [
            ("electronic", "electronic states"),
            ("cache", "cache qubits"),
            ("phase gradient", "phase gradient"),
            ("work", "work qubits"),
            ("mode 1", "same size"),
        ],
    )
    def test_rejects_register_sizes(self, register, match):
        """Tests that errors are raised for wrongly sizes registers."""
        registers = _make_registers(4, 2)
        registers[register] = registers[register][:-1]
        with pytest.raises(ValueError, match=match):
            _validate_registers(registers, 2, 4)


def test_load_coefficients():
    """Test that load_coefficients queues the right operator (QROM) and returns the expected
    updated bit string for the loaded status."""
    precision, coefficients = 3, np.array([0.1, 0.2])
    prev = np.zeros((len(coefficients), precision), dtype=int)
    qrom_wires = {"control_wires": [0, 1], "target_wires": [2, 3, 4], "work_wires": [5]}
    with qp.queuing.AnnotatedQueue() as q:
        new = load_coefficients(coefficients, precision, prev, qrom_wires)
    expected = qp.math.binary_decimals(coefficients, precision, unit=2 * np.pi)
    assert np.allclose(new, expected)
    assert len(q.queue) == 1 and isinstance(q.queue[0], qp.QROM)
    assert np.allclose(q.queue[0].data[0], (prev + expected) % 2)


class TestTrotterVibronic:
    """Tests for the template `trotter_vibronic`."""

    def test_rejects_invalid_num_trotter_steps(self):
        """Test that an error is raised for an invalid number of Trotter steps (0)."""
        fragments = _vibronic_fragment_list()
        registers = _make_registers(fragments[0].states, fragments[0].modes)
        with pytest.raises(ValueError, match="positive integer"):
            trotter_vibronic(1.0, 0, fragments, registers, aqft_order=1)

    def test_zero_time_skips_arithmetic(self, seed):
        """Test that arithmetic operations are skipped if coefficients are zeroed, due
        to zeroed evolution time."""
        fragments = _vibronic_fragment_list(n_modes=3, seed=seed)
        registers = _make_registers(N_STATES, 3)
        queue = _queue_trotter(fragments, registers, 0.0)
        assert _count_ops(queue, *_ARITHMETIC_OPS, AQFT) == 0
        assert all(isinstance(op, (qp.H, qp.CNOT, qp.QROM)) for op in queue)

    @pytest.mark.parametrize("time", [0.0, 0.52])
    def test_num_steps_scales(self, time, seed):
        """Test that queued operators scale linearly with the number of Trotter steps."""
        fragments = _vibronic_fragment_list(seed=seed)
        registers = _make_registers(N_STATES, N_MODES)
        one = _queue_trotter(fragments, registers, time, num_steps=1)
        two = _queue_trotter(fragments, registers, time, num_steps=2)
        assert len(two) == 2 * len(one)

    @pytest.mark.parametrize("position_ops, op", [([()], SemiAdder), ([("Q",)], QROM)])
    def test_position_terms(self, position_ops, op, seed):
        """Test that for a single position and kinetic fragment, some representative arithmetic
        ops are queued."""
        fragments = [
            random_vibronic_fragment(N_STATES, N_MODES, position_ops, seed),
            random_vibronic_fragment(N_STATES, N_MODES, [("P", "P")], seed + 100),
        ]
        queue = _queue_trotter(fragments, _make_registers(N_STATES, N_MODES), 0.1)
        assert _count_ops(queue, op) > 0

    def test_quadratic_queues_more_squares_than_linear(self, seed):
        """Test that including quadratic position fragments increases the number of squares
        queued in the circuit (in addition to kinetic term)."""
        registers = _make_registers(N_STATES, N_MODES)
        lin_frag = random_vibronic_fragment(N_STATES, N_MODES, [("Q",)], seed)
        quad_frag = random_vibronic_fragment(N_STATES, N_MODES, [("Q", "Q")], seed + 325)
        kin_frag = random_vibronic_fragment(N_STATES, N_MODES, [("P", "P")], seed + 281)

        linear = _queue_trotter([lin_frag, kin_frag], registers, 0.1)
        quadratic = _queue_trotter([quad_frag, kin_frag], registers, 0.1)
        assert _count_ops(quadratic, SignedOutSquare) > _count_ops(linear, SignedOutSquare)

    def test_kinetic_at_positive_time(self, seed):
        """Test that BasisState and AQFT are queued for kinetic terms exactly if the
        time is non-zero."""
        registers = _make_registers(N_STATES, N_MODES)
        fragments = _vibronic_fragment_list(seed=seed)
        queue = _queue_trotter(fragments, registers, 0.0)
        assert _count_ops(queue, BasisState, AQFT) == 0
        queue = _queue_trotter(fragments, registers, 0.1)
        assert _count_ops(queue, BasisState) > 0 and _count_ops(queue, AQFT) > 0

    def test_bilinear_term(self):
        """Test that SignedOutMultipliers are queued for a bilinear fragment."""
        fragments = [
            random_vibronic_fragment(2, 3, [("Q", "Q")], skip_quadratic=True),
            random_vibronic_fragment(2, 3, [("P", "P")], 108),
        ]
        queue = _queue_trotter(fragments, _make_registers(2, 3), 0.1)
        assert _count_ops(queue, SignedOutMultiplier) > 0


@pytest.mark.xfail(
    reason="adjoint decomposition rules trace hyperparameters dynamically, which the rules reject."
)
@pytest.mark.catalyst
def test_catalyst_legacy_frontend(seed):
    """Test for compiling and running the template with qjit."""
    # pylint: disable=import-outside-toplevel,no-value-for-parameter

    from catalyst.device.decomposition import catalyst_decompose

    fragments = _vibronic_fragment_list(seed=seed)
    registers = _make_registers(N_STATES, N_MODES)
    num_wires = max(w for reg in registers.values() for w in reg) + 1
    target_gates = {
        "Hadamard",
        "QROM",
        "CNOT",
        "ForLoop",
        "Cond",
        "HybridAdjoint",
        "TemporaryAND",
        "Adjoint(TemporaryAND)",
        "AQFT",
        "X",
    }

    with qp.decomposition.toggle_graph_ctx(True):

        @qp.qjit
        @catalyst_decompose(capabilities=None, target_gates=target_gates)
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            trotter_vibronic(1.0, 2, fragments, registers, aqft_order=1)
            return qp.expval(qp.Z(0))

        assert np.isfinite(circuit())

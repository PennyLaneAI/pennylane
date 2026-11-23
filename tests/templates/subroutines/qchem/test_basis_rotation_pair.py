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
"""Tests for the BasisRotationPair template."""

from collections import Counter

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.templates.subroutines.qchem.basis_rotation_pair import (
    _basis_rotation_pair_two_basis_rotations,
    _complex_basis_rotation_pair_givens_pairs,
    _real_basis_rotation_pair_givens_pairs,
)
from pennylane.templates.subroutines.qchem.givens_pair import _givens_pair_rzlayer
from pennylane.transforms import _tape_decompose
from pennylane.transforms.decompositions import make_rzlayer_to_phase_gradient_decomp
from pennylane.typing import AbstractArray, Complex, Float, Wire
from pennylane.wires import Wires

RULES = [
    _basis_rotation_pair_two_basis_rotations,
    _real_basis_rotation_pair_givens_pairs,
    _complex_basis_rotation_pair_givens_pairs,
]

REAL_KINDS = ["det+1", "det-1"]
COMPLEX_KINDS = ["complex", "real as complex"]

# (rule, kind of matrix that the rule applies to)
GIVENS_RULE_CASES = [(_real_basis_rotation_pair_givens_pairs, kind) for kind in REAL_KINDS] + [
    (_complex_basis_rotation_pair_givens_pairs, kind) for kind in COMPLEX_KINDS
]

# Gates that neither BasisRotation nor the Givens rules emit, so that the decomposed circuits do
# not rely on the matrices of BasisRotationPair, GivensPair or RZLayer.
ELEMENTARY_GATES = {"StatePrep", "SingleExcitation", "PhaseShift", "PauliZ", "RZ", "GlobalPhase"}


def _random_matrix(num_wires, kind, seed):
    """Random orthogonal matrix with determinant +1 (``"det+1"``) or -1 (``"det-1"``), random
    unitary matrix (``"complex"``), or random orthogonal matrix with determinant -1 and complex
    dtype (``"real as complex"``)."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        mat = rng.normal(size=(num_wires, num_wires)) + 1j * rng.normal(size=(num_wires, num_wires))
        q, r = np.linalg.qr(mat)
        return q * (np.diag(r) / np.abs(np.diag(r)))

    q, r = np.linalg.qr(rng.normal(size=(num_wires, num_wires)))
    ortho = q * np.sign(np.diag(r))
    if np.linalg.det(ortho) * (1 if kind == "det+1" else -1) < 0:
        ortho[:, 0] *= -1
    return ortho.astype(complex) if kind == "real as complex" else ortho


def _random_state(num_wires, seed):
    """Random normalized state vector on ``num_wires`` wires."""
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**num_wires) + 1j * rng.normal(size=2**num_wires)
    return state / np.linalg.norm(state)


def _wires(num_wires, num_work_wires):
    """Spin-up and spin-down wires, interleaved, and work wires."""
    wires = list(range(2 * num_wires))
    work_wires = list(range(2 * num_wires, 2 * num_wires + num_work_wires))
    return wires[::2], wires[1::2], work_wires


def _two_basis_rotations_state(unitary_matrix, wires_up, wires_down, in_state):
    """State after BasisRotation on the spin-up and on the spin-down wires of ``in_state``."""
    num_wires = int(np.log2(len(in_state)))

    @qp.qnode(qp.device("default.qubit", wires=num_wires))
    def circuit():
        qp.StatePrep(in_state, wires=range(num_wires))
        qp.BasisRotation(unitary_matrix, wires_up)
        qp.BasisRotation(unitary_matrix, wires_down)
        return qp.state()

    return circuit()


class TestInitialization:
    """Tests for the arguments of BasisRotationPair."""

    def test_arguments(self):
        """Test that the wire arguments are stored and that the work wires are part of
        ``op.wires``."""
        op = qp.BasisRotationPair(np.eye(2), ["u0", "u1"], ["d0", "d1"], ["w"])

        assert op.wires_up == Wires(["u0", "u1"])
        assert op.wires_down == Wires(["d0", "d1"])
        assert op.work_wires == Wires(["w"])
        assert op.wires == Wires(["u0", "u1", "d0", "d1", "w"])

    @pytest.mark.parametrize(
        "unitary_matrix, wires_up, wires_down, work_wires, match",
        [
            (np.ones((2, 3)), [0, 1], [2, 3], [], "should be of shape NxN"),
            (np.eye(2), [0, 1], [2, 3, 4], [], "same number of wires"),
            (np.eye(1), [0], [1], [], "at least two wires"),
            (np.eye(2), [0, 1], [1, 2], [], "wires_up and wires_down must not overlap"),
            (np.eye(2), [0, 1], [2, 3], [3], "wires_down and work_wires must not overlap"),
        ],
    )
    def test_invalid_arguments(self, unitary_matrix, wires_up, wires_down, work_wires, match):
        """Test that a non-square matrix, unequal or too few wires per spin, and overlapping
        wires raise an error."""
        with pytest.raises(ValueError, match=match):
            qp.BasisRotationPair(unitary_matrix, wires_up, wires_down, work_wires)

    def test_unitarity_check(self):
        """Test that a non-unitary matrix raises an error if and only if ``check=True``."""
        non_unitary = np.array([[1.0, 0.5], [0.0, 1.0]])

        with pytest.raises(ValueError, match="should be unitary"):
            qp.BasisRotationPair(non_unitary, [0, 1], [2, 3], check=True)
        qp.BasisRotationPair(non_unitary, [0, 1], [2, 3])

    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128, np.complex64])
    def test_abstract_resource_rep(self, dtype):
        """Test that the resource representation is keyed by the shape and the exact dtype of the
        matrix, like that of BasisRotation, and by the numbers of wires."""
        op = qp.BasisRotationPair(np.eye(3, dtype=dtype), [0, 2, 4], [1, 3, 5], [6, 7])
        abstract_op = qp.BasisRotationPair(
            AbstractArray((3, 3), dtype), wires_up=Wire[3], wires_down=Wire[3], work_wires=Wire[2]
        )

        assert abstractify(op) == abstract_op
        assert hash(abstractify(op)) == hash(abstract_op)

    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    def test_abstract_resource_rep_from_leaf_dtype(self, dtype):
        """Test that a key built from the abstract dtype of the matrix matches the resource
        representation of a double-precision matrix."""
        unitary_matrix = np.eye(3, dtype=dtype)
        dtype_leaf, _ = qp.math.get_abstract_and_cast_dtype(unitary_matrix)
        key = qp.BasisRotationPair(
            dtype_leaf[3, 3], wires_up=Wire[3], wires_down=Wire[3], work_wires=Wire[2]
        )
        op = qp.BasisRotationPair(unitary_matrix, [0, 2, 4], [1, 3, 5], [6, 7])

        assert abstractify(op) == key
        assert hash(abstractify(op)) == hash(key)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize(
    "kind, num_wires, num_work_wires",
    [("det-1", 3, 1), ("det+1", 2, 0), ("complex", 2, 2), ("real as complex", 2, 0)],
)
def test_assert_valid(kind, num_wires, num_work_wires, seed):
    """Run the standard Operator2 validity checks, which include all decomposition rules."""
    unitary_matrix = _random_matrix(num_wires, kind, seed)
    op = qp.BasisRotationPair(unitary_matrix, *_wires(num_wires, num_work_wires))
    qp.testing.assert_valid(op)


@pytest.mark.parametrize("kind", REAL_KINDS + COMPLEX_KINDS)
def test_rule_conditions(kind, seed):
    """Test that the real Givens rule applies to real matrices and the complex Givens rule to
    complex matrices, while the rule with two BasisRotations applies to both."""
    op = qp.BasisRotationPair(_random_matrix(2, kind, seed), [0, 1], [2, 3])
    params = abstractify(op).arguments
    is_complex = kind in COMPLEX_KINDS

    assert _basis_rotation_pair_two_basis_rotations.is_applicable(**params)
    assert _real_basis_rotation_pair_givens_pairs.is_applicable(**params) is not is_complex
    assert _complex_basis_rotation_pair_givens_pairs.is_applicable(**params) is is_complex


class TestDecomposition:
    """Tests for the decomposition rules of BasisRotationPair."""

    @pytest.mark.parametrize("rule, kind", GIVENS_RULE_CASES)
    @pytest.mark.parametrize("num_wires, num_work_wires", [(2, 0), (3, 1)])
    def test_rule_matches_two_basis_rotations(self, rule, kind, num_wires, num_work_wires, seed):
        """Test that each Givens rule maps a random state like BasisRotation on the spin-up wires
        times BasisRotation on the spin-down wires, including the global phase."""
        unitary_matrix = _random_matrix(num_wires, kind, seed)
        wires_up, wires_down, work_wires = _wires(num_wires, num_work_wires)
        num_all_wires = 2 * num_wires + num_work_wires
        in_state = _random_state(num_all_wires, seed)

        @_tape_decompose(gate_set=ELEMENTARY_GATES, fixed_decomps={qp.BasisRotationPair: rule})
        @qp.qnode(qp.device("default.qubit", wires=num_all_wires))
        def circuit():
            qp.StatePrep(in_state, wires=range(num_all_wires))
            qp.BasisRotationPair(unitary_matrix, wires_up, wires_down, work_wires)
            return qp.state()

        expected = _two_basis_rotations_state(unitary_matrix, wires_up, wires_down, in_state)
        assert np.allclose(circuit(), expected)

    def test_two_basis_rotations_rule_emits_basis_rotations(self, seed):
        """Test that the rule applies BasisRotation to the spin-up and to the spin-down wires."""
        unitary_matrix = _random_matrix(3, "complex", seed)
        op = qp.BasisRotationPair(unitary_matrix, [0, 2, 4], [1, 3, 5], [6])

        tape = qp.testing.decomp_rule_to_tape(op, _basis_rotation_pair_two_basis_rotations)

        expected = [
            qp.BasisRotation(unitary_matrix, [0, 2, 4]),
            qp.BasisRotation(unitary_matrix, [1, 3, 5]),
        ]
        assert len(tape.operations) == len(expected)
        for actual_op, expected_op in zip(tape.operations, expected, strict=True):
            qp.assert_equal(actual_op, expected_op)

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize(
        "rule, kind",
        [
            (_real_basis_rotation_pair_givens_pairs, "det-1"),
            (_complex_basis_rotation_pair_givens_pairs, "complex"),
        ],
    )
    def test_givens_rules_emit_givens_pairs(self, rule, kind, seed):
        """Test that the Givens rules emit one GivensPair on ``[up[i], up[j], down[i], down[j]]``
        per Givens rotation ``(i, j)`` of the matrix, that every RZLayer acts on blocks
        ``[up[i], down[i]]``, and that all of them get the work wires."""
        unitary_matrix = _random_matrix(3, kind, seed)
        wires_up, wires_down, work_wires = [6, 1, 4], [0, 3, 2], [9, 8]
        op = qp.BasisRotationPair(unitary_matrix, wires_up, wires_down, work_wires)

        ops = qp.testing.decomp_rule_to_tape(op, rule).operations

        _, givens_list = qp.math.decomposition.givens_decomposition(unitary_matrix)
        expected_wires = [
            Wires([wires_up[i], wires_up[j], wires_down[i], wires_down[j]])
            for _, (i, j) in givens_list
        ]
        givens_pairs = [o for o in ops if isinstance(o, qp.GivensPair)]
        assert [o.arguments["wires"] for o in givens_pairs] == expected_wires

        orbitals = {Wires([up, down]) for up, down in zip(wires_up, wires_down, strict=True)}
        for layer in (o for o in ops if isinstance(o, qp.RZLayer)):
            layer_wires = layer.arguments["wires"]
            assert len(layer_wires) == 2 * len(layer.angles)
            assert all(layer_wires[k : k + 2] in orbitals for k in range(0, len(layer_wires), 2))

        assert all(
            o.work_wires == Wires(work_wires)
            for o in ops
            if isinstance(o, (qp.GivensPair, qp.RZLayer))
        )

    @pytest.mark.parametrize("rule, kind", GIVENS_RULE_CASES)
    def test_resources_are_upper_bounds(self, rule, kind, seed):
        """Test that the declared resources bound the gates that each Givens rule emits, and that
        they are attained for a negative determinant or generic complex entries."""
        op = qp.BasisRotationPair(_random_matrix(3, kind, seed), *_wires(3, 1))

        estimated = rule.compute_resources(**abstractify(op).arguments).gate_counts
        tape = qp.testing.decomp_rule_to_tape(op, rule)
        actual = Counter(abstractify(o) for o in tape.operations)

        assert all(count <= estimated.get(gate, 0) for gate, count in actual.items())
        if kind in ("det-1", "complex"):
            assert actual == estimated

    @pytest.mark.parametrize("num_wires, num_work_wires", [(2, 0), (3, 2)])
    def test_resources_from_shapes(self, num_wires, num_work_wires):
        """Test that the resources of all rules are computed from the abstract shapes of the
        matrix and the wires, and that no rule requests work wires."""
        n = num_wires
        num_givens = n * (n - 1) // 2
        wire_args = {"wires_up": Wire[n], "wires_down": Wire[n], "work_wires": Wire[num_work_wires]}
        givens_pair = qp.GivensPair(Float, Wire[4], Wire[num_work_wires])

        for dtype in (Float, Complex):
            resources = _basis_rotation_pair_two_basis_rotations.compute_resources(
                unitary_matrix=dtype[n, n], **wire_args
            )
            assert resources.gate_counts == {qp.BasisRotation(dtype[n, n], wires=Wire[n]): 2}

        resources = _real_basis_rotation_pair_givens_pairs.compute_resources(
            unitary_matrix=Float[n, n], **wire_args
        )
        assert resources.gate_counts == {abstractify(qp.PauliZ): 2, givens_pair: num_givens}

        resources = _complex_basis_rotation_pair_givens_pairs.compute_resources(
            unitary_matrix=Complex[n, n], **wire_args
        )
        assert resources.gate_counts == {
            qp.RZLayer(Float[n], Wire[2 * n], Wire[num_work_wires]): 1,
            abstractify(qp.GlobalPhase): 1 + num_givens,
            givens_pair: num_givens,
            qp.RZLayer(Float[1], Wire[2], Wire[num_work_wires]): num_givens,
        }

        for rule in RULES:
            assert rule.get_work_wire_spec(unitary_matrix=Float[n, n], **wire_args).total == 0


@pytest.mark.parametrize("num_wires", [2, 3, 4])
def test_phase_gradient_cost(num_wires, seed):
    """Test that, with the Givens-pair rule for a real matrix and the phase gradient rule of
    RZLayer, a BasisRotationPair costs as much as its N(N-1)/2 GivensPairs, i.e., two
    TemporaryANDs outside of one SemiAdder each, and that it requests no further work wires."""
    num_bits = 4
    phase_gradient_wires = {
        "angle_wires": [f"angle_{i}" for i in range(num_bits)],
        "phase_grad_wires": [f"phase_grad_{i}" for i in range(num_bits)],
        "work_wires": [f"adder_work_{i}" for i in range(num_bits - 1)],
    }
    wires_up, wires_down, work_wires = _wires(num_wires, 2)
    op = qp.BasisRotationPair(
        _random_matrix(num_wires, "det-1", seed), wires_up, wires_down, work_wires
    )
    rule = _real_basis_rotation_pair_givens_pairs
    fixed_decomps = {
        qp.BasisRotationPair: rule,
        qp.GivensPair: _givens_pair_rzlayer,
        qp.RZLayer: make_rzlayer_to_phase_gradient_decomp(**phase_gradient_wires),
    }
    gate_set = {
        "PPR_4",
        "PauliX",
        "PauliZ",
        "CNOT",
        "TemporaryAND",
        "Adjoint(TemporaryAND)",
        "SemiAdder",
    }

    [tape], _ = _tape_decompose(
        qp.tape.QuantumScript([op]), gate_set=gate_set, fixed_decomps=fixed_decomps
    )
    assert set(tape.wires) <= set(op.wires).union(*map(Wires, phase_gradient_wires.values()))
    counts = Counter(gate.name for gate in tape.operations)
    num_givens_pairs = num_wires * (num_wires - 1) // 2
    assert (counts["TemporaryAND"], counts["SemiAdder"]) == (2 * num_givens_pairs, num_givens_pairs)
    assert rule.get_work_wire_spec(**op.arguments).total == 0


class TestInterfaces:
    """Tests for the Givens rules under ``jax.jit`` and ``qjit``, where they cannot branch on the
    traced matrix in Python."""

    @pytest.mark.jax
    def test_real_resources_correction_gate(self):
        """Test that the resources of the real Givens rule report the determinant correction as
        PauliZ gates, except under plain jax.jit, where they are PhaseShift gates."""
        import jax

        wire_args = {"wires_up": [0, 1, 2], "wires_down": [3, 4, 5], "work_wires": []}
        givens_pair = qp.GivensPair(Float, Wire[4], Wire[0])

        @jax.jit
        def check_jit_resources(unitary_matrix):
            resources = _real_basis_rotation_pair_givens_pairs.compute_resources(
                unitary_matrix, **wire_args
            )
            assert resources.gate_counts == {abstractify(qp.PhaseShift): 2, givens_pair: 3}
            return unitary_matrix

        check_jit_resources(np.eye(3))

        resources = _real_basis_rotation_pair_givens_pairs.compute_resources(np.eye(3), **wire_args)
        assert resources.gate_counts == {abstractify(qp.PauliZ): 2, givens_pair: 3}

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "rule, kind",
        [
            (_real_basis_rotation_pair_givens_pairs, "det+1"),
            (_real_basis_rotation_pair_givens_pairs, "det-1"),
            (_complex_basis_rotation_pair_givens_pairs, "complex"),
        ],
    )
    def test_jax_jit(self, rule, kind, seed):
        """Test that the Givens rules map a random state like BasisRotation on both spins under
        jax.jit."""
        import jax

        unitary_matrix = _random_matrix(3, kind, seed)
        in_state = _random_state(6, seed)

        @_tape_decompose(gate_set=ELEMENTARY_GATES, fixed_decomps={qp.BasisRotationPair: rule})
        @qp.qnode(qp.device("default.qubit", wires=6))
        def circuit(matrix):
            qp.StatePrep(in_state, wires=range(6))
            qp.BasisRotationPair(matrix, wires_up=[0, 2, 4], wires_down=[1, 3, 5])
            return qp.state()

        res = jax.jit(circuit)(jax.numpy.array(unitary_matrix))
        expected = _two_basis_rotations_state(unitary_matrix, [0, 2, 4], [1, 3, 5], in_state)
        assert np.allclose(res, expected)

    @pytest.mark.catalyst
    @pytest.mark.parametrize(
        "rule, kind",
        [
            (_real_basis_rotation_pair_givens_pairs, "det-1"),
            (_complex_basis_rotation_pair_givens_pairs, "complex"),
        ],
    )
    def test_qjit(self, rule, kind, seed):
        """Test that the Givens rules, compiled with qjit, map a random state like BasisRotation
        on both spins, including the global phase."""
        pytest.importorskip("catalyst")

        unitary_matrix = _random_matrix(3, kind, seed)
        in_state = _random_state(7, seed)

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=ELEMENTARY_GATES, fixed_decomps={qp.BasisRotationPair: rule})
        @qp.qnode(qp.device("lightning.qubit", wires=7))
        def circuit(matrix):
            qp.StatePrep(in_state, wires=range(7))
            qp.BasisRotationPair(matrix, wires_up=[0, 2, 4], wires_down=[1, 3, 5], work_wires=[6])
            return qp.state()

        expected = _two_basis_rotations_state(unitary_matrix, [0, 2, 4], [1, 3, 5], in_state)
        assert np.allclose(circuit(unitary_matrix), expected)

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
"""Tests for the GivensPair template."""

from collections import Counter

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.tape import QuantumScript
from pennylane.templates.subroutines.qchem.double_basis_rotation import (
    _real_double_basis_rotation_givens_pairs,
)
from pennylane.templates.subroutines.qchem.givens_pair import (
    _givens_pair_rzlayer,
    _givens_pair_two_single_excitations,
    _GivensPairFrame,
)
from pennylane.transforms import _tape_decompose
from pennylane.transforms.decompositions import make_rzlayer_to_phase_gradient_decomp
from pennylane.typing import Float, Wire
from pennylane.wires import Wires

RULES = [_givens_pair_two_single_excitations, _givens_pair_rzlayer]

# (phi, wires, work_wires)
SMALL_CASES = [
    (0.4, [0, 1, 2, 3], []),
    (-1.3, [3, 1, 0, 2], [5, 4]),
]

# Registers of b-bit phase gradient rules, and the gates that the frame rule decomposes into with
# the phase gradient rule of RZLayer, plus the PauliZ gates of the DoubleBasisRotation rule for a
# real matrix with determinant -1
NUM_BITS = 4
PHASE_GRADIENT_WIRES = {
    "angle_wires": Wires([f"angle_{i}" for i in range(NUM_BITS)]),
    "phase_grad_wires": Wires([f"phase_grad_{i}" for i in range(NUM_BITS)]),
    "work_wires": Wires([f"adder_work_{i}" for i in range(NUM_BITS - 1)]),
}
PHASE_GRADIENT_GATES = {
    "PPR_4",
    "PauliX",
    "PauliZ",
    "CNOT",
    "TemporaryAND",
    "Adjoint(TemporaryAND)",
    "SemiAdder",
}


def _two_single_excitations_matrix(phi, wires, wire_order):
    """Matrix of SingleExcitation(phi) on the spin-up pair ``wires[:2]`` times SingleExcitation(phi)
    on the spin-down pair ``wires[2:]``."""
    pair = qp.SingleExcitation(phi, wires[:2]) @ qp.SingleExcitation(phi, wires[2:])
    return qp.matrix(pair, wire_order=wire_order)


def _givens_pair_circuit(phi):
    """Apply a GivensPair to ``|1010>``. Each SingleExcitation maps ``|10>`` to
    ``cos(phi/2)|10> - sin(phi/2)|01>``, so that the expectation value of Z on the wires 0 and 2
    is ``cos(phi)**2``."""
    qp.X(0)
    qp.X(2)
    qp.GivensPair(phi, wires=[0, 1, 2, 3])
    return qp.expval(qp.Z(0) @ qp.Z(2))


def _phase_gradient_counts(op, fixed_decomps):
    """Gate counts of ``op``, decomposed with ``fixed_decomps`` into the gates of the phase
    gradient rules, checking that no auxiliary wires other than the phase gradient registers are
    used."""
    [tape], _ = _tape_decompose(
        QuantumScript([op]), gate_set=PHASE_GRADIENT_GATES, fixed_decomps=fixed_decomps
    )
    assert set(tape.wires) <= set(op.wires).union(*PHASE_GRADIENT_WIRES.values())
    return Counter(gate.name for gate in tape.operations)


class TestInitialization:
    """Tests for the arguments of GivensPair."""

    @pytest.mark.parametrize(
        "wires, work_wires", [([0, 1, 2, 3], []), (["pu", "qu", "pd", "qd"], ["t0", "t1"])]
    )
    def test_arguments(self, wires, work_wires):
        """Test that the work wires are part of ``op.wires``."""
        op = qp.GivensPair(0.4, wires, work_wires)

        assert op.arguments["wires"] == Wires(wires)
        assert op.work_wires == Wires(work_wires)
        assert op.wires == Wires(wires + work_wires)

    @pytest.mark.parametrize(
        "wires, work_wires, match",
        [([0, 1, 2], [], "Expected 4 wires"), ([0, 1, 2, 3], [3], "must not overlap")],
    )
    def test_invalid_arguments(self, wires, work_wires, match):
        """Test that a wrong number of wires and overlapping wires raise an error."""
        with pytest.raises(ValueError, match=match):
            qp.GivensPair(0.4, wires, work_wires)

    @pytest.mark.parametrize("phi", [0.4, np.float32(0.4)])
    @pytest.mark.parametrize("num_work_wires", [0, 2])
    def test_abstract_resource_rep(self, phi, num_work_wires):
        """Test that the resource representation is keyed by the number of work wires, not by the
        type of the angle, and can be built from abstract inputs."""
        op = qp.GivensPair(phi, wires=range(4), work_wires=range(4, 4 + num_work_wires))
        abstract_op = qp.GivensPair(Float, Wire[4], Wire[num_work_wires])

        assert abstractify(op) == abstract_op
        assert hash(abstractify(op)) == hash(abstract_op)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("phi, wires, work_wires", SMALL_CASES)
def test_assert_valid(phi, wires, work_wires):
    """Run the standard Operator2 validity checks, which include those of the decomposition rules
    of GivensPair and its adjoint, with their matrices."""
    qp.testing.assert_valid(qp.GivensPair(phi, wires, work_wires))


@pytest.mark.parametrize("phi, wires, work_wires", SMALL_CASES)
def test_matrix_matches_two_single_excitations(phi, wires, work_wires):
    """Test that the matrix, including its global phase, is SingleExcitation on ``[p_up, q_up]``
    times SingleExcitation on ``[p_down, q_down]``, and the identity on the work wires."""
    wire_order = sorted(wires + work_wires)
    op = qp.GivensPair(phi, wires, work_wires)

    expected = _two_single_excitations_matrix(phi, wires, wire_order)
    assert np.allclose(qp.matrix(op, wire_order=wire_order), expected)


class TestDecomposition:
    """Tests for the decomposition rules of GivensPair."""

    def test_rzlayer_rule_emits_one_equal_angle_group(self):
        """Test that the rule emits one RZLayer with the single angle ``phi / 2`` on all four
        wires, with the work wires passed through, in a Clifford frame."""
        wires, work_wires = ["pu", "qu", "pd", "qd"], ["t0", "t1"]
        op = qp.GivensPair(0.4, wires, work_wires)

        [cob] = qp.testing.decomp_rule_to_tape(op, _givens_pair_rzlayer).operations

        qp.assert_equal(cob.compute_op, _GivensPairFrame(wires))
        qp.assert_equal(cob.target_op, qp.RZLayer(np.array([0.2]), wires, work_wires))
        qp.assert_equal(cob.uncompute_op, qp.adjoint(_GivensPairFrame(wires)))

    def test_rzlayer_rule_gates_and_matrix(self):
        """Test that the frame rule, fully decomposed, costs 4 PPR_4, 4 PauliX and 4 RZ and equals
        SingleExcitation on both pairs including the global phase."""
        phi, wires, work_wires = -1.3, [3, 1, 0, 2], [5, 4]
        [tape], _ = _tape_decompose(
            QuantumScript([qp.GivensPair(phi, wires, work_wires)]),
            gate_set={"PPR_4", "PauliX", "RZ"},
            fixed_decomps={qp.GivensPair: _givens_pair_rzlayer},
        )

        assert Counter(op.name for op in tape.operations) == {"PPR_4": 4, "PauliX": 4, "RZ": 4}
        wire_order = wires + work_wires
        expected = _two_single_excitations_matrix(phi, wires, wire_order)
        assert np.allclose(qp.matrix(tape, wire_order=wire_order), expected)

    @pytest.mark.parametrize("num_work_wires", [0, 2])
    def test_resources_from_shapes(self, num_work_wires):
        """Test that the resources of both rules are computed from the abstract shapes ``Float``,
        ``Wire[4]`` and ``Wire[w]``, and that neither rule requests work wires."""
        abstract_args = {"phi": Float, "wires": Wire[4], "work_wires": Wire[num_work_wires]}

        resources = _givens_pair_two_single_excitations.compute_resources(**abstract_args)
        assert resources.gate_counts == {abstractify(qp.SingleExcitation): 2}

        frame = _GivensPairFrame(Wire[4])
        target = qp.RZLayer(Float[1], Wire[4], Wire[num_work_wires])
        resources = _givens_pair_rzlayer.compute_resources(**abstract_args)
        assert resources.gate_counts == {qp.ops.ChangeOpBasis2(frame, target, qp.adjoint(frame)): 1}

        for rule in RULES:
            assert rule.get_work_wire_spec(**abstract_args).total == 0

    @pytest.mark.catalyst
    @pytest.mark.parametrize(
        "rule, gate_set, expected_counts",
        [
            (
                _givens_pair_two_single_excitations,
                {"PauliX", "SingleExcitation"},
                {"PauliX": 2, "SingleExcitation": 2},
            ),
            (_givens_pair_rzlayer, {"PauliX", "PPR_4", "RZ"}, {"PauliX": 6, "PPR_4": 4, "RZ": 4}),
        ],
    )
    def test_qjit(self, rule, gate_set, expected_counts):
        """Test that each rule, compiled with qjit, gives the expected value and gates."""
        pytest.importorskip("catalyst")

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=gate_set, fixed_decomps={qp.GivensPair: rule})
        @qp.qnode(qp.device("lightning.qubit", wires=6))
        def circuit(phi):
            qp.X(0)
            qp.X(2)
            qp.GivensPair(phi, wires=[0, 1, 2, 3], work_wires=[4, 5])
            return qp.expval(qp.Z(0) @ qp.Z(2))

        assert np.isclose(circuit(0.4), np.cos(0.4) ** 2)
        assert qp.analyze(circuit, level=1)(0.4).resources.counts == expected_counts


class TestPhaseGradientCost:
    """Costs of the frame rule with the phase gradient rule of RZLayer, which applies the four
    equal rotations of a GivensPair with one SemiAdder."""

    @pytest.mark.parametrize("phi", [0.4, 0.0])
    def test_givens_pair(self, phi):
        """Test that, with two work wires, a GivensPair costs two TemporaryANDs outside of one
        SemiAdder, also at phi = 0, and that neither rule requests further work wires."""
        rzlayer_rule = make_rzlayer_to_phase_gradient_decomp(**PHASE_GRADIENT_WIRES)
        op = qp.GivensPair(phi, [0, 1, 2, 3], work_wires=[4, 5])

        counts = _phase_gradient_counts(
            op, {qp.GivensPair: _givens_pair_rzlayer, qp.RZLayer: rzlayer_rule}
        )
        assert (counts["TemporaryAND"], counts["SemiAdder"]) == (2, 1)

        [cob] = qp.testing.decomp_rule_to_tape(op, _givens_pair_rzlayer).operations
        assert _givens_pair_rzlayer.get_work_wire_spec(**op.arguments).total == 0
        assert rzlayer_rule.get_work_wire_spec(**cob.target_op.arguments).total == 0

    @pytest.mark.parametrize("num_wires", [2, 3, 4])
    def test_double_basis_rotation(self, num_wires, seed):
        """Test that, with the Givens-pair rule for a real matrix, a DoubleBasisRotation costs as
        much as its N(N-1)/2 GivensPairs."""
        rng = np.random.default_rng(seed)
        q, r = np.linalg.qr(rng.normal(size=(num_wires, num_wires)))
        wires = list(range(2 * num_wires))
        work_wires = [2 * num_wires, 2 * num_wires + 1]
        op = qp.DoubleBasisRotation(q * np.sign(np.diag(r)), wires[::2], wires[1::2], work_wires)
        rule = _real_double_basis_rotation_givens_pairs
        rzlayer_rule = make_rzlayer_to_phase_gradient_decomp(**PHASE_GRADIENT_WIRES)

        counts = _phase_gradient_counts(
            op,
            {
                qp.DoubleBasisRotation: rule,
                qp.GivensPair: _givens_pair_rzlayer,
                qp.RZLayer: rzlayer_rule,
            },
        )
        num_givens_pairs = num_wires * (num_wires - 1) // 2
        assert counts["TemporaryAND"] == 2 * num_givens_pairs
        assert counts["SemiAdder"] == num_givens_pairs
        assert rule.get_work_wire_spec(**op.arguments).total == 0


class TestInterfaces:
    """Tests for the gradient of GivensPair with the autograd and jax interfaces."""

    @pytest.mark.autograd
    def test_autograd(self):
        """Test that the gradient with autograd is the analytic gradient."""
        circuit = qp.QNode(_givens_pair_circuit, qp.device("default.qubit", wires=4))

        grad = qp.grad(circuit)(qp.numpy.array(0.4, requires_grad=True))
        assert np.isclose(grad, -np.sin(0.8))

    @pytest.mark.jax
    def test_jax_jit(self):
        """Test that the value and the gradient with jax under ``jax.jit`` are correct."""
        import jax

        circuit = qp.QNode(_givens_pair_circuit, qp.device("default.qubit", wires=4))

        value, grad = jax.jit(jax.value_and_grad(circuit))(jax.numpy.array(0.4))
        assert np.isclose(value, np.cos(0.4) ** 2)
        assert np.isclose(grad, -np.sin(0.8))


def test_adjoint_negates_angle():
    """Test that the adjoint negates the angle and keeps the wires and work wires."""
    op = qp.GivensPair(0.4, wires=[0, 1, 2, 3], work_wires=[4])

    assert op.has_adjoint
    qp.assert_equal(op.adjoint(), qp.GivensPair(-0.4, wires=[0, 1, 2, 3], work_wires=[4]))
    assert np.allclose(qp.matrix(op.adjoint()), qp.matrix(op).conj().T)

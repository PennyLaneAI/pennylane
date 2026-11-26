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

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.tape import QuantumScript
from pennylane.templates.subroutines.qchem.givens_pair import (
    _givens_pair_rzlayer,
    _givens_pair_two_single_excitations,
    _GivensPairFrame,
)
from pennylane.transforms import _tape_decompose
from pennylane.typing import Float, Wire
from pennylane.wires import Wires

RULES = [_givens_pair_two_single_excitations, _givens_pair_rzlayer]

# (phi, wires, work_wires)
SMALL_CASES = [
    (0.4, [0, 1, 2, 3], []),
    (-1.3, [3, 1, 0, 2], [5, 4]),
]


def _two_single_excitations_matrix(phi, wires, wire_order):
    """Matrix of SingleExcitation(phi) on the spin-up pair ``wires[:2]`` times SingleExcitation(phi)
    on the spin-down pair ``wires[2:]``."""
    pair = qp.SingleExcitation(phi, wires[:2]) @ qp.SingleExcitation(phi, wires[2:])
    return qp.matrix(pair, wire_order=wire_order)


class TestInitialization:
    """Tests for the arguments of GivensPair."""

    @pytest.mark.parametrize("wires, work_wires", [([0, 1, 2, 3], []), ([7, 5, 4, 6], [9, 8])])
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
    of GivensPair and its adjoint."""
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
        wires, work_wires = [3, 1, 0, 2], [5, 4]
        op = qp.GivensPair(0.4, wires, work_wires)

        [cob] = qp.testing.decomp_rule_to_tape(op, _givens_pair_rzlayer).operations

        qp.assert_equal(cob.compute_op, _GivensPairFrame(wires))
        qp.assert_equal(cob.target_op, qp.RZLayer(0.2, wires, work_wires))
        qp.assert_equal(cob.uncompute_op, qp.adjoint(_GivensPairFrame(wires)))

    def test_rzlayer_rule_matrix(self):
        """Test that the frame rule, fully decomposed, equals SingleExcitation on both pairs
        including the global phase."""
        phi, wires, work_wires = -1.3, [3, 1, 0, 2], [5, 4]
        [tape], _ = _tape_decompose(
            QuantumScript([qp.GivensPair(phi, wires, work_wires)]),
            gate_set={"PPR_4", "PauliX", "RZ"},
            fixed_decomps={qp.GivensPair: _givens_pair_rzlayer},
        )

        wire_order = wires + work_wires
        expected = _two_single_excitations_matrix(phi, wires, wire_order)
        assert np.allclose(qp.matrix(tape, wire_order=wire_order), expected)

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


def test_rule_names():
    """Test the names of the decomposition rules, which select them in ``fixed_decomps``."""
    names = {rule.name for rule in qp.list_decomps(qp.GivensPair)}
    assert {"to_single_excitations", "to_rz_layer"} <= names
    assert "negate_angle" in [rule.name for rule in qp.list_decomps("Adjoint(GivensPair)")]


def test_adjoint_negates_angle():
    """Test that the adjoint negates the angle and keeps the wires and work wires."""
    op = qp.GivensPair(0.4, wires=[0, 1, 2, 3], work_wires=[4])

    assert op.has_adjoint
    qp.assert_equal(op.adjoint(), qp.GivensPair(-0.4, wires=[0, 1, 2, 3], work_wires=[4]))
    assert np.allclose(qp.matrix(op.adjoint()), qp.matrix(op).conj().T)

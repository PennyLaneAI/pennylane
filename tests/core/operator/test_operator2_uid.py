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
"""Tests for ``Operator2`` UID generation."""

# pylint: disable=too-few-public-methods

import pytest
from operator2_utils import HybridOp, HybridWireOp, MixedHybridOp, StaticOp

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.core.operator.generate_uid import _serialize_static, generate_uid


class _Opaque:
    """Opaque type for testing."""


class TestSerializeStatic:
    """Tests for the ``_serialize_static`` helper."""

    @pytest.mark.parametrize(
        "value",
        [
            None,
            True,
            1,
            1.5,
            1 + 2j,
            "s",
            [1, True],
            (1, "a"),
            {"k": 1},
            {1, 2},
            frozenset([1]),
            _Opaque(),
        ],
    )
    def test_supported_types(self, value):
        """Test that common static Python types are serialized for UID hashing."""
        ser = _serialize_static(value, "name")
        assert hash(ser)


class TestGenerateUID:
    """Tests for ``generate_uid``."""

    def test_deterministic(self):
        """Test that generating a UID for the same operator twice gives the same result."""
        op = StaticOp("hello", wires=[0])
        assert generate_uid(op) == generate_uid(op)

    def test_wire_labels_do_not_affect_uid(self):
        """Test that operators only differing in their (concrete) wire labels have the same
        UID."""
        op_a = StaticOp("hello", wires=[0])
        op_b = StaticOp("hello", wires=[5])
        assert generate_uid(op_a) == generate_uid(op_b)

    def test_different_static_args_different_uid(self):
        """Test that operators with different static arguments have different UIDs."""
        op_a = StaticOp("hello", wires=[0])
        op_b = StaticOp("world", wires=[0])
        assert generate_uid(op_a) != generate_uid(op_b)

    def test_different_wire_count_different_uid(self):
        """Test that operators with a different number of wires have different UIDs."""
        op_a = StaticOp("hello", wires=[0])
        op_b = StaticOp("hello", wires=[0, 1])
        assert generate_uid(op_a) != generate_uid(op_b)

    def test_same_hybrid_wire_count_same_uid(self):
        """Test that operators with the same number of hybrid wires and the same PyTree
        structure have the same UID, regardless of the concrete wire labels."""
        op_a = HybridWireOp(pytree_wires=[[0, 1]])
        op_b = HybridWireOp(pytree_wires=[[7, 8]])
        assert generate_uid(op_a) == generate_uid(op_b)

    def test_different_hybrid_wire_structure_different_uid(self):
        """Test that operators with the same number of hybrid wires but a different PyTree
        structure have different UIDs."""
        op_a = HybridWireOp(pytree_wires=[[0, 1]])
        op_b = HybridWireOp(pytree_wires=[[0], [1]])
        assert generate_uid(op_a) != generate_uid(op_b)

    def test_abstract_wires_match_concrete_wires(self):
        """Test that a hybrid wire argument given as ``AbstractWires`` produces the same UID as
        an equivalent concrete ``Wires`` argument."""
        op_concrete = HybridWireOp(pytree_wires=[[0, 1]])
        op_abstract = abstractify(op_concrete)
        assert generate_uid(op_concrete) == generate_uid(op_abstract)

    def test_nested_operator_wire_count_affects_uid(self):
        """Test that the wire count of an operator nested inside a (non-wire) hybrid argument
        affects the UID."""
        op_a = HybridOp(ops=StaticOp("x", wires=[0, 1, 2]), wires=[])
        op_b = HybridOp(ops=StaticOp("x", wires=[0, 1]), wires=[])
        assert generate_uid(op_a) != generate_uid(op_b)

    def test_nested_operator_wire_labels_do_not_affect_uid(self):
        """Test that the concrete wire labels of an operator nested inside a (non-wire) hybrid
        argument do not affect the UID."""
        op_a = HybridOp(ops=StaticOp("x", wires=[0, 1, 2]), wires=[])
        op_b = HybridOp(ops=StaticOp("x", wires=[9, 10, 11]), wires=[])
        assert generate_uid(op_a) == generate_uid(op_b)

    def test_mixed_hybrid_op(self):
        """Test that ``generate_uid`` supports operators combining dynamic, non-hybrid wire,
        hybrid wire, and hybrid operator arguments."""
        op_a = MixedHybridOp(
            phi=0.5, ops=StaticOp("x", wires=[0]), pytree_wires=[[1, 2]], wires=[3]
        )
        op_b = MixedHybridOp(
            phi=0.5, ops=StaticOp("x", wires=[9]), pytree_wires=[[4, 5]], wires=[6]
        )
        assert generate_uid(op_a) == generate_uid(op_b)

    def test_adjoint_and_n_ctrls_affect_uid(self):
        """Test that the ``adjoint`` and ``n_ctrls`` flags affect the generated UID."""
        op = StaticOp("hello", wires=[0])
        base_uid = generate_uid(op)
        assert generate_uid(op, adjoint=True) != base_uid
        assert generate_uid(op, n_ctrls=1) != base_uid
        assert generate_uid(op, adjoint=True) != generate_uid(op, n_ctrls=1)

    def test_qp_adjoint_matches_adjoint_flag(self):
        """Test that ``generate_uid`` on a ``qp.adjoint``-wrapped operator matches calling
        ``generate_uid`` on the base operator with ``adjoint=True``."""
        op = StaticOp("hello", wires=[0])
        assert generate_uid(qp.adjoint(op)) == generate_uid(op, adjoint=True)

    def test_qp_ctrl_matches_n_ctrls_flag(self):
        """Test that ``generate_uid`` on a ``qp.ctrl``-wrapped operator matches calling
        ``generate_uid`` on the base operator with the corresponding ``n_ctrls``."""
        op = StaticOp("hello", wires=[0])
        ctrl_op = qp.ctrl(op, control=[1, 2])
        assert generate_uid(ctrl_op) == generate_uid(op, n_ctrls=2)

    def test_qp_adjoint_of_qp_ctrl_matches_both_flags(self):
        """Test that ``generate_uid`` on ``qp.adjoint(qp.ctrl(op))`` matches calling
        ``generate_uid`` on the base operator with both ``adjoint=True`` and the corresponding
        ``n_ctrls``."""
        op = StaticOp("hello", wires=[0])
        wrapped = qp.adjoint(qp.ctrl(op, control=[1, 2]))
        assert generate_uid(wrapped) == generate_uid(op, adjoint=True, n_ctrls=2)

    def test_qp_ctrl_of_qp_adjoint_matches_both_flags(self):
        """Test that ``generate_uid`` on ``qp.ctrl(qp.adjoint(op))`` matches calling
        ``generate_uid`` on the base operator with both ``adjoint=True`` and the corresponding
        ``n_ctrls``, regardless of wrapping order."""
        op = StaticOp("hello", wires=[0])
        wrapped = qp.ctrl(qp.adjoint(op), control=[1, 2])
        assert generate_uid(wrapped) == generate_uid(op, adjoint=True, n_ctrls=2)
        assert generate_uid(wrapped) == generate_uid(qp.adjoint(qp.ctrl(op, control=[1, 2])))

    def test_qp_ctrl_wire_labels_do_not_affect_uid(self):
        """Test that the concrete control wire labels do not affect the UID, only their count,
        for both ``qp.adjoint`` and ``qp.ctrl`` wrapped operators."""
        op = StaticOp("hello", wires=[0])
        ctrl_a = qp.ctrl(qp.adjoint(op), control=[1, 2])
        ctrl_b = qp.ctrl(qp.adjoint(op), control=[9, 10])
        assert generate_uid(ctrl_a) == generate_uid(ctrl_b)

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
from operator2_utils import HybridOp, HybridWireOp, MixedHybridOp, NonParametricOp, StaticOp

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.core.operator.generate_uid import _serialize, generate_uid


class _Opaque:
    """Opaque type for testing."""


class TestSerialize:
    """Tests for the ``_serialize`` helper."""

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
        ser = _serialize(value)
        assert hash(ser)


class TestGenerateUID:
    """Tests for ``generate_uid``."""

    def test_no_static_or_hybrid_returns_none(self):
        """Test that operators without static or hybrid arguments do not get a UID."""
        assert generate_uid(NonParametricOp(wires=[0])) is None
        assert generate_uid(qp.X(0)) is None

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

    def test_non_hybrid_wire_count_does_not_affect_uid(self):
        """Test that the number of non-hybrid wires does not affect the UID."""
        op_a = StaticOp("hello", wires=[0])
        op_b = StaticOp("hello", wires=[0, 1])
        assert generate_uid(op_a) == generate_uid(op_b)

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

    def test_dynamic_values_do_not_affect_uid(self):
        """Test that the concrete values of dynamic arguments do not affect the UID."""
        op_a = MixedHybridOp(
            phi=0.5, ops=StaticOp("x", wires=[0]), pytree_wires=[[1, 2]], wires=[3]
        )
        op_b = MixedHybridOp(
            phi=1.5, ops=StaticOp("x", wires=[0]), pytree_wires=[[1, 2]], wires=[3]
        )
        assert generate_uid(op_a) == generate_uid(op_b)

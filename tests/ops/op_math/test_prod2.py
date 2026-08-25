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
"""Tests for the ``Prod2`` operator."""

# pylint: disable=import-outside-toplevel

import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.op_math.prod2 import Prod2


class TestInitialization:
    """Test construction and basic container behaviour."""

    def test_construction_and_operands(self):
        """Test the operands, length, iteration and indexing of a product."""
        op = Prod2([qp.X(0), qp.Z(1)])
        assert op.operands == (qp.X(0), qp.Z(1))
        assert len(op) == 2
        assert list(op) == [qp.X(0), qp.Z(1)]
        assert op[0] == qp.X(0)

    def test_wires(self):
        """Test that the wires are the union of the operands' wires."""
        op = Prod2([qp.X(0), qp.Z(1), qp.Y(0)])
        assert op.wires == qp.wires.Wires([0, 1])
        assert op.num_wires == 2

    def test_repr(self):
        """Test the string representation of a product."""
        assert repr(Prod2([qp.X(0), qp.Z(1)])) == "X(0) @ Z(1)"

    def test_arithmetic_depth(self):
        """Test the arithmetic depth of flat and nested products."""
        assert Prod2([qp.X(0), qp.Z(1)]).arithmetic_depth == 1
        nested = Prod2([qp.X(0), Prod2([qp.Y(1), qp.Z(2)])])
        assert nested.arithmetic_depth == 2

    def test_mcm_operands_raises(self):
        """Test that a product of mid-circuit measurements raises an error."""
        m = qp.measurements.MidMeasureMP(qp.wires.Wires([0]))
        with pytest.raises(ValueError, match="mid-circuit measurements"):
            Prod2([qp.X(1), m])

    def test_invalid_operands_raises(self):
        """Test that non-operator operands raise a ``TypeError``."""
        with pytest.raises(TypeError, match="operands must be Operator2"):
            Prod2([qp.X(0), 5])

    def test_legacy_operands_raise(self):
        """Test that legacy (op1) operands raise a ``TypeError``."""

        class LegacyOp(qp.core.Operator):  # pylint: disable=too-few-public-methods
            """Dummy operator for testing."""

        with pytest.raises(TypeError, match="Legacy operators"):
            Prod2([LegacyOp(wires=0), qp.Z(wires=1)])


class TestParameters:
    """Test parameter and batching behaviour."""

    def test_data_and_num_params(self):
        """Test that ``data`` and ``num_params`` are gathered from the operands."""
        op = Prod2([qp.RZ(0.5, 0), qp.PhaseShift(0.3, 1), qp.X(2)])
        assert op.num_params == 2
        assert qp.math.allclose(op.data, (0.5, 0.3))

    def test_no_batching(self):
        """Test that an unbatched product has a ``batch_size`` of ``None``."""
        assert Prod2([qp.RZ(0.5, 0), qp.X(1)]).batch_size is None

    def test_batching(self):
        """Test that the batch size is taken from batched operands."""
        op = Prod2([qp.RZ(np.array([0.1, 0.2, 0.3]), 0), qp.X(1)])
        assert op.batch_size == 3

    def test_batching_mismatch_raises(self):
        """Test that mismatched operand batch sizes raise an error."""
        op = Prod2([qp.RZ(np.array([0.1, 0.2]), 0), qp.PhaseShift(np.array([0.1, 0.2, 0.3]), 1)])
        with pytest.raises(ValueError, match="do not match"):
            _ = op.batch_size


class TestMatrix:
    """Test the matrix representations against the legacy ``Prod``."""

    @pytest.mark.parametrize(
        "factors",
        [
            [qp.X(0), qp.Z(1)],
            [qp.RZ(1.23, 0), qp.X(0), qp.Z(1)],
            [qp.Hadamard(0), qp.CNOT([0, 1])],
        ],
    )
    def test_matrix_matches_prod(self, factors):
        """Test that the matrix matches that of the legacy ``Prod``."""
        wire_order = qp.wires.Wires.all_wires([f.wires for f in factors])
        p2 = qp.matrix(Prod2(factors), wire_order=wire_order)
        p1 = qp.matrix(qp.prod(*factors), wire_order=wire_order)
        assert np.allclose(p2, p1)

    def test_sparse_matrix_matches_prod(self):
        """Test that the sparse matrix matches that of the legacy ``Prod``."""
        factors = [qp.X(0), qp.Z(1)]
        p2 = Prod2(factors).sparse_matrix(wire_order=[0, 1]).todense()
        p1 = qp.prod(*factors).sparse_matrix(wire_order=[0, 1]).todense()
        assert np.allclose(p2, p1)

    def test_pauli_rep(self):
        """Test the Pauli representation of a product."""
        assert Prod2([qp.X(0), qp.Z(1)]).pauli_rep == qp.pauli.PauliSentence(
            {qp.pauli.PauliWord({0: "X", 1: "Z"}): 1}
        )


class TestActions:
    """Test decomposition, adjoint, map_wires and queuing."""

    def test_decomposition_reverses_operands(self):
        """Test that the decomposition reverses the operand order."""
        op = Prod2([qp.X(0), qp.Z(1)])
        assert op.decomposition() == [qp.Z(1), qp.X(0)]

    def test_adjoint(self):
        """Test that the adjoint is a ``Prod2`` with the correct matrix."""
        op = Prod2([qp.RZ(0.5, 0), qp.PhaseShift(0.3, 1)])
        adj = op.adjoint()
        assert isinstance(adj, Prod2)
        assert np.allclose(
            qp.matrix(adj, wire_order=[0, 1]),
            qp.matrix(qp.adjoint(qp.prod(qp.RZ(0.5, 0), qp.PhaseShift(0.3, 1))), wire_order=[0, 1]),
        )

    def test_map_wires(self):
        """Test that ``map_wires`` relabels the operands' wires."""
        op = Prod2([qp.X(0), qp.Z(1)]).map_wires({0: 2, 1: 3})
        assert op == Prod2([qp.X(2), qp.Z(3)])

    def test_queuing_dequeues_operands(self):
        """Test that constructing a product dequeues its operands."""
        with qp.queuing.AnnotatedQueue() as q:
            factors = [qp.X(0), qp.Z(1)]
            op = Prod2(factors)
        assert q.queue == [op]


class TestEqualityAndHash:
    """Test equality and hashing, including commuting operands."""

    def test_equal(self):
        """Test that equal products compare equal."""
        assert Prod2([qp.X(0), qp.Z(1)]) == Prod2([qp.X(0), qp.Z(1)])

    def test_equal_commuting_operands(self):
        """Test that products of commuting operands compare equal."""
        # operands on distinct wires commute and are treated as equal
        assert Prod2([qp.X(0), qp.Z(1)]) == Prod2([qp.Z(1), qp.X(0)])

    def test_hash_commuting_operands(self):
        """Test that products of commuting operands hash equally."""
        assert hash(Prod2([qp.X(0), qp.Z(1)])) == hash(Prod2([qp.Z(1), qp.X(0)]))

    def test_not_equal_different_operands(self):
        """Test that products with different operands compare unequal."""
        assert Prod2([qp.X(0), qp.Z(1)]) != Prod2([qp.X(0), qp.Y(1)])


class TestValidity:  # pylint: disable=too-few-public-methods
    """Standard validity checks."""

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_assert_valid(self):
        """Test that ``Prod2`` is defined correctly."""
        # Due to incompatibilities with program capture + assert_valid + Operator2,
        # we skip some checks when program capture is enabled
        if qp.capture.enabled():
            kwargs = {"skip_wire_mapping": True, "skip_new_decomp": True}
        else:
            kwargs = {"skip_capture": True}

        qp.ops.functions.assert_valid(
            Prod2([qp.RZ(0.5, 0), qp.Z(1)]), skip_differentiation=True, **kwargs
        )


class TestIntegration:
    """Test circuit execution and decomposition parity with ``Prod``."""

    def test_execution_matches_prod(self):
        """Test that circuit execution matches the legacy ``Prod``."""
        dev = qp.device("default.qubit", wires=2)

        def make(prod_fn):
            @qp.qnode(dev)
            def circuit():
                qp.Hadamard(0)
                qp.Hadamard(1)
                prod_fn(qp.RZ(0.3, 0), qp.PhaseShift(0.7, 1))
                return qp.expval(qp.Z(0) @ qp.Z(1))

            return circuit

        assert np.allclose(make(lambda *o: Prod2(o))(), make(qp.prod)())

    def test_controlled_decomposition_matches_prod(self):
        """Test that a controlled product's execution matches the legacy ``Prod``."""
        dev = qp.device("default.qubit", wires=3)

        def make(prod_fn):
            @qp.qnode(dev)
            def circuit():
                qp.Hadamard(0)
                qp.ctrl(prod_fn(qp.X(1), qp.Z(2)), control=0)
                return qp.probs(wires=[0, 1, 2])

            return circuit

        assert np.allclose(make(lambda *o: Prod2(o))(), make(qp.prod)())

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize(
        "factors", [[qp.X(1), qp.Z(2)], [qp.RZ(0.3, 1), qp.PhaseShift(0.7, 2)]]
    )
    def test_graph_decomposition_matches_prod(self, factors):
        """Test that the graph-based decomposition matches the legacy ``Prod``."""
        gate_set = {"RZ", "PhaseShift", "CNOT", "X", "Z", "GlobalPhase"}
        wire_order = qp.wires.Wires.all_wires([f.wires for f in factors])

        def decomposed_matrix(op):
            tape = qp.tape.QuantumScript([op])
            [decomposed], _ = qp.transforms.decompose(tape, gate_set=gate_set)
            # the product should have been broken apart into its (decomposed) operands
            assert decomposed.operations
            assert not any(isinstance(o, (Prod2, qp.ops.Prod)) for o in decomposed.operations)
            return qp.matrix(decomposed, wire_order=wire_order)

        assert np.allclose(decomposed_matrix(Prod2(factors)), decomposed_matrix(qp.prod(*factors)))

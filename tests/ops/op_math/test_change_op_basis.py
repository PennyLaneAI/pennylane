# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the ChangeOpBasis arithmetic class of qubit operations
"""
import re

# pylint:disable=protected-access, unused-argument
import pytest

import pennylane as qml
import pennylane.numpy as qnp
from pennylane.decomposition import resource_rep
from pennylane.exceptions import DeviceError
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.ops.op_math import ChangeOpBasis, change_op_basis
from pennylane.wires import Wires

X, Y, Z = qml.PauliX, qml.PauliY, qml.PauliZ

ops = (
    (qml.PauliZ(0), qml.PauliX(1), qml.PauliZ(0)),
    (qml.Hadamard(wires=0), qml.PauliZ(wires=0), qml.Hadamard(wires=0)),
    (qml.CNOT(wires=[0, 1]), qml.RX(1.23, wires=1), qml.CNOT(wires=[0, 1])),
)


def test_basic_validity():
    """Run basic validity checks on a change_op_basis operator."""
    op1 = qml.PauliZ(0)
    op2 = qml.Rot(1.2, 2.3, 3.4, wires=0)
    op3 = qml.PauliZ(0)
    op = qml.change_op_basis(op1, op2, op3)
    qml.ops.functions.assert_valid(op)


class MyOp(qml.RX):  # pylint:disable=too-few-public-methods
    """Variant of qml.RX that claims to not have `adjoint` or a matrix defined."""

    has_matrix = False
    has_adjoint = False
    has_decomposition = False
    has_diagonalizing_gates = False


class TestInitialization:  # pylint:disable=too-many-public-methods
    """Test the initialization."""

    def test_init_change_op_basis_op(self):
        """Test the initialization of a ChangeOpBasis operator."""
        change_op_basis_op = ChangeOpBasis(qml.PauliX(wires=0), qml.RZ(0.23, wires="a"))

        assert change_op_basis_op.wires == Wires((0, "a"))
        assert change_op_basis_op.num_wires == 2
        assert change_op_basis_op.name == "ChangeOpBasis"

        assert change_op_basis_op.data == (0.23,)
        assert change_op_basis_op.parameters == [0.23]
        assert change_op_basis_op.num_params == 1

    def test_hash(self):
        """Testing some situations for the hash property."""
        # test not the same hash if different order
        op1 = qml.change_op_basis(qml.PauliX("a"), qml.PauliY("a"), qml.PauliX(1))
        op2 = qml.change_op_basis(qml.PauliY("a"), qml.PauliX("a"), qml.PauliX(1))
        assert op1.hash != op2.hash

    def test_batch_size(self):
        """Test that batch size returns the batch size of a base operation if it is batched."""
        x = qml.numpy.array([1.0, 2.0, 3.0])
        change_op_basis_op = change_op_basis(qml.PauliX(0), qml.RX(x, wires=0))
        assert change_op_basis_op.batch_size == 3

    def test_batch_size_None(self):
        """Test that the batch size is none if no factors have batching."""
        change_op_basis_op = change_op_basis(qml.PauliX(0), qml.RX(1.0, wires=0))
        assert change_op_basis_op.batch_size is None

    @pytest.mark.parametrize(
        "factors",
        (
            [qml.PauliX(wires=0), qml.PauliZ(wires=0)],
            [qml.PauliX(wires=0), qml.RZ(0.612, "r")],
            [qml.PauliZ(wires=0), qml.PauliX(wires=0)],
            [MyOp(3.1, 0), qml.CNOT([0, 2])],
        ),
    )
    def test_has_adjoint_true_always(self, factors):
        """Test that a change_op_basis of operators that have `has_adjoint=True`
        has `has_adjoint=True` as well."""

        change_op_basis_op = change_op_basis(*factors)
        assert change_op_basis_op.has_adjoint is True

    @pytest.mark.parametrize(
        "factors",
        (
            [qml.PauliX(wires=0), qml.PauliZ(wires=0)],
            [qml.PauliX(wires=0), qml.RZ(0.612, "r")],
            [qml.PauliZ(wires=0), qml.PauliX(wires=0)],
            [MyOp(3.1, 0), qml.CNOT([0, 2])],
        ),
    )
    def test_has_decomposition_true_always(self, factors):
        """Test that a change_op_basis of operators that have `has_decomposition=True`
        has `has_decomposition=True` as well."""

        change_op_basis_op = change_op_basis(*factors)
        assert change_op_basis_op.has_decomposition is True

    def test_has_diagonalizing_gates_false_via_factor(self):
        """Test that a change_op_basis of operators of which one has
        `has_diagonalizing_gates=False` has `has_diagonalizing_gates=False` as well."""

        change_op_basis_op = change_op_basis(MyOp(3.1, 0), qml.PauliX(2))
        assert change_op_basis_op.has_diagonalizing_gates is False


class TestProperties:  # pylint: disable=too-few-public-methods
    """Test class properties."""

    @pytest.mark.parametrize("ops_lst", list(ops))
    def test_adjoint(self, ops_lst):
        """Tests the adjoint of a ChangeOpBasis is correct."""
        change_op_basis_op = ChangeOpBasis(*ops_lst)
        adjoint_ops = []
        for op in change_op_basis_op:
            adjoint_ops.append(op.adjoint())
        for i, op in enumerate(change_op_basis_op.adjoint()):
            assert op == adjoint_ops[i]

    @pytest.mark.parametrize("ops_lst", list(ops))
    def test_is_hermitian(self, ops_lst):
        """Test is_hermitian property updates correctly."""
        middle_op = ops_lst[1]
        change_op = change_op_basis(*ops_lst)
        assert middle_op.is_hermitian == change_op.is_hermitian

    @pytest.mark.parametrize("ops_lst", ops)
    def test_queue_category_ops(self, ops_lst):
        """Test _queue_category property is '_ops' when all factors are `_ops`."""
        change_op_basis_op = change_op_basis(*ops_lst)
        assert change_op_basis_op._queue_category == "_ops"


class TestWrapperFunc:  # pylint: disable=too-few-public-methods
    """Test wrapper function."""

    def test_op_change_op_basis_top_level(self):
        """Test that the top level function constructs an identical instance to one
        created using the class."""

        factors = (qml.PauliX(wires=1), qml.RX(1.23, wires=0), qml.CNOT(wires=[0, 1]))

        change_op_basis_func_op = change_op_basis(*factors)
        change_op_basis_class_op = ChangeOpBasis(*factors)
        qml.assert_equal(change_op_basis_func_op, change_op_basis_class_op)


class TestIntegration:
    """Integration tests for the ChangeOpBasis class."""

    def test_non_supported_obs_not_supported(self):
        """Test that non-supported ops in a measurement process will raise an error."""
        wires = [0, 1]
        dev = qml.device("default.qubit", wires=wires)
        change_op_basis_op = ChangeOpBasis(qml.RX(1.23, wires=0), qml.Identity(wires=1))

        @qml.qnode(dev)
        def my_circ():
            qml.PauliX(0)
            return qml.expval(change_op_basis_op)

        with pytest.raises(
            DeviceError,
            match=re.escape(
                "Measurement expval((Adjoint(RX(1.23, wires=[0]))) @ I(1) @ RX(1.23, wires=[0])) not accepted for analytic simulation on default.qubit"
            ),
        ):
            my_circ()

    def test_params_can_be_considered_trainable(self):
        """Tests that the parameters of a ChangeOpBasis are considered trainable."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, U):
            qml.RX(x, 0)
            return qml.expval(qml.change_op_basis(qml.Hermitian(U, 0), qml.PauliX(1)))

        x = qnp.array(0.1, requires_grad=False)
        U = qnp.array([[1.0, 0.0], [0.0, -1.0]], requires_grad=True)

        tape = qml.workflow.construct_tape(circuit)(x, U)
        assert tape.trainable_params == [1, 2]


class TestDecomposition:

    def test_resource_keys(self):
        """Test that the resource keys of `ChangeOpBasis` are op_reps."""
        assert ChangeOpBasis.resource_keys == frozenset({"compute_op", "target_op", "uncompute_op"})
        change_op_basis_op = ChangeOpBasis(qml.X(0), qml.Y(1), qml.X(2))
        assert change_op_basis_op.resource_params == {
            "compute_op": resource_rep(qml.X),
            "target_op": resource_rep(qml.Y),
            "uncompute_op": resource_rep(qml.X),
        }

    def test_registered_decomp(self):
        """Test that the decomposition of change_op_basis is registered."""

        decomps = qml.decomposition.list_decomps(ChangeOpBasis)

        default_decomp = decomps[0]
        _ops = [qml.X(0), qml.MultiRZ(0.5, wires=(0, 1)), qml.X(0)]
        resources = {qml.resource_rep(qml.X): 2, qml.resource_rep(qml.MultiRZ, num_wires=2): 1}

        resource_obj = default_decomp.compute_resources(
            compute_op=resource_rep(qml.X),
            target_op=resource_rep(qml.MultiRZ, num_wires=2),
            uncompute_op=resource_rep(qml.X),
        )

        assert resource_obj.num_gates == 3
        assert resource_obj.gate_counts == resources

        with qml.queuing.AnnotatedQueue() as q:
            default_decomp(operands=_ops)

        assert q.queue == _ops

    @pytest.mark.parametrize("ops_lst", ops)
    def test_decomposition(self, ops_lst):
        """Test the decomposition of a change_op_basis of operators is a list
        of the provided factors."""
        change_op_basis_op = change_op_basis(*ops_lst)
        decomposition = change_op_basis_op.decomposition()
        true_decomposition = list(ops_lst)  # reversed list of factors

        assert isinstance(decomposition, list)
        for op1, op2 in zip(decomposition, true_decomposition):
            qml.assert_equal(op1, op2)

    @pytest.mark.parametrize("ops_lst", ops)
    def test_decomposition_new(self, ops_lst):
        """Test the qfunc decomposition."""
        change_op_basis_op = change_op_basis(*ops_lst)

        for rule in qml.list_decomps(ChangeOpBasis):
            _test_decomposition_rule(change_op_basis_op, rule)

    @pytest.mark.parametrize("ops_lst", ops)
    @pytest.mark.capture
    def test_decomposition_new_capture(self, ops_lst):
        """Test the qfunc decomposition."""
        change_op_basis_op = change_op_basis(*ops_lst)

        for rule in qml.list_decomps(ChangeOpBasis):
            _test_decomposition_rule(change_op_basis_op, rule)

    @pytest.mark.parametrize("ops_lst", ops)
    def test_controlled_decomposition_new(self, ops_lst):
        """Tests the decomposition rule implemented with the new system."""
        control_wires = [4]
        work_wires = [2, 3]
        op = qml.ops.Controlled(
            change_op_basis(*ops_lst),
            control_wires,
            [1],
            work_wires=work_wires,
        )
        for rule in qml.list_decomps("C(ChangeOpBasis)"):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize("ops_lst", ops)
    @pytest.mark.capture
    def test_controlled_decomposition_new_capture(self, ops_lst):
        """Tests the decomposition rule implemented with the new system."""
        control_wires = [4]
        work_wires = [2, 3]
        op = qml.ops.Controlled(
            change_op_basis(*ops_lst),
            control_wires,
            [1],
            work_wires=work_wires,
        )
        for rule in qml.list_decomps("C(ChangeOpBasis)"):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize("ops_lst", ops)
    def test_decomposition_on_tape(self, ops_lst):
        """Test the decomposition of a change_op_basis of operators is a list
        of the provided factors on a tape."""
        change_op_basis_op = change_op_basis(*ops_lst)
        true_decomposition = list(ops_lst)  # reversed list of factors
        with qml.queuing.AnnotatedQueue() as q:
            change_op_basis_op.decomposition()

        tape = qml.tape.QuantumScript.from_queue(q)
        for op1, op2 in zip(tape.operations, true_decomposition):
            qml.assert_equal(op1, op2)

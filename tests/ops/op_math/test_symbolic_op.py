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
"""Unit tests for the SymbolicOp class of qubit operations"""
from copy import copy

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Operator
from pennylane.ops.op_math import ScalarSymbolicOp, SymbolicOp
from pennylane.wires import Wires


class TempScalar(ScalarSymbolicOp):  # pylint:disable=too-few-public-methods
    """Temporary scalar symbolic op class."""

    _name = "TempScalar"

    @staticmethod
    def _matrix(scalar, mat):
        pass


class TempScalarCopy(ScalarSymbolicOp):  # pylint:disable=too-few-public-methods
    """Copy of temporary scalar symbolic op class."""

    _name = "TempScalarCopy"

    @staticmethod
    def _matrix(scalar, mat):
        pass


def test_intialization():
    """Test initialization for a SymbolicOp"""
    base = Operator("a")

    op = SymbolicOp(base, id="something")

    assert op.base is base
    assert op.hyperparameters["base"] is base
    assert op.id == "something"
    assert op.name == "Symbolic"


def test_copy():
    """Test that a copy of the operator can have its parameters updated independently
    of the original operator."""
    param1 = 1.234
    base = Operator(param1, "a")
    op = SymbolicOp(base)

    copied_op = copy(op)

    assert copied_op.__class__ is op.__class__
    assert copied_op.data == (param1,)

    copied_op.data = (6.54,)
    assert op.data == (param1,)


def test_map_wires():
    """Test the map_wires method."""
    base = Operator("a")
    op = SymbolicOp(base, id="something")
    # pylint:disable=attribute-defined-outside-init,protected-access
    op._pauli_rep = qml.pauli.PauliSentence({qml.pauli.PauliWord({"a": "X"}): 1})
    wire_map = {"a": 5}
    mapped_op = op.map_wires(wire_map=wire_map)
    assert op.wires == Wires("a")
    assert op.base.wires == Wires("a")
    assert mapped_op.wires == Wires(5)
    assert mapped_op.base.wires == Wires(5)
    assert mapped_op.pauli_rep is not op.pauli_rep
    assert mapped_op.pauli_rep == qml.pauli.PauliSentence({qml.pauli.PauliWord({5: "X"}): 1})


class TestProperties:
    """Test the properties of the symbolic op."""

    # pylint:disable=too-few-public-methods

    def test_data(self):
        """Test that the data property for symbolic ops allows for the getting
        and setting of the base operator's data."""
        x = np.array(1.234)

        base = Operator(x, "a")
        op = SymbolicOp(base)

        assert op.data == (x,)

        # update parameters through op
        x_new = np.array(2.345)
        op.data = (x_new,)
        assert base.data == (x_new,)
        assert op.data == (x_new,)

        # update base data updates symbolic data
        x_new2 = np.array(3.45)
        base.data = (x_new2,)
        assert op.data == (x_new2,)

    def test_parameters(self):
        """Test parameter property is a list of the base's trainable parameters."""
        x = np.array(9.876)
        base = Operator(x, "b")
        op = SymbolicOp(base)
        assert op.parameters == [x]

    def test_num_params(self):
        """Test symbolic ops defer num-params to those of the base operator."""
        base = Operator(1.234, 3.432, 0.5490, 8.789453, wires="b")
        op = SymbolicOp(base)

        assert op.num_params == base.num_params == 4

    @pytest.mark.parametrize("has_mat", (True, False))
    def test_has_matrix(self, has_mat):
        """Test that a symbolic op has a matrix if its base has a matrix."""

        class DummyOp(Operator):
            has_matrix = has_mat

        base = DummyOp("b")
        op = SymbolicOp(base)
        assert op.has_matrix == has_mat

    def test_has_matrix_hamiltonian(self):
        """Test that it has a matrix if the base is a hamiltonian."""

        H = qml.Hamiltonian([1.0], [qml.PauliX(0)])
        op = TempScalar(H, 2)
        assert op.has_matrix

    @pytest.mark.parametrize("is_herm", (True, False))
    def test_is_hermitian(self, is_herm):
        """Test that symbolic op is hermitian if the base is hermitian."""

        class DummyOp(Operator):
            is_hermitian = is_herm

        base = DummyOp("b")
        op = SymbolicOp(base)
        assert op.is_hermitian == is_herm

    @pytest.mark.parametrize("queue_cat", ("_ops", None))
    def test_queuecateory(self, queue_cat):
        """Test that a symbolic operator inherits the queue_category from its base."""

        class DummyOp(Operator):
            _queue_category = queue_cat

        op = SymbolicOp(DummyOp("b"))
        assert op._queue_category == queue_cat  # pylint:disable=protected-access

    def test_map_wires(self):
        """Test that base wires can be set through the operator's private `_wires` property."""
        w = qml.wires.Wires("a")
        base = Operator(w)
        op = SymbolicOp(base)

        new_op = op.map_wires(wire_map={"a": "c"})

        assert new_op.wires == Wires("c")

    def test_num_wires(self):
        """Test that the number of wires is the length of the `wires` property, rather
        than the `num_wires` set by the base."""

        t = Operator(wires=(0, 1, 2))
        op = SymbolicOp(t)
        assert op.num_wires == 3

    def test_pauli_rep(self):
        """Test that pauli_rep is None by default"""
        base = Operator("a")
        op = SymbolicOp(base)
        assert op.pauli_rep is None

    def test_raise_error_with_mcm_input(self):
        """Test that symbolic ops of mid-circuit measurements are not supported."""
        mcm = qml.measurements.MidMeasureMP(0)
        with pytest.raises(ValueError, match="Symbolic operators of mid-circuit"):
            _ = SymbolicOp(mcm)


class TestQueuing:
    """Test that Symbolic Operators queue and update base metadata."""

    def test_queuing(self):
        """Test symbolic op queues and updates base metadata."""
        with qml.queuing.AnnotatedQueue() as q:
            base = Operator("a")
            op = SymbolicOp(base)

        assert base not in q
        assert q.queue[0] is op
        assert len(q) == 1

    def test_queuing_base_defined_outside(self):
        """Test symbolic op queues without adding base to the queue if it isn't already in the queue."""

        base = Operator("b")
        with qml.queuing.AnnotatedQueue() as q:
            op = SymbolicOp(base)

        assert len(q) == 1
        assert q.queue[0] is op


class TestScalarSymbolicOp:
    """Tests for the ScalarSymbolicOp class."""

    def test_init(self):
        base = Operator(1.1, wires=[0])
        scalar = 2.2
        op = TempScalar(base, scalar)
        assert isinstance(op.scalar, float)
        assert op.scalar == 2.2
        assert op.data == (2.2, 1.1)

        base = Operator(1.1, wires=[0])
        scalar = [2.2, 3.3]
        op = TempScalar(base, scalar)
        assert isinstance(op.scalar, np.ndarray)
        assert np.all(op.scalar == [2.2, 3.3])
        assert np.all(op.data[0] == op.scalar)
        assert op.data[1] == 1.1

    def test_data(self):
        """Tests the data property."""
        op = TempScalar(Operator(1.1, wires=[0]), 2.2)
        assert op.scalar == 2.2
        assert op.data == (2.2, 1.1)

        # check setting through ScalarSymbolicOp
        op.data = (3.3, 4.4)  # pylint:disable=attribute-defined-outside-init
        assert op.data == (3.3, 4.4)
        assert op.scalar == 3.3
        assert op.base.data == (4.4,)

        # check setting through base
        op.base.data = (5.5,)
        assert op.data == (3.3, 5.5)
        assert op.scalar == 3.3
        assert op.base.data == (5.5,)

    def test_hash(self):
        """Test that a hash correctly identifies ScalarSymbolicOps."""
        op0 = TempScalar(Operator(1.1, wires=[0]), 0.3)
        op1 = TempScalar(Operator(1.1, wires=[0]), 0.3)
        op2 = TempScalar(Operator(1.1, wires=[0]), 0.6)
        op3 = TempScalar(Operator(1.2, wires=[0]), 0.3)
        op4 = TempScalarCopy(Operator(1.1, wires=[0]), 0.3)
        assert op0.hash == op1.hash
        for second_op in [op2, op3, op4]:
            assert op0.hash != second_op.hash

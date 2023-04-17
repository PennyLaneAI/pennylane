# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the new arithmetic operator dunder methods.
"""
import pytest

import pennylane as qml
from pennylane.ops import Sum, Prod, SProd


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    qml.enable_op_arithmetic()
    yield
    qml.disable_op_arithmetic()


pairs_of_ops = [
    (qml.S(0), qml.T(0)),
    (qml.S(0), qml.PauliX(0)),
    (qml.PauliX(0), qml.S(0)),
    (qml.PauliX(0), qml.PauliY(0)),
]


class TestAdd:
    """Test the __add__/__radd__/__sub__ dunders."""

    @pytest.mark.parametrize("op0,op1", pairs_of_ops)
    def test_add_operators(self, op0, op1):
        """Tests adding two operators, observable or not."""
        op = op0 + op1
        assert isinstance(op, Sum)
        assert qml.equal(op[0], op0)
        assert qml.equal(op[1], op1)

    @pytest.mark.parametrize("op0,op1", pairs_of_ops)
    def test_sub_operators(self, op0, op1):
        """Tests subtracting two operators."""
        op = op0 - op1
        assert isinstance(op, Sum)
        assert qml.equal(op[0], op0)
        assert isinstance(op[1], SProd)
        assert op[1].scalar == -1
        assert qml.equal(op[1].base, op1)

    def test_op_with_scalar(self):
        """Tests adding/subtracting ops with scalars."""
        x0 = qml.PauliX(0)
        for op in [x0 + 1, 1 + x0]:
            assert isinstance(op, Sum)
            assert qml.equal(op[0], x0)
            assert isinstance(op[1], SProd)
            assert op[1].scalar == 1
            assert qml.equal(op[1].base, qml.Identity(0))

        x1 = qml.PauliX(1)
        op = x1 - 1.1
        assert isinstance(op, Sum)
        assert qml.equal(op[0], x1)
        assert isinstance(op[1], SProd)
        assert op[1].scalar == -1.1
        assert qml.equal(op[1].base, qml.Identity(1))

        op = 1.1 - x1  # will use radd
        assert isinstance(op, Sum)
        assert isinstance(op[0], SProd)
        assert op[0].scalar == -1
        assert qml.equal(op[0].base, x1)
        assert isinstance(op[1], SProd)
        assert op[1].scalar == 1.1
        assert qml.equal(op[1].base, qml.Identity(1))

    def test_adding_many_does_not_auto_simplify(self):
        """Tests that adding more than two operators creates nested Sums."""
        op0, op1, op2 = qml.S(0), qml.T(0), qml.PauliZ(0)
        op = op0 + op1 + op2
        assert isinstance(op, Sum)
        assert len(op) == 2
        assert isinstance(op[0], Sum)
        assert qml.equal(op[0][0], op0)
        assert qml.equal(op[0][1], op1)
        assert qml.equal(op[1], op2)


class TestMul:
    """Test the __mul__/__rmul__ dunders."""

    @pytest.mark.parametrize("scalar", [0, 1, 1.1, 1 + 2j, [3, 4j]])
    def test_mul(self, scalar):
        """Tests multiplying an operator by a scalar coefficient works as expected."""
        base = qml.PauliX(0)
        for op in [scalar * base, base * scalar]:
            assert isinstance(op, SProd)
            assert qml.math.allequal(op.scalar, scalar)
            assert qml.equal(op.base, base)

    @pytest.mark.parametrize("scalar", [1, 1.1, 1 + 2j, qml.numpy.array([3, 4j])])
    def test_div(self, scalar):
        """Tests diviing an operator by a scalar coefficient works as expected."""
        base = qml.PauliX(0)
        op = base / scalar
        assert isinstance(op, SProd)
        assert qml.math.allequal(op.scalar, 1 / scalar)
        assert qml.equal(op.base, base)

    def test_mul_does_not_auto_simplify(self):
        """Tests that multiplying an SProd with a scalar creates nested SProds."""
        op = 2 * qml.PauliX(0)
        nested = 0.5 * op
        assert isinstance(nested, SProd)
        assert nested.scalar == 0.5
        assert qml.equal(nested.base, op)


class TestMatMul:
    """Test the __matmul__/__rmatmul__ dunders."""

    @pytest.mark.parametrize("op0,op1", pairs_of_ops)
    def test_matmul_operators(self, op0, op1):
        """Tests matrix-multiplication of two operators, observable or not."""
        op = op0 @ op1
        assert isinstance(op, Prod)
        assert qml.equal(op[0], op0)
        assert qml.equal(op[1], op1)

    def test_mul_does_not_auto_simplify(self):
        """Tests that matrix-multiplying a Prod with another operator creates nested Prods."""
        op0, op1, op2 = qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)
        op = op0 @ op1 @ op2
        assert isinstance(op, Prod)
        assert len(op) == 2
        assert isinstance(op[0], Prod)
        assert qml.equal(op[0], op0 @ op1)
        assert qml.equal(op[1], op2)


def test_op_arithmetic_toggle():
    """Tests toggling op arithmetic on and off."""
    qml.enable_op_arithmetic()
    assert qml.active_op_arithmetic()
    assert isinstance(qml.PauliX(0) @ qml.PauliZ(1), Prod)

    qml.disable_op_arithmetic()
    assert not qml.active_op_arithmetic()
    assert isinstance(qml.PauliX(0) @ qml.PauliZ(1), qml.operation.Tensor)

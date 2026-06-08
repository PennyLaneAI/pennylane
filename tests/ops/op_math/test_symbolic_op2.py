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

"""Unit tests for the SymbolicOp2 class."""

import pennylane as qp
from pennylane.core.operator import Operator2
from pennylane.ops.op_math.symbolicop2 import SymbolicOp2
from pennylane.wires import Wires


class RX2(Operator2):
    """A simplified RX operator that inherit from Operator2."""

    dynamic_argnames = ("theta",)

    def __init__(self, theta, wires):  # pylint: disable=useless-parent-delegation
        super().__init__(theta, wires)

    @property
    def is_verified_hermitian(self) -> bool:
        return True

    @staticmethod
    def compute_matrix(theta):
        return qp.math.array(
            [
                [qp.math.cos(theta / 2), -1j * qp.math.sin(theta / 2)],
                [-1j * qp.math.sin(theta / 2), qp.math.cos(theta / 2)],
            ]
        )


class CustomSymbolicOp(SymbolicOp2):  # pylint: disable=too-few-public-methods
    """A custom symbolic operator class."""

    dynamic_argnames = ("val",)

    compilable_argnames = ("toggle",)

    def __init__(self, val, base, toggle: bool = True):  # pylint: disable=useless-parent-delegation
        super().__init__(val, base, toggle)


def test_initialization():
    """Tests that a subclass of SymbolicOp2 can be initialized."""

    base_op = RX2(0.5, wires=0)
    op = CustomSymbolicOp(0.6, base_op, True)

    assert op.wires == Wires([0])
    assert op.base == base_op
    assert op.val == 0.6
    assert op.toggle is True
    assert op.arithmetic_depth == 1
    assert op.is_verified_hermitian
    assert op.has_matrix
    assert not op.has_sparse_matrix
    assert op.arguments == {"val": 0.6, "base": base_op, "toggle": True}
    assert op.dynamic_args == {"val": 0.6}
    assert op.compilable_args == {"toggle": True}
    assert op.hybrid_args == {"base": base_op}
    assert op.wire_args == {}


def test_queuing():
    """Tests that a symbolic op is queued properly."""

    with qp.queuing.AnnotatedQueue() as q:
        base_op = RX2(0.5, wires=0)
        op = CustomSymbolicOp(0.6, base_op, True)

    assert len(q.queue) == 1
    assert q.queue[0] == op


def test_flatten_unflatten():
    """Tests that a symbolic op can be flattened and unflattened."""

    base_op = RX2(0.5, wires=1)
    op = CustomSymbolicOp(0.6, base_op, True)

    data, struct = qp.pytrees.flatten(op)
    assert data == [0.6, 0.5, 1]

    reconstructed = qp.pytrees.unflatten(data, struct)
    qp.assert_equal(reconstructed, op)

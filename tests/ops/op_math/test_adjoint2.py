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

"""Tests for the Adjoint2 class."""

import numpy as np

import pennylane as qp
from pennylane.core.operator import Operator2
from pennylane.ops.op_math.adjoint2 import Adjoint2
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
    def compute_matrix(theta, wires):  # pylint: disable=unused-argument
        return qp.math.array(
            [
                [qp.math.cos(theta / 2), -1j * qp.math.sin(theta / 2)],
                [-1j * qp.math.sin(theta / 2), qp.math.cos(theta / 2)],
            ]
        )

    def generator(self):
        return qp.Hamiltonian([-0.5], [qp.X(wires=self.wires)])

    def adjoint(self):
        return RX2(-self.theta, wires=self.wires)

    def simplify(self):
        theta = self.theta % (4 * np.pi)
        if qp.math.isclose(theta, 0.0):
            return qp.Identity()
        return RX2(theta, wires=self.wires)


def test_initialization():
    """Tests initializing an Adjoint2 operator."""

    op = qp.adjoint(RX2(0.5, wires=0))
    assert isinstance(op, Adjoint2)

    op = qp.adjoint(RX2)(0.5, wires=0)
    assert isinstance(op, Adjoint2)

    def inner():
        RX2(0.5, wires=0)

    with qp.queuing.AnnotatedQueue() as q:
        qp.adjoint(inner)()

    assert len(q.queue) == 1
    op = q.queue[0]
    assert isinstance(op, Adjoint2)


def test_attributes():
    """Tests the basic properties of an Adjoint2 operator."""

    op = qp.adjoint(RX2(-0.5, wires=0))

    assert op.base == RX2(-0.5, wires=0)
    assert op.wires == Wires([0])
    assert op.arithmetic_depth == 1
    assert op.is_verified_hermitian
    assert op.has_matrix
    assert not op.has_sparse_matrix
    assert op.arguments == {"base": RX2(-0.5, wires=0)}
    assert op.wire_args == {}
    assert op.has_generator


def test_methods():
    """Tests the operator methods."""

    op = qp.adjoint(RX2(0.5, wires=0))

    expected_matrix = (
        np.array(
            [
                [np.cos(0.5 / 2), 1j * np.sin(0.5 / 2)],
                [1j * np.sin(0.5 / 2), np.cos(0.5 / 2)],
            ]
        ),
    )
    assert qp.math.allclose(op.matrix(), expected_matrix)
    assert op.generator().simplify() == qp.Hamiltonian([0.5], [qp.X(0)])
    assert op.simplify() == RX2(np.pi * 4 - 0.5, wires=0)


def test_flatten_unflatten():
    """Tests that an adjoint op can be flattened and unflattened."""

    base_op = RX2(0.5, wires=1)
    op = qp.adjoint(base_op)

    data, struct = qp.pytrees.flatten(op)
    assert data == [0.5, 1]

    reconstructed = qp.pytrees.unflatten(data, struct)
    qp.assert_equal(reconstructed, op)

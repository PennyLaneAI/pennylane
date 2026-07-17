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

"""Tests for the Adjoint2 class."""

import numpy as np
import pytest
from scipy import sparse

import pennylane as qp
from pennylane.core.operator import Operator2
from pennylane.decomposition.decomposition_rule import register_resources
from pennylane.ops.op_math.adjoint import Adjoint, AdjointOperation
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.wires import Wires

# pylint: disable=unused-argument,arguments-differ,useless-parent-delegation


class SX2(Operator2):
    """A new SX gate."""

    wire_sizes = (1,)

    is_verified_hermitian = True

    def __init__(self, wires):
        super().__init__(wires)

    @property
    def pauli_rep(self):
        return qp.pauli.PauliSentence(
            {
                qp.pauli.PauliWord({self.wires[0]: "I"}): (0.5 + 0.5j),
                qp.pauli.PauliWord({self.wires[0]: "X"}): (0.5 - 0.5j),
            }
        )

    @staticmethod
    def compute_matrix(wires):
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])

    @staticmethod
    def compute_sparse_matrix(wires, format="csr"):
        return 0.5 * sparse.csr_matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]).asformat(format=format)

    @staticmethod
    def compute_eigvals(wires):
        return np.array([1, 1j])

    @staticmethod
    def compute_diagonalizing_gates(wires):
        return [qp.Hadamard(wires=wires)]


class RX2(Operator2):
    """A simplified RX operator that inherit from Operator2."""

    dynamic_argnames = ("theta",)

    wire_sizes = (1,)

    is_verified_hermitian = True

    def __init__(self, theta, wires):
        super().__init__(theta, wires)

    @staticmethod
    def compute_matrix(theta, wires):
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
    assert not op.has_diagonalizing_gates
    assert op.pauli_rep is None

    op2 = qp.adjoint(SX2(0))
    assert op2.has_diagonalizing_gates
    assert op2.has_sparse_matrix
    assert op2.pauli_rep == qp.pauli.PauliSentence(
        {
            qp.pauli.PauliWord({0: "I"}): (0.5 - 0.5j),
            qp.pauli.PauliWord({0: "X"}): (0.5 + 0.5j),
        }
    )
    assert op2.has_adjoint


def test_old_decomp_integartion():
    """Tests that adjoint2 is compatible with the old decomposition convention."""

    @register_resources({qp.RX: 1})
    def _sx_to_rx(wires):
        qp.RX(np.pi / 2, wires=wires)

    with qp.decomposition.local_decomps():

        qp.add_decomps(SX2, _sx_to_rx)
        op = qp.adjoint(SX2(0))
        assert op.has_decomposition
        assert op.decomposition() == [qp.adjoint(qp.RX(np.pi / 2, wires=[0]))]


def test_parameter_frequencies():
    """Tests that adjoint2 ops have the correct parameter frequencies."""

    op = qp.adjoint(RX2(-0.5, wires=0))
    assert qp.gradients.parameter_frequencies(op) == [(1,)]


def test_instance_check():
    """Tests that Adjoint2 objects are considered instances of Adjoint."""

    op = qp.adjoint(RX2(-0.5, wires=0))
    assert isinstance(op, Adjoint)
    assert isinstance(op, AdjointOperation)


def test_methods():
    """Tests the operator methods."""

    op = qp.adjoint(RX2(0.5, wires=0))

    c, s = np.cos(0.5 / 2), 1j * np.sin(0.5 / 2)
    expected_matrix = np.array([[c, s], [s, c]])
    assert qp.math.allclose(op.matrix(), expected_matrix)
    assert op.generator().simplify() == qp.Hamiltonian([0.5], [qp.X(0)])
    assert op.simplify() == RX2(np.pi * 4 - 0.5, wires=0)
    assert op.adjoint() == RX2(0.5, wires=0)

    op2 = qp.adjoint(SX2(0))
    expected_matrix = 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]])
    assert qp.math.allclose(op2.matrix(), expected_matrix)
    assert qp.math.allclose(op2.sparse_matrix(), expected_matrix)
    qp.assert_equal(op2.simplify(), Adjoint2(SX2(0)))
    assert op2.diagonalizing_gates() == [qp.H(0)]
    assert qp.math.allclose(op2.eigvals(), [1, -1j])


def test_flatten_unflatten():
    """Tests that an adjoint op can be flattened and unflattened."""

    base_op = RX2(0.5, wires=1)
    op = qp.adjoint(base_op)

    data, struct = qp.pytrees.flatten(op)
    assert data == [0.5, 1]

    reconstructed = qp.pytrees.unflatten(data, struct)
    qp.assert_equal(reconstructed, op)


def test_representation():
    """Tests that repr and label of an adjoint operator."""

    base_op = RX2(0.5, wires=1)
    op = qp.adjoint(base_op)
    assert repr(op) == "Adjoint(RX2(theta=0.5, wires=[1]))"
    assert op.label() == "RX2†"
    assert op.name == "Adjoint(RX2)"

    nested_op = qp.adjoint(op)
    assert repr(nested_op) == "Adjoint(Adjoint(RX2(theta=0.5, wires=[1])))"
    assert nested_op.label() == "(RX2†)†"
    assert nested_op.name == "Adjoint(Adjoint(RX2))"


def test_adjoint_equality():
    """Tests comparing adjoint operators."""

    base_op = RX2(0.5, wires=1)

    class OldOp(qp.core.operator.Operator1):  # pylint: disable=too-few-public-methods
        pass

    another_base = OldOp(wires=0)

    assert qp.adjoint(base_op) == Adjoint2(base_op)

    with pytest.raises(AssertionError, match="different base operations"):
        qp.assert_equal(qp.adjoint(base_op), qp.adjoint(another_base))

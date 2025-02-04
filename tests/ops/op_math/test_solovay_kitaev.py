# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for Solovay-Kitaev decomposition's implementation."""

import math

import pytest
import scipy as sp

import pennylane as qml
from pennylane.ops.op_math.decompositions.solovay_kitaev import (
    _approximate_set,
    _contains_SU2,
    _group_commutator_decompose,
    _quaternion_transform,
    _SU2_transform,
    sk_decomposition,
)


@pytest.mark.parametrize(
    ("op", "matrix", "phase"),
    [
        (
            qml.Identity(0),  # computed manually
            qml.math.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]),
            0.0,
        ),
        (
            qml.T(0),  # computed manually
            qml.math.array(
                [[0.92387953 - 0.38268343j, 0.0 + 0.0j], [0.0 + 0.0j, 0.92387953 + 0.38268343j]]
            ),
            0.3926990817,
        ),
        (
            qml.Hadamard(0),  # computed manually
            qml.math.array(
                [
                    [-0.70710678j, -0.70710678j],
                    [-0.70710678j, 0.70710678j],
                ]
            ),
            1.5707963268,
        ),
    ],
)
def test_su2_transform(op, matrix, phase):
    """Test the functionality to create SU(2) representation"""
    su2_matrix, su2_phase = _SU2_transform(op.matrix())
    assert qml.math.allclose(su2_matrix, matrix)
    assert qml.math.isclose(su2_phase, phase)


@pytest.mark.parametrize(
    "op,quaternion",
    [
        (qml.Identity(0), qml.math.array([1.0, 0.0, 0.0, 0.0])),
        (qml.T(0), qml.math.array([0.92387953, 0.0, 0.0, 0.38268343])),
        (qml.Hadamard(0), qml.math.array([0, 0.707106781, 0, 0.707106781])),
    ],
)
def test_quaternion_transform(op, quaternion):
    """Test the functionality to create quaternion representation"""
    su2_matrix, _ = _SU2_transform(op.matrix())
    op_quaternion = _quaternion_transform(su2_matrix)
    assert qml.math.allclose(op_quaternion, quaternion)


def test_contains_SU2():
    """Test the functionality to check an operation is in an existing approximate set"""
    op = qml.prod(qml.T(0), qml.T(0), qml.T(0), qml.T(0), qml.T(0))
    target = _SU2_transform(op.matrix())[0]

    approx_ids, _, _, approx_vec = _approximate_set(("T", "T*", "H"), max_length=3)

    exists, quaternion, _ = _contains_SU2(target, approx_vec)
    assert exists

    result = [qml.adjoint(qml.T(0)), qml.adjoint(qml.T(0)), qml.adjoint(qml.T(0))]

    node_points = qml.math.array(approx_vec)
    seq_node = qml.math.array([quaternion])
    _, [index] = sp.spatial.KDTree(node_points).query(seq_node, workers=-1)

    assert approx_ids[index] == result


def test_approximate_sets():
    """Test the functionality to create approximate set"""
    # Check length of approximated sets with gate depths
    approx_set1, _, _, _ = _approximate_set((), max_length=0)
    approx_set2, _, _, _ = _approximate_set(("t", "h"), max_length=1)
    assert len(approx_set1) == 0
    assert len(approx_set2) == 2

    # Verify exist-check reduce redundancy via GP-like summation
    approx_set3, _, _, _ = _approximate_set(("t", "t*", "h"), max_length=3)
    approx_set4, _, _, _ = _approximate_set(("t", "t*", "h"), max_length=5)
    assert len(approx_set3) == 21  # should be <  39 = 3 + 3x3 + 3x3x3
    assert len(approx_set4) == 84  # should be < 263 = 3 + ... + 3x3x3x3x3


@pytest.mark.parametrize(
    ("op"),
    [
        qml.U3(0.0, 0.0, 0.0, wires=[0]),
        qml.U3(1.0, 2.0, 3.0, wires=[1]),
        qml.U3(-1.0, 2.0, 3.0, wires=["b"]),
        qml.U3(1.0, 2.0, -3.0, wires=[0]),
    ],
)
def test_group_commutator_decompose(op):
    """Test group commutator decomposition method for SU(2)"""

    su2_matrix, _ = _SU2_transform(qml.matrix(op))

    v_hat, w_hat = _group_commutator_decompose(su2_matrix)
    decomposed_matrix = (
        v_hat
        @ w_hat
        @ qml.math.conj(qml.math.transpose(v_hat))
        @ qml.math.conj(qml.math.transpose(w_hat))
    )

    assert qml.math.allclose(decomposed_matrix, su2_matrix)


@pytest.mark.parametrize(
    ("op", "max_depth"),
    [
        (qml.RX(math.pi / 42, wires=[1]), 5),
        (qml.RY(math.pi / 7, wires=["a"]), 5),
        (qml.prod(*[qml.RX(1.0, "a"), qml.T("a")]), 5),
        (qml.prod(*[qml.T(0), qml.Hadamard(0)] * 5), 5),
        (qml.RZ(-math.pi / 2, wires=[1]), 1),
        (qml.adjoint(qml.S(wires=["a"])), 1),
        (qml.PhaseShift(5 * math.pi / 2, wires=[0]), 1),
        (qml.PhaseShift(-3 * math.pi / 4, wires=["b"]), 1),
    ],
)
def test_solovay_kitaev(op, max_depth):
    """Test Solovay-Kitaev decomposition method with specified max-depth"""

    with qml.queuing.AnnotatedQueue() as q:
        gates = sk_decomposition(op, epsilon=1e-4, max_depth=max_depth, basis_set=("T", "T*", "H"))
    assert q.queue == gates

    matrix_sk = qml.matrix(qml.tape.QuantumScript(gates))

    assert qml.math.allclose(qml.matrix(op), matrix_sk, atol=1e-2)
    assert qml.prod(*gates, lazy=False).wires == op.wires


@pytest.mark.parametrize(
    ("basis_length", "basis_set"),
    [(10, ("T*", "T", "H")), (8, ("H", "T")), (10, ("H", "S", "T"))],
)
def test_solovay_kitaev_with_basis_gates(basis_length, basis_set):
    """Test Solovay-Kitaev decomposition method with additional basis information provided."""
    op = qml.PhaseShift(math.pi / 4, 0)
    gates = sk_decomposition(op, 1e-4, max_depth=3, basis_set=basis_set, basis_length=basis_length)

    matrix_sk = qml.prod(*reversed(gates)).matrix()

    assert qml.math.allclose(qml.matrix(op), matrix_sk, atol=1e-5)


@pytest.mark.parametrize(
    "op,query_count",
    [
        (qml.Hadamard(0), 1),
        (qml.prod(qml.Hadamard(0), qml.T(0)), 1),
        (qml.prod(qml.Hadamard(0), qml.T(0), qml.RX(1e-9, 0)), 1),
        (qml.RZ(1.23, 0), 27),
        (qml.prod(*[qml.T(0), qml.Hadamard(0)] * 5), 1),
    ],
)
def test_close_approximations_do_not_go_deep(op, query_count, mocker):
    """Test that the recursive solver is only used when necessary."""
    basis_set = ("H", "T", "T*")
    basis_length = 10
    _ = _approximate_set(basis_set, max_length=basis_length)  # pre-compute so spy is accurate

    spy = mocker.spy(sp.spatial.KDTree, "query")
    _ = sk_decomposition(op, 1e-4, max_depth=3, basis_set=basis_set, basis_length=basis_length)
    assert spy.call_count == query_count


@pytest.mark.parametrize("epsilon", [2e-2, 3e-2, 7e-2])
def test_epsilon_value_respected(epsilon):
    """Test that resulting decompositions are within an epsilon value."""
    op = qml.RX(1.234, 0)
    gates = sk_decomposition(op, epsilon, max_depth=5)
    matrix_sk = qml.prod(*reversed(gates)).matrix()
    assert qml.math.norm(qml.matrix(op)[0] - matrix_sk[0]) < epsilon


def test_epsilon_value_effect():
    """Test that different epsilon values create different decompositions."""
    op = qml.RZ(math.pi / 5, 0)
    decomp_with_error = sk_decomposition(op, 9e-2, max_depth=5)
    decomp_less_error = sk_decomposition(op, 1e-2, max_depth=5)
    assert len(decomp_with_error) < len(decomp_less_error)


def test_exception():
    """Test operation wire exception in Solovay-Kitaev"""
    op = qml.SingleExcitation(1.0, wires=[1, 2])

    with pytest.raises(
        ValueError,
        match=r"Operator must be a single qubit operation",
    ):
        sk_decomposition(op, epsilon=1e-4, max_depth=1)

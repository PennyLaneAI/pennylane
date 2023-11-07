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

from pennylane.transforms.decompositions.clifford_t.solovay_kitaev import (
    _SU2_transform,
    _quaternion_transform,
    _contains_SU2,
    _approximate_set,
    _group_commutator_decompose,
    sk_decomposition,
)


@pytest.mark.parametrize(
    ("op", "su2", "quaternion"),
    [
        (
            qml.Identity(0),  # computed manually
            (qml.math.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]), 0.0),
            qml.math.array([1.0, 0.0, 0.0, 0.0]),
        ),
        (
            qml.T(0),  # computed manually
            (
                qml.math.array(
                    [[0.92387953 - 0.38268343j, 0.0 + 0.0j], [0.0 + 0.0j, 0.92387953 + 0.38268343j]]
                ),
                0.3926990817,
            ),
            qml.math.array([0.92387953, 0.0, 0.0, 0.38268343]),
        ),
        (
            qml.Hadamard(0),  # computed manually
            (
                qml.math.array(
                    [
                        [-0.70710678j, -0.70710678j],
                        [-0.70710678j, 0.70710678j],
                    ]
                ),
                1.5707963268,
            ),
            qml.math.array([0, 0.707106781, 0, 0.707106781]),
        ),
    ],
)
def test_su2_and_quaternion_transforms(op, su2, quaternion):
    """Test the functionality to create SU(2) and quaternion representations"""
    su2_matrix, su2_phase = _SU2_transform(op.matrix())
    assert qml.math.allclose(su2_matrix, su2[0])
    assert qml.math.isclose(su2_phase, su2[1])

    op_quaternion = _quaternion_transform(su2_matrix)
    assert qml.math.allclose(op_quaternion, quaternion)


def test_contains_SU2():
    """Test the functionality to check an operation is in an existing approximate set"""
    op = qml.prod(qml.T(0), qml.T(0), qml.T(0), qml.T(0), qml.T(0))
    target = _SU2_transform(op.matrix())[0]

    approx_ids, _, approx_vec = _approximate_set(("t", "tdg", "h"), max_length=3)

    assert _contains_SU2(target, approx_vec)[0]

    result = [qml.adjoint(qml.T(0)), qml.adjoint(qml.T(0)), qml.adjoint(qml.T(0))]

    node_points = qml.math.array(approx_vec)
    seq_node = qml.math.array([_quaternion_transform(target)])
    _, index = sp.spatial.KDTree(node_points).query(seq_node, workers=-1)

    assert approx_ids[index[0]] == result


def test_approximate_sets():
    """Test the functionality to create approximate set"""
    # Check length of approximated sets with gate depths
    approx_set1, _, _ = _approximate_set((), max_length=0)
    approx_set2, _, _ = _approximate_set(("t", "h"), max_length=1)
    assert len(approx_set1) == 1
    assert len(approx_set2) == 3

    # Verify exist-check reduce redundancy via GP-like summation
    approx_set3, _, _ = _approximate_set(("t", "tdg", "h"), max_length=3)
    approx_set4, _, _ = _approximate_set(("t", "tdg", "h"), max_length=5)
    assert len(approx_set3) < 39 and len(approx_set3) == 22  # 3 + 3x3 + 3x3x3
    assert len(approx_set4) < 263 and len(approx_set4) == 89  # 3 + ... + 3x3x3x3x3


@pytest.mark.parametrize(
    ("op"),
    [
        qml.U3(1.0, 2.0, 3.0, wires=[1]),
        qml.U3(-1.0, 2.0, 3.0, wires=["b"]),
        qml.U3(1.0, 2.0, -3.0, wires=[0]),
    ],
)
def test_group_commutator_decompose(op):
    """Test group commutator deocompose method for SU(2)"""

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
    ("op"),
    [
        qml.RX(math.pi / 42, wires=[1]),
        qml.RZ(math.pi / 128, wires=[0]),
        qml.RY(math.pi / 7, wires=["a"]),
    ],
)
def test_solovay_kitaev(op):
    """Test Solovay-Kitaev decomposition method"""

    with qml.queuing.AnnotatedQueue() as q:
        gates = sk_decomposition(op, depth=5, basis_set=("T", "Tdg", "H"))
    assert q.queue == gates

    matrix_sk, phase = _SU2_transform(qml.prod(*reversed(gates)).matrix())

    assert qml.math.allclose(qml.matrix(op), matrix_sk * qml.math.exp(-1j * phase), atol=1e-2)
    assert qml.prod(*gates, lazy=False).wires == op.wires


@pytest.mark.parametrize(
    ("basis_depth", "basis_set"),
    [(10, ()), (8, (("H", "T"))), (10, ("H", "S", "T"))],
)
def test_solovay_kitaev_with_basis_gates(basis_depth, basis_set):
    """Test Solovay-Kitaev decomposition method without providing approximation set"""
    op = qml.PhaseShift(math.pi / 4, 0)
    gates = sk_decomposition(op, depth=3, basis_set=basis_set, basis_depth=basis_depth)

    matrix_sk = qml.prod(*gates[::-1]).matrix()

    assert qml.math.allclose(qml.matrix(op), matrix_sk, atol=1e-5)


def test_exception():
    """Test operation wire exception in Solovay-Kitaev"""
    op = qml.SingleExcitation(1.0, wires=[1, 2])

    with pytest.raises(
        ValueError,
        match=r"Operator must be a single qubit operation",
    ):
        sk_decomposition(op, depth=1)

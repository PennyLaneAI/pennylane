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
"""Tests for Solovay-Kitaev implementation."""

import math
import functools

import pytest
import pennylane as qml

from pennylane.transforms.decompositions.clifford_plus_t.solovay_kitaev import (
    GateSet,
    TreeSet,
    _group_commutator_decompose,
    _unitary_bloch,
    sk_approximate_set,
    sk_decomposition,
)


def test_gate_sets():
    """Test various functionalities for the GateSet object"""

    gateset1 = GateSet([])
    gateset2 = GateSet([qml.PauliX(0), qml.PauliZ(0)])

    # test names and string representation
    assert gateset1.name == "" and gateset2.name == "PauliXPauliZ"
    assert str(gateset1) == "Gates: [], Matrix: [[1. 0.]\n [0. 1.]]"
    assert str(gateset2) == "Gates: [PauliX, PauliZ], Matrix: [[ 0 -1]\n [ 1  0]]"

    # test lengths and getter
    assert len(gateset1) == 0 and len(gateset2) == 2
    assert isinstance(gateset2[0], qml.PauliX)

    # test append functionality
    assert gateset1 != gateset2
    assert gateset1.append(qml.PauliX(0)).append(qml.PauliZ(0)) == gateset2

    # test copy functionality
    assert gateset2.copy() == gateset2

    gateset3 = GateSet([qml.PauliZ(0), qml.PauliX(0)])

    # test adjoint functionality
    assert gateset2 != gateset3
    assert gateset2.adjoint() == gateset3

    # test dot functionality
    matrix = gateset2.dot(gateset3).matrix
    assert qml.math.allclose(matrix, qml.math.eye(2))

    gateset4 = GateSet.from_matrix(matrix)

    # test SU(2) amd SO(3) matrix generators
    assert gateset4 != GateSet.from_matrix(1j * matrix)
    assert qml.math.allclose(gateset1.get_SU2_matrix(matrix)[0], gateset4.su2_matrix)
    assert qml.math.allclose(gateset1.get_SO3_matrix(matrix), gateset4.so3_matrix)

    # test equality with phase difference
    gateset5 = gateset4.copy()
    gateset5.so3_matrix = gateset4.get_SO3_matrix(
        gateset4.su2_matrix * qml.math.exp(0.5j * math.pi)
    )
    assert gateset5 != gateset4

    # test .from_matrix exception
    with pytest.raises(
        ValueError,
        match=r"Matrix should be of shape \(2, 2\), got \(3, 3\).",
    ):
        gateset4 = GateSet.from_matrix(qml.math.eye(3))


def test_tree_sets():
    """Test various functionalities for the TreeSet object"""

    treeset1 = TreeSet(GateSet(), [], [])
    treeset2 = TreeSet(GateSet(), ["h", "t"], [])

    # Check process_node method expands the approximated gates with depth
    assert len(treeset1.basic_approximation(depth=0)) == 1
    assert len(treeset2.basic_approximation(depth=1)) == 3

    # Verify check_nodes help reduce redundancy via GP-like summation
    treeset3 = TreeSet(GateSet(), ["h", "t", "tdg"], [])
    assert len(treeset3.basic_approximation(depth=3)) < 39  # 3 + 3x3 + 3x3x3
    assert len(treeset3.basic_approximation(depth=5)) < 263  # 3 + ... + 3x3x3x3x3

    # Verify sequence check in check_nodes method
    treeset4 = TreeSet(GateSet(), ["x", "z"], [])
    seqs1 = GateSet([qml.PauliX(0)])
    seqs2 = [GateSet([]), GateSet([qml.PauliX(0)]), GateSet([qml.PauliX(0), qml.PauliZ(0)])]
    assert not treeset4.check_nodes(seqs1, seqs2)


class TestSolovayKitaev:
    """Test for Solovay-Kitaev functionality"""

    @pytest.mark.parametrize(
        ("op"),
        [
            qml.U3(1.0, 2.0, 3.0, wires=[1]),
            qml.U3(-1.0, 2.0, 3.0, wires=["b"]),
            qml.U3(1.0, 2.0, -3.0, wires=[0]),
        ],
    )
    def test_group_commutator_decompose(self, op):
        """Test group commutator deocompose method for SU(2)"""

        matrix = qml.matrix(op)
        su2_matrix = matrix / qml.math.sqrt((1 + 0j) * qml.math.linalg.det(matrix))

        v_hat, w_hat = _group_commutator_decompose(su2_matrix)
        decomposed_matrix = (
            v_hat
            @ w_hat
            @ qml.math.conj(qml.math.transpose(v_hat))
            @ qml.math.conj(qml.math.transpose(w_hat))
        )

        assert qml.math.allclose(decomposed_matrix, su2_matrix)

    @pytest.mark.parametrize(
        ("op", "axis", "angle"),
        [
            (qml.Identity(wires=[0]), [0.0, 0.0, 1.0], 0.0),
            (qml.S(wires=[0]), [0.0, 0.0, 1.0], math.pi / 2),
            (qml.Hadamard(wires=["a"]), [1 / math.sqrt(2), 0.0, 1 / math.sqrt(2)], math.pi),
        ],
    )
    def test_unitary_bloch(self, op, axis, angle):
        """Test for finding bloch vector and angle for an SU(2)"""

        matrix = qml.matrix(op)
        su2_matrix = matrix / qml.math.sqrt((1 + 0j) * qml.math.linalg.det(matrix))

        op_axis, op_angle = _unitary_bloch(su2_matrix)

        assert qml.math.allclose(op_axis, axis) and qml.math.allclose(op_angle, angle)

    def test_sk_approximate_set(self):
        """Test for building approximate sets"""

        approx_set1 = sk_approximate_set(basis_depth=5)
        approx_set2 = sk_approximate_set(basis_set=["t", "tdg", "h"], basis_depth=5)

        assert approx_set1 == approx_set2
        assert len(approx_set1) < 263 and len(approx_set2) < 263  # 3 + ... + 3x3x3x3x3

    @pytest.mark.parametrize(
        ("op"),
        [
            qml.RX(math.pi / 42, wires=[1]),
            qml.RZ(math.pi / 128, wires=[0]),
            qml.RY(math.pi / 7, wires=["a"]),
        ],
    )
    def test_solovay_kitaev(self, op):
        """Test Solovay-Kitaev decomposition method"""

        approximate_set = sk_approximate_set(
            basis_set=[
                "t",
                "tdg",
                "h",
            ],
            basis_depth=10,
        )
        gates = sk_decomposition(op, depth=5, approximate_set=approximate_set)

        matrix_sk = functools.reduce(lambda x, y: x @ y, map(qml.matrix, gates))
        matrix_sk /= qml.math.sqrt((1 + 0j) * qml.math.linalg.det(matrix_sk))

        assert qml.math.allclose(qml.matrix(op), matrix_sk, atol=1e-2)

    @pytest.mark.parametrize(
        ("basis_depth", "basis_set"),
        [(10, ()), (8, (["h", "t"])), (10, ["h", "t"])],
    )
    def test_solovay_kitaev_without_approx_set(self, basis_depth, basis_set):
        """Test Solovay-Kitaev decomposition method without providing approximation set"""
        op = qml.PhaseShift(math.pi/4, 0)
        gates = sk_decomposition(op, depth=1, basis_depth=basis_depth, basis_set=basis_set)

        matrix_sk = functools.reduce(lambda x, y: x @ y, map(qml.matrix, gates))
        #matrix_sk /= qml.math.sqrt((1 + 0j) * qml.math.linalg.det(matrix_sk))

        assert qml.math.allclose(qml.matrix(op), matrix_sk, atol=1e-2)

    def test_exception(self):
        """Test operation wire exception in Solovay-Kitaev"""
        op = qml.SingleExcitation(1.0, wires=[1, 2])

        with pytest.raises(
            ValueError,
            match=r"Operator must be a single qubit operation",
        ):
            sk_decomposition(op, depth=1)

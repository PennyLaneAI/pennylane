# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np

import pennylane as qml
from pennylane.wires import Wires
from pennylane import apply

from gate_data import I

from pennylane.transforms.optimization import single_qubit_fusion


class TestSingleQubitFusion:
    """Test fusion of groups of adjacent single-qubit gates."""

    @pytest.mark.parametrize(
        ("op_list"),
        [
            ([qml.Rot(0.0, 0.5, 0.0, wires=0), qml.RY(-0.5, wires=0)]),
            ([qml.RZ(0.3, wires=3), qml.Rot(0.0, 0.0, -0.3, wires=3)]),
            ([qml.Rot(0.0, 0.0, -0.3, wires=3), qml.RZ(0.3, wires=3)]),
            ([qml.RZ(-np.pi / 2, wires=0), qml.S(wires=0)]),
        ],
    )
    def test_single_qubit_fusion_zero_angles(self, op_list):
        """Test a sequence of single-qubit operations that cancel upon fusion actually do so."""

        def qfunc():
            for op in op_list:
                apply(op)

        transformed_qfunc = single_qubit_fusion(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 0

    @pytest.mark.parametrize(
        ("op_list"),
        [
            ([qml.Rot(0.0, 0.0, 0.0, wires=3), qml.Rot(0.3, 0.4, 0.5, wires=3)]),
            ([qml.Rot(0.0, 0.3, 0.0, wires=0), qml.RY(0.3, wires=0)]),
            ([qml.RZ(0.1, wires=0), qml.RX(0.2, wires=0), qml.Hadamard(wires=0)]),
            (
                [
                    qml.Rot(0.1, 0.2, 0.3, wires="a"),
                    qml.PhaseShift(0.2, wires="a"),
                    qml.S(wires="a"),
                    qml.RY(0.2, wires="a"),
                ]
            ),
        ],
    )
    def test_single_qubit_fusion_same_wire(self, op_list):
        """Test a sequence of single-qubit operations on the same wire are fused to a single Rot"""

        def qfunc():
            for op in op_list:
                apply(op)

        transformed_qfunc = single_qubit_fusion(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 1
        assert ops[0].name == "Rot"

        # Compare matrix representations (up to a global phase)
        expected_mat = I
        for op in op_list:
            expected_mat = np.dot(op.matrix, expected_mat)

        obtained_mat = ops[0].matrix

        # Check equivalence by seeing if U^\dagger U is I up to a phase
        mat_product = np.dot(np.conj(obtained_mat.T), expected_mat)
        mat_product /= mat_product[0, 0]

        assert qml.math.allclose(mat_product, I)

    @pytest.mark.parametrize(
        ("op_list_q1,op_list_q2"),
        [
            (
                [qml.Rot(0.0, 0.0, 0.0, wires=3), qml.Rot(0.3, 0.4, 0.5, wires=3)],
                [qml.RZ(0.2, wires=2), qml.T(wires=2), qml.PauliX(wires=2)],
            ),
            (
                [qml.PauliX(wires="a"), qml.PauliZ(wires="a"), qml.RY(-0.2, wires="a")],
                [qml.RZ(0.2, wires="b"), qml.S(wires="b"), qml.PhaseShift(0.5, wires="b")],
            ),
            (
                [qml.PauliY(wires=0), qml.PauliZ(wires=0), qml.PauliY(wires=0)],
                [qml.SX(wires="b"), qml.RX(0.0, wires="b")],
            ),
        ],
    )
    def test_single_qubit_fusion_two_wires(self, op_list_q1, op_list_q2):
        """Test that independent sequences of rotations on different qubits properly fuse."""

        def qfunc():
            for op in op_list_q1 + op_list_q2:
                apply(op)

        transformed_qfunc = single_qubit_fusion(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 2

        assert ops[0].name == "Rot"
        assert ops[1].name == "Rot"

        # Compare matrix representations (up to a global phase)
        expected_mat_q1 = I
        for op in op_list_q1:
            expected_mat_q1 = np.dot(op.matrix, expected_mat_q1)

        expected_mat_q2 = I
        for op in op_list_q2:
            expected_mat_q2 = np.dot(op.matrix, expected_mat_q2)

        # Check equivalence by seeing if U^\dagger U is I up to a phase
        mat_product_1 = np.dot(np.conj(ops[0].matrix.T), expected_mat_q1)
        mat_product_1 /= mat_product_1[0, 0]

        mat_product_2 = np.dot(np.conj(ops[1].matrix.T), expected_mat_q2)
        mat_product_2 /= mat_product_2[0, 0]

        assert qml.math.allclose(mat_product_1, I)
        assert qml.math.allclose(mat_product_2, I)

    @pytest.mark.parametrize(
        ("op_list_1,op_list_2"),
        [
            (
                [qml.Rot(0.0, 0.0, 0.0, wires=3), qml.Rot(0.3, 0.4, 0.5, wires=3)],
                [qml.RZ(0.2, wires=3), qml.T(wires=3), qml.PauliX(wires=3)],
            ),
            (
                [qml.PauliX(wires="a"), qml.PauliZ(wires="a"), qml.RY(-0.2, wires="a")],
                [qml.RZ(0.2, wires="a"), qml.S(wires="a"), qml.PhaseShift(0.5, wires="a")],
            ),
            (
                [qml.PauliY(wires=0), qml.PauliZ(wires=0), qml.PauliY(wires=0)],
                [qml.SX(wires=0), qml.RX(0.0, wires=0)],
            ),
        ],
    )
    def test_single_qubit_fusion_blocked_by_cnot(self, op_list_1, op_list_2):
        """Test that sequences of rotations on each side of a CNOT fuse independently."""

        def qfunc():
            for op in op_list_1:
                apply(op)

            qml.CNOT(wires=[op_list_1[0].wires[0], "c"])

            for op in op_list_2:
                apply(op)

        transformed_qfunc = single_qubit_fusion(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Rot"
        assert ops[0].wires[0] == op_list_1[0].wires[0]

        assert ops[1].name == "CNOT"
        assert ops[1].wires == Wires([op_list_1[0].wires[0], "c"])

        assert ops[2].name == "Rot"
        assert ops[2].wires[0] == op_list_2[0].wires[0]

        # Compare matrix representations (up to a global phase)
        expected_mat_1 = I
        for op in op_list_1:
            expected_mat_1 = np.dot(op.matrix, expected_mat_1)

        expected_mat_2 = I
        for op in op_list_2:
            expected_mat_2 = np.dot(op.matrix, expected_mat_2)

        # Check equivalence by seeing if U^\dagger U is I up to a phase
        mat_product_1 = np.dot(np.conj(ops[0].matrix.T), expected_mat_1)
        mat_product_1 /= mat_product_1[0, 0]

        mat_product_2 = np.dot(np.conj(ops[2].matrix.T), expected_mat_2)
        mat_product_2 /= mat_product_2[0, 0]

        assert qml.math.allclose(mat_product_1, I)
        assert qml.math.allclose(mat_product_2, I)

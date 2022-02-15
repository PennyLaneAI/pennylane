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

import pennylane as qml
from pennylane.transforms.optimization.optimization_utils import (
    find_next_gate,
    _yzy_to_zyz,
    fuse_rot_angles,
)

from utils import check_matrix_equivalence
from pennylane.transforms.get_unitary_matrix import get_unitary_matrix


sample_op_list = [
    qml.Hadamard(wires="a"),
    qml.CNOT(wires=["a", "b"]),
    qml.Toffoli(wires=[0, 1, "b"]),
    qml.Hadamard(wires="c"),
]


class TestFindNextGate:
    @pytest.mark.parametrize(
        ("wires,op_list,next_gate_idx"),
        [
            ("a", sample_op_list, 0),
            ("b", sample_op_list, 1),
            ("e", sample_op_list, None),
            ([0, 2], sample_op_list, 2),
            ([0, 1], sample_op_list, 2),
            ("c", sample_op_list, 3),
        ],
    )
    def test_find_next_gate(self, wires, op_list, next_gate_idx):
        """Test find_next_gate correctly identifies the next gate in a list of operations that share any number of wires."""
        assert find_next_gate(qml.wires.Wires(wires), op_list) == next_gate_idx


class TestRotGateFusion:
    """Test that utility functions for fusing two qml.Rot gates function as expected."""

    @pytest.mark.parametrize(
        ("angles"),
        [([0.15, 0.25, -0.90]), ([0.0, 0.0, 0.0]), ([0.15, 0.25, -0.90]), ([0.05, -1.34, 4.12])],
    )
    def test_yzy_to_zyz(self, angles):
        """Test that a set of rotations of the form YZY is correctly converted
        to a sequence of the form ZYZ."""

        def original_ops():
            qml.RY(angles[0], wires=0),
            qml.RZ(angles[1], wires=0),
            qml.RY(angles[2], wires=0),

        compute_matrix = get_unitary_matrix(original_ops, [0])
        product_yzy = compute_matrix()

        z1, y, z2 = _yzy_to_zyz(angles)

        def transformed_ops():
            qml.RZ(z1, wires=0)
            qml.RY(y, wires=0)
            qml.RZ(z2, wires=0)

        compute_transformed_matrix = get_unitary_matrix(transformed_ops, [0])
        product_zyz = compute_transformed_matrix()

        assert check_matrix_equivalence(product_yzy, product_zyz)

    @pytest.mark.parametrize(
        ("angles_1", "angles_2"),
        [
            ([0.15, 0.25, -0.90], [-0.5, 0.25, 1.3]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.1, 0.2, 0.3], [0.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.2, 0.4, -0.5]),
            ([0.15, 0.25, -0.90], [-0.15, -0.25, 0.9]),
            ([0.05, -1.34, 4.12], [-0.8, 0.2, 0.12]),
            ([0.05, 0.0, 4.12], [-0.8, 0.2, 0.12]),
            ([0.05, -1.34, 0.0], [-0.8, 0.2, 0.12]),
            ([0.05, 0.0, 0.1], [-0.2, 0.0, 0.12]),
            ([0.05, 0.0, 0.0], [0.0, 0.0, 0.12]),
            ([0.05, 0.2, 0.0], [0.0, -0.6, 0.12]),
            ([0.05, -1.34, 4.12], [0.0, 0.2, 0.12]),
            ([0.05, -1.34, 4.12], [0.3, 0.0, 0.12]),
        ],
    )
    def test_full_rot_fusion_autograd(self, angles_1, angles_2):
        """Test that the fusion of two Rot gates has the same effect as
        applying the Rots sequentially."""

        def original_ops():
            qml.Rot(*angles_1, wires=0)
            qml.Rot(*angles_2, wires=0)

        compute_matrix = get_unitary_matrix(original_ops, [0])
        matrix_expected = compute_matrix()

        fused_angles = fuse_rot_angles(angles_1, angles_2)
        matrix_obtained = qml.Rot(*fused_angles, wires=0).get_matrix()

        assert check_matrix_equivalence(matrix_expected, matrix_obtained)
